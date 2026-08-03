"""Policy-to-human and human-to-human full-body metrics."""

from __future__ import annotations

import numpy as np

from mm_evalkit.stats.rsa import compute_rsm as _compute_rsm
from mm_evalkit.stats.rsa import rsa

from .numerics import finite_mean, paired_pearson, paired_rmse
from .state import GaitAnalysis, HumanToHumanMetrics, PerSubjectMetrics, PopulationMetrics


def compute_rsm(subject_muscle_data):
    """Return the channel-by-channel correlation matrix for one profile."""
    return _compute_rsm(subject_muscle_data)


def _empty_per_channel(names, aggregate):
    """One empty list per channel, plus the across-channels aggregate key."""
    return {**{name: [] for name in names}, aggregate: []}


def compute_human_to_model_metrics_population(analysis: GaitAnalysis) -> None:
    """Score each policy population profile against every human subject."""
    channels = analysis.channels
    human = analysis.human
    n_human_subjects = human.n_subjects

    for run in analysis.policies:
        metrics = PopulationMetrics(
            joint_rmse=_empty_per_channel(channels.joint_names, "all_joints"),
            joint_corr=_empty_per_channel(channels.joint_names, "all_joints"),
            muscle_corr=_empty_per_channel(channels.muscle_names, "all_muscles"),
            rsa_matrix=np.full((1, n_human_subjects), np.nan),
            rsa_matrix_pearson=np.full((1, n_human_subjects), np.nan),
        )
        run.population = metrics

        policy_avg_joint_gaits = run.profiles.avg_joint_gaits
        policy_avg_muscle_gaits = run.profiles.avg_muscle_gaits
        policy_muscle_rsm = run.profiles.population_muscle_rsm

        # A valid rollout can still contain no retained gait cycles at this position.
        if not np.isfinite(policy_avg_joint_gaits).any():
            print(f"  Skipping metrics computation for {run.label} - no retained gait cycles.")
            metrics.avg_joint_rmse = dict.fromkeys(metrics.joint_rmse, np.nan)
            metrics.avg_joint_corr = dict.fromkeys(metrics.joint_corr, np.nan)
            metrics.avg_muscle_corr = dict.fromkeys(metrics.muscle_corr, np.nan)
            metrics.avg_rsa = np.nan
            metrics.avg_rsa_pearson = np.nan
            continue

        for joint_idx, joint_name in enumerate(channels.joint_names):
            metrics.joint_rmse[joint_name] = [
                paired_rmse(policy_avg_joint_gaits[joint_idx], human.joint_data[subj_idx][joint_idx])
                for subj_idx in range(n_human_subjects)
            ]
            metrics.joint_corr[joint_name] = [
                paired_pearson(policy_avg_joint_gaits[joint_idx], human.joint_data[subj_idx][joint_idx])
                for subj_idx in range(n_human_subjects)
            ]

        metrics.joint_rmse["all_joints"] = [
            finite_mean([metrics.joint_rmse[joint][subject] for joint in channels.joint_names])
            for subject in range(n_human_subjects)
        ]
        metrics.joint_corr["all_joints"] = [
            finite_mean([metrics.joint_corr[joint][subject] for joint in channels.joint_names])
            for subject in range(n_human_subjects)
        ]

        for muscle_idx, muscle_name in enumerate(channels.muscle_names):
            metrics.muscle_corr[muscle_name] = [
                paired_pearson(policy_avg_muscle_gaits[muscle_idx], human.muscle_data[subj_idx][muscle_idx])
                for subj_idx in range(n_human_subjects)
            ]

        metrics.muscle_corr["all_muscles"] = [
            finite_mean([metrics.muscle_corr[muscle][subject] for muscle in channels.muscle_names])
            for subject in range(n_human_subjects)
        ]

        for subj_idx in range(n_human_subjects):
            rsa_spearman, rsa_pearson = rsa(policy_muscle_rsm, human.muscle_rsms[subj_idx])
            metrics.rsa_matrix[0, subj_idx] = rsa_spearman
            metrics.rsa_matrix_pearson[0, subj_idx] = rsa_pearson

        metrics.avg_joint_rmse = {joint: finite_mean(values) for joint, values in metrics.joint_rmse.items()}
        metrics.avg_joint_corr = {joint: finite_mean(values) for joint, values in metrics.joint_corr.items()}
        metrics.avg_muscle_corr = {muscle: finite_mean(values) for muscle, values in metrics.muscle_corr.items()}
        metrics.avg_rsa = finite_mean(metrics.rsa_matrix)
        metrics.avg_rsa_pearson = finite_mean(metrics.rsa_matrix_pearson)


def compute_human_to_model_metrics_per_subject(analysis: GaitAnalysis) -> None:
    """Score each policy subject profile against the matching human subject."""
    if not analysis.subject_based:
        print("Subject-based plotting is disabled. Set subject_based=True to enable.")
        return
    channels = analysis.channels
    human = analysis.human
    n_human_subjects = human.n_subjects

    for run in analysis.policies:
        metrics = PerSubjectMetrics(
            joint_rmse=_empty_per_channel(channels.joint_names, "all_joints"),
            joint_corr=_empty_per_channel(channels.joint_names, "all_joints"),
            muscle_corr=_empty_per_channel(channels.muscle_names, "all_muscles"),
            rsa_matrix=np.full((1, n_human_subjects), np.nan),
            rsa_matrix_pearson=np.full((1, n_human_subjects), np.nan),
        )
        run.per_subject = metrics

        # A validated rollout can still have zero retained cycles at this gait.
        if run.profiles.subject_ids is None:
            print(f"  Skipping per-subject metrics computation for {run.label} - no retained gait cycles.")
            metrics.subjects_without_gait_cycles = list(human.subjects)
            continue

        subjects_with_data = np.unique(run.profiles.subject_ids)
        metrics.subjects_without_gait_cycles = np.setdiff1d(human.subjects, subjects_with_data).tolist()

        for subj_idx in subjects_with_data:
            # Subject ids are not 1..N; human data is stored in cohort order, so map
            # the subject id to its position in that order.
            subj_pos = human.subjects.index(int(subj_idx))
            human_joint_gaits = human.joint_data[subj_pos]
            human_muscle_gaits = human.muscle_data[subj_pos]
            human_rsm = human.muscle_rsms[subj_pos]

            policy_joint_gaits = run.profiles.per_subject_joint[subj_idx]
            policy_muscle_gaits = run.profiles.per_subject_muscle[subj_idx]

            for joint_idx, joint_name in enumerate(channels.joint_names):
                metrics.joint_rmse[joint_name].append(
                    paired_rmse(policy_joint_gaits[joint_idx], human_joint_gaits[joint_idx])
                )
                metrics.joint_corr[joint_name].append(
                    paired_pearson(policy_joint_gaits[joint_idx], human_joint_gaits[joint_idx])
                )

            for muscle_idx, muscle_name in enumerate(channels.muscle_names):
                metrics.muscle_corr[muscle_name].append(
                    paired_pearson(policy_muscle_gaits[muscle_idx], human_muscle_gaits[muscle_idx])
                )

            # Per-subject RSA: one RSM per subject (computed before any group averaging)
            # on both sides, then correlated over the lower off-diagonal triangle.
            rsa_spearman, rsa_pearson = rsa(run.profiles.per_subject_muscle_rsm[subj_idx], human_rsm)
            metrics.rsa_matrix[0, subj_pos] = rsa_spearman
            metrics.rsa_matrix_pearson[0, subj_pos] = rsa_pearson
            metrics.rsa.append(rsa_spearman)
            metrics.rsa_pearson.append(rsa_pearson)
            metrics.subjects_with_rsa_data.append(subj_idx)

        # Averages cover only the subjects with retained gait cycles.
        if len(subjects_with_data) == 0:
            continue
        metrics.joint_rmse["all_joints"] = [
            finite_mean([metrics.joint_rmse[joint][subject] for joint in channels.joint_names])
            for subject in range(len(subjects_with_data))
        ]
        metrics.joint_corr["all_joints"] = [
            finite_mean([metrics.joint_corr[joint][subject] for joint in channels.joint_names])
            for subject in range(len(subjects_with_data))
        ]
        metrics.muscle_corr["all_muscles"] = [
            finite_mean([metrics.muscle_corr[muscle][subject] for muscle in channels.muscle_names])
            for subject in range(len(subjects_with_data))
        ]
        metrics.avg_rsa = finite_mean(metrics.rsa) if metrics.rsa else np.nan
        metrics.avg_rsa_pearson = finite_mean(metrics.rsa_pearson) if metrics.rsa_pearson else np.nan


def compute_human_to_human_metrics(analysis: GaitAnalysis) -> None:
    """Score every pair of human subjects: the between-subject reference band.

    Per-channel lists carry one value per subject *pair*; the RSA matrices are
    subject x subject, with Spearman as the primary score and Pearson kept as a
    sensitivity check.
    """
    channels = analysis.channels
    human = analysis.human
    n_human_subjects = human.n_subjects

    metrics = HumanToHumanMetrics(
        pairwise_joint_rmse=_empty_per_channel(channels.joint_names, "all_joints"),
        pairwise_joint_corr=_empty_per_channel(channels.joint_names, "all_joints"),
        pairwise_muscle_corr=_empty_per_channel(channels.muscle_names, "all_muscles"),
        rsa_matrix=np.full((n_human_subjects, n_human_subjects), np.nan),
        rsa_matrix_pearson=np.full((n_human_subjects, n_human_subjects), np.nan),
    )
    analysis.human_metrics = metrics

    for subj_idx_1 in range(n_human_subjects):
        human_joint_gaits_1 = human.joint_data[subj_idx_1]
        human_muscle_gaits_1 = human.muscle_data[subj_idx_1]
        rsm_1 = human.muscle_rsms[subj_idx_1]
        metrics.rsa_matrix[subj_idx_1, subj_idx_1] = 1.0  # A subject against itself
        metrics.rsa_matrix_pearson[subj_idx_1, subj_idx_1] = 1.0

        for subj_idx_2 in range(subj_idx_1 + 1, n_human_subjects):
            human_joint_gaits_2 = human.joint_data[subj_idx_2]
            human_muscle_gaits_2 = human.muscle_data[subj_idx_2]
            rsm_2 = human.muscle_rsms[subj_idx_2]

            for joint_idx, joint_name in enumerate(channels.joint_names):
                metrics.pairwise_joint_rmse[joint_name].append(
                    paired_rmse(human_joint_gaits_1[joint_idx], human_joint_gaits_2[joint_idx])
                )
                metrics.pairwise_joint_corr[joint_name].append(
                    paired_pearson(human_joint_gaits_1[joint_idx], human_joint_gaits_2[joint_idx])
                )

            for muscle_idx, muscle_name in enumerate(channels.muscle_names):
                metrics.pairwise_muscle_corr[muscle_name].append(
                    paired_pearson(human_muscle_gaits_1[muscle_idx], human_muscle_gaits_2[muscle_idx])
                )

            rsa_spearman, rsa_pearson = rsa(rsm_1, rsm_2)
            metrics.rsa_matrix[subj_idx_2, subj_idx_1] = rsa_spearman
            metrics.rsa_matrix_pearson[subj_idx_2, subj_idx_1] = rsa_pearson

    n_pairs = len(metrics.pairwise_joint_rmse[channels.joint_names[0]])
    metrics.pairwise_joint_rmse["all_joints"] = [
        finite_mean([metrics.pairwise_joint_rmse[joint][pair] for joint in channels.joint_names])
        for pair in range(n_pairs)
    ]
    metrics.pairwise_joint_corr["all_joints"] = [
        finite_mean([metrics.pairwise_joint_corr[joint][pair] for joint in channels.joint_names])
        for pair in range(n_pairs)
    ]
    metrics.pairwise_muscle_corr["all_muscles"] = [
        finite_mean([metrics.pairwise_muscle_corr[muscle][pair] for muscle in channels.muscle_names])
        for pair in range(n_pairs)
    ]

    metrics.avg_pairwise_joint_rmse = {
        joint: finite_mean(values) for joint, values in metrics.pairwise_joint_rmse.items()
    }
    metrics.avg_pairwise_joint_corr = {
        joint: finite_mean(values) for joint, values in metrics.pairwise_joint_corr.items()
    }
    metrics.avg_pairwise_muscle_corr = {
        muscle: finite_mean(values) for muscle, values in metrics.pairwise_muscle_corr.items()
    }
    metrics.avg_pairwise_muscle_rsa = finite_mean(metrics.rsa_matrix[np.tril_indices_from(metrics.rsa_matrix, k=-1)])
    metrics.avg_pairwise_muscle_rsa_pearson = finite_mean(
        metrics.rsa_matrix_pearson[np.tril_indices_from(metrics.rsa_matrix_pearson, k=-1)]
    )
