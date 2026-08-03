"""Thin orchestration for the validated ULTra-MoCap analysis stages."""

from __future__ import annotations

from collections.abc import Sequence

from mm_evalkit.policies import PolicySpec

from .aggregate_waveforms import plot_aggregated_waveforms
from .constants import PRIMARY_JOINTS
from .metrics import trial_metrics
from .models import (
    AnalysisConfig,
    EpisodeExclusion,
    MuscleGroup,
    PolicyAnalysis,
)
from .muscles import build_muscle_groups, load_muscle_names
from .references import merge_real_records
from .reporting import save_run_manifest, save_trial_exclusions
from .rollout import open_rollout, resolve_trial_stems, validate_policy_rollouts
from .rsm_plots import plot_rsm_matrices
from .statistics import run_wilcoxon_tests
from .summary_plots import (
    plot_aggregated_bars,
    plot_aggregated_by_channel,
    plot_summary_bars,
    plot_summary_by_channel,
)
from .tables import plot_metrics_table
from .trial_plots import plot_joint_trial, plot_muscle_trial, plot_trial_waveforms
from .trials import load_trial


def validate_comparable_trials(results, policies) -> None:
    """Require identical segmented trials and repetition boundaries.

    Descriptive and inferential policy comparisons must use the same real
    repetitions. A policy-specific exclusion or boundary set is reported as an
    input incompatibility instead of silently changing the comparison sample.
    """
    first_policy = policies[0]
    baseline = results[first_policy.id].trials
    baseline_keys = set(baseline)
    for policy in policies[1:]:
        current = results[policy.id].trials
        current_keys = set(current)
        if current_keys != baseline_keys:
            missing = sorted(baseline_keys - current_keys)
            extra = sorted(current_keys - baseline_keys)
            raise ValueError(
                f"Policy {policy.id!r} has a different segmented trial set "
                f"from policy {first_policy.id!r}; missing={missing}, extra={extra}."
            )
        for key in sorted(baseline_keys):
            expected = baseline[key]
            observed = current[key]
            if observed.cycle_boundaries != expected.cycle_boundaries:
                raise ValueError(f"Policy {policy.id!r} uses different joint repetition boundaries for trial {key!r}.")
            if observed.muscle_cycle_boundaries != expected.muscle_cycle_boundaries:
                raise ValueError(f"Policy {policy.id!r} uses different paired muscle boundaries for trial {key!r}.")


def analyse_policy(
    policy: PolicySpec,
    options: AnalysisConfig,
    joint_names: tuple[str, ...],
    activation_count: int,
    muscle_columns: Sequence[int],
    groups: Sequence[MuscleGroup],
    joint_selection: Sequence[int],
) -> PolicyAnalysis:
    """Load, identify, segment, and score every requested rollout episode."""
    print(f"\n=== {policy.label.replace(chr(10), ' ')} ({policy.id}) ===")
    with open_rollout(
        policy.trajectory_path,
        expected_activation_count=activation_count,
        expected_motion_group=options.motion_group,
    ) as (trajectory, metadata):
        requested = tuple(range(metadata.n_episodes)) if options.episodes is None else options.episodes
        motions = resolve_trial_stems(
            trajectory,
            metadata,
            options.motion_paths,
            episodes=requested,
        )
        print(
            f"Loaded {policy.trajectory_path}: {metadata.n_episodes} episodes, "
            f"dt={metadata.dt:g}; processing {len(requested)}."
        )

        trials = {}
        records = {}
        exclusions: list[EpisodeExclusion] = []
        for episode, trial_stem in motions.items():
            trial = load_trial(
                trajectory,
                episode,
                trial_stem,
                joint_names,
                muscle_columns,
                groups,
                options.arm,
                options.emg_roots,
            )
            if not trial.is_segmented:
                exclusions.append(
                    EpisodeExclusion(
                        episode=episode,
                        trial_stem=trial_stem,
                        stage="segmentation",
                        reason=(trial.exclusion_reason or "no sufficiently periodic reference joint"),
                        detail=(f"rollout_frames={trial.rollout_frames}; reference_frames={trial.reference_frames}"),
                    )
                )
                continue

            key = trial.key
            if key in trials:
                raise ValueError(f"Policy {policy.id!r} maps multiple episodes to trial {key!r}.")
            trials[key] = trial
            records[key] = trial_metrics(trial, groups, joint_selection)

    return PolicyAnalysis(
        trials=trials,
        records=records,
        exclusions=tuple(exclusions),
        requested_episodes=len(requested),
    )


def _select_joints(
    joint_names: tuple[str, ...],
    arm: str,
    all_joints: bool,
) -> tuple[list[int], list[str]]:
    if all_joints:
        selection = list(range(len(joint_names)))
    else:
        suffix = "_l" if arm == "left" else "_r"
        required = [f"{joint}{suffix}" for joint in PRIMARY_JOINTS]
        missing = [joint for joint in required if joint not in joint_names]
        if missing:
            raise ValueError(
                f"Rollout is missing primary joint(s) required by the bimanual analysis: {', '.join(missing)}."
            )
        selection = [joint_names.index(joint) for joint in required]
    return selection, [joint_names[index] for index in selection]


def _plot_trial_comparisons(
    options: AnalysisConfig,
    policies,
    policy_results,
    real_records,
    groups,
    muscle_entries,
    muscle_labels,
    joint_selection,
    joint_labels,
) -> None:
    for movement, condition, subject in sorted(real_records):
        key = (movement, condition, subject)
        entries = [
            (policy, policy_results[policy.id].records[key])
            for policy in policies
            if key in policy_results[policy.id].records
        ]
        out_dir = options.plots_dir / f"{movement}_{condition}"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{subject} {movement} {condition}:")
        plot_trial_waveforms(
            entries,
            groups,
            joint_labels,
            muscle_labels,
            title=f"{subject} {movement} {condition} ({options.arm} arm) -",
            joints_path=str(out_dir / f"{subject}_joints.png"),
            muscles_path=str(out_dir / f"{subject}_muscles.png"),
        )

        if not options.raw_trials:
            continue
        for policy in policies:
            trial = policy_results[policy.id].trials.get(key)
            if trial is None:
                continue
            raw_dir = out_dir / "raw"
            if len(policies) > 1:
                raw_dir /= policy.slug
            raw_dir.mkdir(parents=True, exist_ok=True)
            plot_joint_trial(
                trial.policy_qpos[:, joint_selection],
                trial.reference_qpos[:, joint_selection],
                joint_labels,
                title=(f"{trial.stem} - joint angles ({options.arm} arm){trial.early_stop_label}"),
                out_path=str(raw_dir / f"{subject}_joints.png"),
            )
            plot_muscle_trial(
                trial.policy_activation,
                trial.emg,
                muscle_entries,
                title=(f"{trial.stem} - muscles ({options.arm} arm){trial.early_stop_label}"),
                out_path=str(raw_dir / f"{subject}_muscles.png"),
            )


def _plot_policy_summaries(
    options: AnalysisConfig,
    policies,
    policy_records,
    groups,
    joint_labels,
) -> None:
    for policy in policies:
        records = policy_records[policy.id]
        if not records:
            continue
        out_dir = options.plots_dir if len(policies) == 1 else options.plots_dir / policy.slug
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = "" if len(policies) == 1 else f"\n{policy.label}"
        plot_summary_bars(
            records,
            str(out_dir / "summary_per_condition.png"),
            arm=options.arm,
            tag=tag,
        )
        plot_summary_by_channel(
            records,
            "joint_rmse_by",
            joint_labels,
            "RMSE (deg)",
            "Joint tracking error per joint, per movement/condition and subject "
            f"(bar = score of the repetition-averaged trial){tag}",
            str(out_dir / "summary_per_joint.png"),
        )
        plot_summary_by_channel(
            records,
            "muscle_corr_by",
            [group.title() for group in groups],
            "correlation (r)",
            f"Muscle-group activation vs real EMG per electrode, per movement/condition and subject{tag}",
            str(out_dir / "summary_per_muscle.png"),
            zero_line=True,
        )


def _plot_population_reports(
    options: AnalysisConfig,
    policies,
    policy_records,
    real_records,
    groups,
    joint_labels,
) -> None:
    root = options.plots_dir
    muscle_titles = [group.title() for group in groups]
    plot_aggregated_bars(
        policy_records,
        policies,
        real_records,
        str(root / "aggregated_per_condition.png"),
        arm=options.arm,
    )
    plot_aggregated_by_channel(
        policy_records,
        policies,
        real_records,
        "joint_rmse_by",
        joint_labels,
        "RMSE (deg)",
        "Joint tracking error per joint, aggregated over subjects "
        "(bar = mean over subjects, whisker = between-subject spread)",
        str(root / "aggregated_per_joint.png"),
    )
    plot_aggregated_by_channel(
        policy_records,
        policies,
        real_records,
        "muscle_corr_by",
        muscle_titles,
        "correlation (r)",
        "Muscle-group activation vs real EMG per electrode, aggregated over subjects "
        "(bar = mean over subjects, whisker = between-subject spread)",
        str(root / "aggregated_per_muscle.png"),
        zero_line=True,
    )
    plot_aggregated_waveforms(
        policy_records,
        policies,
        real_records,
        joint_labels,
        [f"{group.label}\n({group.channel})" for group in groups],
        str(root),
        arm=options.arm,
    )
    plot_rsm_matrices(
        policy_records,
        policies,
        real_records,
        [group.label for group in groups],
        str(root),
    )
    plot_metrics_table(
        policy_records,
        policies,
        real_records,
        str(root / "metrics_summary_table"),
        arm=options.arm,
    )
    run_wilcoxon_tests(
        policy_records,
        policies,
        real_records,
        joint_labels,
        muscle_titles,
        str(root),
    )


def run_analysis(options: AnalysisConfig) -> int:
    """Validate inputs, compute every policy, then create reports."""
    policies = options.policies
    muscle_names = load_muscle_names()
    muscle_suffix = "_left" if options.arm == "left" else ""
    groups, muscle_columns = build_muscle_groups(muscle_names, muscle_suffix)

    joint_names = validate_policy_rollouts(
        policies,
        len(muscle_names),
        options.motion_group,
        episodes=options.episodes,
    )
    joint_selection, joint_labels = _select_joints(
        joint_names,
        options.arm,
        options.all_joints,
    )
    muscle_entries = [
        (muscle, column, group.channel, group.label)
        for group in groups
        for muscle, column in zip(
            group.muscles,
            group.activation_columns,
            strict=True,
        )
    ]
    muscle_labels = [group.title(list_muscles=len(policies) == 1) for group in groups]

    print(
        f"Comparing {len(policies)} "
        f"polic{'y' if len(policies) == 1 else 'ies'}: "
        + ", ".join(policy.label.replace("\n", " ") for policy in policies)
    )
    results = {
        policy.id: analyse_policy(
            policy,
            options,
            joint_names,
            len(muscle_names),
            muscle_columns,
            groups,
            joint_selection,
        )
        for policy in policies
    }
    validate_comparable_trials(results, policies)
    policy_records = {policy.id: results[policy.id].records for policy in policies}
    real_records = merge_real_records(policy_records, policies)

    options.plots_dir.mkdir(parents=True, exist_ok=True)
    save_run_manifest(options, results)
    save_trial_exclusions(results, policies, options.plots_dir)
    if not real_records:
        print("\nNo segmentable trials in any policy; no figures were generated.")
        return 1

    _plot_trial_comparisons(
        options,
        policies,
        results,
        real_records,
        groups,
        muscle_entries,
        muscle_labels,
        joint_selection,
        joint_labels,
    )
    _plot_policy_summaries(
        options,
        policies,
        policy_records,
        groups,
        joint_labels,
    )
    _plot_population_reports(
        options,
        policies,
        policy_records,
        real_records,
        groups,
        joint_labels,
    )

    print(f"\nDone. Figures in {options.plots_dir}")
    for policy in policies:
        result = results[policy.id]
        print(
            f"  {policy.label.replace(chr(10), ' ')}: "
            f"{len(result.records)}/{result.requested_episodes} trials segmented"
        )
        for exclusion in result.exclusions:
            identity = exclusion.trial_stem or f"episode {exclusion.episode}"
            print(f"    - excluded {identity}: {exclusion.reason} ({exclusion.detail})")
    return 0
