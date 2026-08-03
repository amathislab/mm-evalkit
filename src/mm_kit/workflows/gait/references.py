"""Human reliability, policy variability, and sensitivity references."""

from __future__ import annotations

from mm_kit.stats.rsa import rsa

from .human_data import human_step_profile
from .metrics import _empty_per_channel, compute_rsm
from .numerics import finite_mean, paired_pearson, paired_rmse
from .state import GaitAnalysis, NoiseCeiling, SubjectReference


def compute_human_reference_benchmarks(analysis: GaitAnalysis) -> None:
    """Compute the human reference benchmarks for the current gait position.

    Pairwise human-to-human similarity (already in compute_human_to_human_metrics)
    measures *between-subject* variability, which is not the right upper bound for a
    subject-matched model. Two subject-matched references are added here:

    * **Within-subject reliability** -- for each subject, trial 5 (the tested trial,
      same step as the current gait) vs the mean of that subject's trials 1-4. This is
      the appropriate upper benchmark: how well a subject reproduces their own gait.
    * **RSA noise ceiling (leave-one-subject-out)** -- for each subject, their muscle
      RSM vs the mean RSM of all *other* subjects. The ceiling on how well any single
      representational geometry can match the group.

    Both are computed over the subjects/step of the current gait position. Requires the
    per-step trial-5 and trials-1-4 human data (Gait120); left as None otherwise.
    """
    analysis.references.within_subject = None
    analysis.references.rsa_noise_ceiling = None
    human = analysis.human
    if not analysis.subject_based or not human.earlier_joint_by_step:
        return

    channels = analysis.channels
    n_subjects = human.n_subjects
    n_joints = len(channels.joint_names)
    n_muscles = len(channels.muscle_names)

    # --- Within-subject reliability: trial 5 vs mean(trials 1-4), same step ---
    # Kept per joint / per muscle (plus "all_*" aggregates), mirroring the pairwise
    # human-human structure, so the subject-based bar plots can use it per channel.
    wj_rmse = _empty_per_channel(channels.joint_names, "all_joints")
    wj_corr = _empty_per_channel(channels.joint_names, "all_joints")
    wm_corr = _empty_per_channel(channels.muscle_names, "all_muscles")
    w_rsa_sp, w_rsa_pe = [], []
    # Subject id behind each entry of the lists above, in the same order: the paired
    # statistics match this reference to a policy subject by subject.
    w_subjects = []
    for subj in human.subjects:
        train_joint = human_step_profile(human.earlier_joint_by_step, subj, analysis.gait_position)
        train_muscle = human_step_profile(human.earlier_muscle_by_step, subj, analysis.gait_position)
        if train_joint is None or train_muscle is None:
            continue  # this subject has no earlier-trial data at this step
        # Always use the raw *human* trial-5 joints/muscles here (not human.joint_data,
        # which is the MFB-retargeted target in 'mfb-retargeted' mode): within-subject
        # reliability is a property of the human measurement, independent of the model's
        # kinematic ground truth.
        test_joint = human_step_profile(human.joint_by_step, subj, analysis.gait_position)
        test_muscle = human_step_profile(human.muscle_by_step, subj, analysis.gait_position)

        per_joint_rmse = [paired_rmse(test_joint[k], train_joint[k]) for k in range(n_joints)]
        per_joint_corr = [paired_pearson(test_joint[k], train_joint[k]) for k in range(n_joints)]
        for k, jname in enumerate(channels.joint_names):
            wj_rmse[jname].append(per_joint_rmse[k])
            wj_corr[jname].append(per_joint_corr[k])
        wj_rmse["all_joints"].append(finite_mean(per_joint_rmse))
        wj_corr["all_joints"].append(finite_mean(per_joint_corr))

        per_muscle_corr = [paired_pearson(test_muscle[k], train_muscle[k]) for k in range(n_muscles)]
        for k, mname in enumerate(channels.muscle_names):
            wm_corr[mname].append(per_muscle_corr[k])
        wm_corr["all_muscles"].append(finite_mean(per_muscle_corr))

        sp, pe = rsa(compute_rsm(test_muscle), compute_rsm(train_muscle))
        w_rsa_sp.append(sp)
        w_rsa_pe.append(pe)
        w_subjects.append(int(subj))

    analysis.references.within_subject = SubjectReference(
        n=len(wj_rmse["all_joints"]),
        subjects=w_subjects,
        joint_rmse=wj_rmse,
        joint_corr=wj_corr,
        muscle_corr=wm_corr,
        rsa_spearman=w_rsa_sp,
        rsa_pearson=w_rsa_pe,
    )

    # --- RSA noise ceiling (leave-one-subject-out) ---
    nc_sp, nc_pe = [], []
    rsms = [human.muscle_rsms[i] for i in range(n_subjects)]
    for i in range(n_subjects):
        others = [rsms[j] for j in range(n_subjects) if j != i]
        if not others:
            continue
        loso_mean = finite_mean(others, axis=0)
        sp, pe = rsa(rsms[i], loso_mean)
        nc_sp.append(sp)
        nc_pe.append(pe)
    analysis.references.rsa_noise_ceiling = NoiseCeiling(n=len(nc_sp), rsa_spearman=nc_sp, rsa_pearson=nc_pe)


def compute_policy_within_subject_variability(analysis: GaitAnalysis) -> None:
    """Compute the *per-policy* within-subject variability reference for this gait.

    The model-side analogue of within-subject reliability, computed separately for each
    policy so it can be shown as that policy's own reproducibility bound. Per policy and
    per subject the recipe is: average each trial's rollouts into one profile, compare
    the tested trial (5) against each earlier trial (1-4) separately, then average those
    (up to four) per-trial numbers into one per-subject value. A policy's row then
    summarises those per-subject values as mean +- std over subjects -- a bound on how
    reproducible the policy's own output is across a subject's trials, independent of the
    human data.

    The result is stored on ``run.within_subject_variability`` (None when that policy has
    no earlier-trial rollouts at this gait); the same records are collected on
    ``analysis.references.policy_within_subject`` for reporting. Requires the optional
    trials-1-4 rollouts (``trajectory_data_path_trials_1_4`` in the policy config),
    per-trial gait-segmented by ``select_gait_subset``; None otherwise.
    """
    analysis.references.policy_within_subject = None
    if not analysis.subject_based:
        return

    channels = analysis.channels
    n_joints = len(channels.joint_names)
    n_muscles = len(channels.muscle_names)
    summaries = []

    for run in analysis.policies:
        run.within_subject_variability = None
        per_trial = run.earlier_trial_profiles
        if not per_trial:
            continue  # this policy has no earlier-trial rollouts at this gait position
        t5_joint = run.profiles.per_subject_joint or {}
        t5_muscle = run.profiles.per_subject_muscle or {}
        t5_rsm = run.profiles.per_subject_muscle_rsm or {}

        # Per-subject values: average across the earlier trials the policy has for that
        # subject at this gait (trial 5 vs trial 1, vs trial 2, ...), one number each.
        # Joint RMSE and muscle correlation are kept as per-channel vectors (averaging
        # over the channels afterwards gives exactly the same "all_*" aggregate) so this
        # reference can also take part in the across-joints / across-muscles statistics.
        subj_joint_rmse, subj_muscle_corr, subj_rsa_sp, subj_rsa_pe = [], [], [], []
        subjects = []
        for subj in t5_joint:
            if subj not in t5_muscle:
                continue
            j5, m5, rsm5 = t5_joint[subj], t5_muscle[subj], t5_rsm[subj]
            trial_joint_rmse, trial_muscle_corr, trial_rsa_sp, trial_rsa_pe = [], [], [], []
            for profiles in per_trial.values():
                if subj not in profiles.per_subject_joint:
                    continue
                jt = profiles.per_subject_joint[subj]
                mt = profiles.per_subject_muscle[subj]
                rsmt = profiles.per_subject_muscle_rsm[subj]
                trial_joint_rmse.append([paired_rmse(j5[k], jt[k]) for k in range(n_joints)])
                trial_muscle_corr.append([paired_pearson(m5[k], mt[k]) for k in range(n_muscles)])
                sp, pe = rsa(rsm5, rsmt)
                trial_rsa_sp.append(sp)
                trial_rsa_pe.append(pe)
            if not trial_joint_rmse:
                continue  # no earlier trial for this subject at this gait
            subj_joint_rmse.append(finite_mean(trial_joint_rmse, axis=0))
            subj_muscle_corr.append(finite_mean(trial_muscle_corr, axis=0))
            subj_rsa_sp.append(finite_mean(trial_rsa_sp))
            subj_rsa_pe.append(finite_mean(trial_rsa_pe))
            subjects.append(int(subj))

        if not subj_joint_rmse:
            continue
        result = SubjectReference(
            label=run.label,
            n=len(subj_joint_rmse),
            subjects=subjects,
            joint_rmse={
                **{name: [v[k] for v in subj_joint_rmse] for k, name in enumerate(channels.joint_names)},
                "all_joints": [float(finite_mean(v)) for v in subj_joint_rmse],
            },
            muscle_corr={
                **{name: [v[k] for v in subj_muscle_corr] for k, name in enumerate(channels.muscle_names)},
                "all_muscles": [float(finite_mean(v)) for v in subj_muscle_corr],
            },
            rsa_spearman=subj_rsa_sp,
            rsa_pearson=subj_rsa_pe,
        )
        run.within_subject_variability = result
        summaries.append(result)

    analysis.references.policy_within_subject = summaries or None


def compute_subject_profile_metrics(model_joint, model_muscle, human_joint, human_muscle):
    """Compute kinematic, EMG, and RSA metrics for one matched subject."""
    joint_rmse = [paired_rmse(model_joint[k], human_joint[k]) for k in range(model_joint.shape[0])]
    joint_corr = [paired_pearson(model_joint[k], human_joint[k]) for k in range(model_joint.shape[0])]
    muscle_corr = [paired_pearson(model_muscle[k], human_muscle[k]) for k in range(model_muscle.shape[0])]
    rsa_spearman, rsa_pearson = rsa(
        compute_rsm(model_muscle),
        compute_rsm(human_muscle),
    )
    return joint_rmse, joint_corr, muscle_corr, rsa_spearman, rsa_pearson


def compute_policy_to_earlier_human_sensitivity(analysis: GaitAnalysis) -> None:
    """Compare each policy's Trial-5 output with mean human Trials 1-4.

    This is the trial-reference sensitivity analysis requested alongside the primary
    subject-matched comparison (Policy Trial 5 vs Human Trial 5). For every policy and
    subject with both a retained policy gait and an earlier-human profile at the current
    gait position, it computes the same per-subject kinematic, EMG, and RSA metrics used
    by the primary analysis. The Gait120 loader has already averaged available
    Trials 1-4 into subject/step profiles; subjects with unavailable data on
    either side remain missing and are reported through ``n`` and ``subjects``.
    """
    for run in analysis.policies:
        run.earlier_human_sensitivity = None

    if not analysis.subject_based:
        return

    channels = analysis.channels
    human = analysis.human
    for run in analysis.policies:
        policy_joint = run.profiles.per_subject_joint
        if not policy_joint:
            continue
        policy_muscle = run.profiles.per_subject_muscle

        joint_rmse = _empty_per_channel(channels.joint_names, "all_joints")
        joint_corr = _empty_per_channel(channels.joint_names, "all_joints")
        muscle_corr = _empty_per_channel(channels.muscle_names, "all_muscles")
        rsa_spearman, rsa_pearson, subjects = [], [], []

        # Preserve the human cohort order in the result and skip, rather than drop from
        # the cohort globally, subjects missing either the policy or earlier-human gait.
        for subj in human.subjects:
            if subj not in policy_joint:
                continue
            human_joint = human_step_profile(human.earlier_joint_by_step, subj, analysis.gait_position)
            human_muscle = human_step_profile(human.earlier_muscle_by_step, subj, analysis.gait_position)
            if human_joint is None or human_muscle is None:
                continue

            per_joint_rmse, per_joint_corr, per_muscle_corr, sp, pe = compute_subject_profile_metrics(
                policy_joint[subj],
                policy_muscle[subj],
                human_joint,
                human_muscle,
            )
            for k, joint_name in enumerate(channels.joint_names):
                joint_rmse[joint_name].append(per_joint_rmse[k])
                joint_corr[joint_name].append(per_joint_corr[k])
            joint_rmse["all_joints"].append(finite_mean(per_joint_rmse))
            joint_corr["all_joints"].append(finite_mean(per_joint_corr))

            for k, muscle_name in enumerate(channels.muscle_names):
                muscle_corr[muscle_name].append(per_muscle_corr[k])
            muscle_corr["all_muscles"].append(finite_mean(per_muscle_corr))

            rsa_spearman.append(sp)
            rsa_pearson.append(pe)
            subjects.append(int(subj))

        if not subjects:
            continue
        run.earlier_human_sensitivity = SubjectReference(
            n=len(subjects),
            subjects=subjects,
            joint_rmse=joint_rmse,
            joint_corr=joint_corr,
            muscle_corr=muscle_corr,
            rsa_spearman=rsa_spearman,
            rsa_pearson=rsa_pearson,
        )
