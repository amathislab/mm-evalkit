"""Human dataset loading and gait-position selection."""

from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path

import numpy as np

from mm_evalkit.datasets.gait120 import load_gait120
from mm_evalkit.datasets.wang import load_wang

from . import kinetics_data
from .cohorts import shared_policy_subjects
from .metrics import compute_rsm
from .models import NoSubjectsForGaitError
from .state import GaitAnalysis, GaitChannels, HumanCohort, PolicyCycles, PolicyProfiles


def save_gait120_omission_audit(plots_dir, omissions):
    """Write one stable CSV row for every Gait120 input excluded from an average."""
    path = Path(plots_dir) / "gait120_input_omissions.csv"
    fieldnames = ("subject", "trial", "step", "trial_group", "reason", "source")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(omission) for omission in omissions)
    return path


def load_human_cohort(config, channels: GaitChannels, muscle_map, subjects, plots_dir) -> HumanCohort:
    """Load the human reference selected by the validated analysis config."""
    print(f" ### --------------------Loading Human Data: {config.dataset_name}-------------------- ### ")
    print(f"  Subjects: {subjects}")

    empty: dict[int, dict[int, np.ndarray]] = {}
    if config.dataset_name == "Gait120":
        data = load_gait120(
            str(config.dataset_dir),
            requested_subjects=subjects,
            selected_joint_names=list(channels.joint_names),
            selected_muscle_names=list(channels.muscle_names),
            muscle_map=muscle_map,
            missing_subject_policy=config.human_subject_policy,
        )
        cohort = HumanCohort(
            all_subjects=list(data.subjects),
            joint_names=list(data.joint_names),
            muscle_names=list(data.muscle_names),
            joint_by_step=data.joint_by_step,
            muscle_by_step=data.muscle_by_step,
            earlier_joint_by_step=data.earlier_joint_by_step,
            earlier_muscle_by_step=data.earlier_muscle_by_step,
            omissions=data.omissions,
            subjects=list(data.subjects),
            joint_data=data.joint_data,
            muscle_data=data.muscle_data,
        )
        audit_path = save_gait120_omission_audit(plots_dir, data.omissions)
        print(f"  Gait120 omission audit: {audit_path}")
    else:
        data = load_wang(
            str(config.dataset_dir),
            selected_joint_names=list(channels.joint_names),
            selected_muscle_names=list(channels.muscle_names),
            muscle_map=muscle_map,
        )
        cohort = HumanCohort(
            all_subjects=[],
            joint_names=list(data.joint_names),
            muscle_names=list(data.muscle_names),
            joint_by_step=empty,
            muscle_by_step=empty,
            earlier_joint_by_step=empty,
            earlier_muscle_by_step=empty,
            omissions=(),
            subjects=[],
            joint_data=data.joint_data,
            muscle_data=data.muscle_data,
        )

    refresh_human_rsms(cohort)
    print(f"  Loaded joint data shape: {cohort.joint_data.shape} (subjects, joints, time_steps)")
    print(f"  Loaded muscle data shape: {cohort.muscle_data.shape} (subjects, muscles, time_steps)")
    return cohort


def refresh_human_rsms(cohort: HumanCohort) -> None:
    """Recompute the per-subject and group muscle RSMs from the current cohort."""
    cohort.muscle_rsms = {index: compute_rsm(cohort.muscle_data[index]) for index in range(cohort.n_subjects)}
    cohort.muscle_rsms_avg = compute_rsm(np.mean(cohort.muscle_data, axis=0))


def choose_plot_subjects(cohort: HumanCohort, max_subjects: int) -> None:
    """Pick the subset of subjects the per-subject figures show.

    Figures stay legible with a hundred subjects by drawing a fixed random
    subset; the seed is constant so every figure of a run shows the same people.
    These are positions into ``subjects``, not subject ids.
    """
    n_subjects = cohort.n_subjects
    rng = np.random.default_rng(0)
    if n_subjects > max_subjects:
        cohort.plot_subject_positions = sorted(rng.choice(n_subjects, size=max_subjects, replace=False).tolist())
    else:
        cohort.plot_subject_positions = list(range(n_subjects))
    print(
        f"  Per-subject plots will show {len(cohort.plot_subject_positions)}/{n_subjects} subjects "
        f"(positions: {cohort.plot_subject_positions})"
    )


def build_policy_profiles(joint, joint_ref, muscle, subj_ids) -> PolicyProfiles:
    """Build the per-subject and equal-subject-weighted population profiles.

    Given the per-cycle joint / reference-joint / muscle arrays (each
    ``(n_cycles, n_channels, 100)``) and the ``subj_ids`` of each cycle, returns the
    profiles every metric and plot consumes: per-subject means (one profile per
    subject, keyed by subject id), per-subject muscle RSMs, and the population
    profiles (each subject weighted equally, then averaged across subjects).
    Used to (re)build a policy for a single gait position.
    """
    per_muscle, per_joint, per_joint_ref, per_rsm = {}, {}, {}, {}
    for s in np.unique(subj_ids):
        m = subj_ids == s
        s = int(s)
        per_muscle[s] = np.mean(muscle[m], axis=0)
        per_joint[s] = np.mean(joint[m], axis=0)
        per_joint_ref[s] = np.mean(joint_ref[m], axis=0)
        per_rsm[s] = compute_rsm(per_muscle[s])

    avg_muscle = np.mean(list(per_muscle.values()), axis=0)
    return PolicyProfiles(
        avg_joint_gaits=np.mean(list(per_joint.values()), axis=0),
        std_joint_gaits=np.std(joint, axis=0),
        avg_joint_gaits_ref=np.mean(list(per_joint_ref.values()), axis=0),
        std_joint_gaits_ref=np.std(joint_ref, axis=0),
        avg_muscle_gaits=avg_muscle,
        std_muscle_gaits=np.std(muscle, axis=0),
        population_muscle_rsm=compute_rsm(avg_muscle),
        subject_ids=np.asarray([int(s) for s in subj_ids], dtype=int),
        per_subject_muscle=per_muscle,
        per_subject_joint=per_joint,
        per_subject_joint_ref=per_joint_ref,
        per_subject_muscle_rsm=per_rsm,
    )


def gait_selection_name(gait_position):
    """Output-folder name for a gait selection (``None`` = every gait pooled)."""
    return "all_gaits" if gait_position is None else f"gait{gait_position + 1}"


def gait_selection_label(gait_position):
    """Human-readable name of the current gait selection, for titles and captions."""
    if gait_position is None:
        return "all gaits pooled"
    return f"gait {gait_position + 1}"


def human_step_profile(by_step, subj, gait_position):
    """The human profile of ``subj`` for one gait selection, or ``None``.

    For a single gait position that is the matching step; for the pooled analysis
    (``gait_position is None``) it is the mean over every step the subject
    has, matching how the policy side pools its cycles across gait positions.
    """
    steps = by_step.get(subj, {})
    if not steps:
        return None
    if gait_position is None:
        return np.mean(list(steps.values()), axis=0)
    return steps.get(gait_position)


def _cycle_mask(cycles: PolicyCycles, active_subjects, gait_position):
    """Cycles belonging to an active subject at the selected gait position."""
    if cycles.subject_ids is None or len(cycles.subject_ids) == 0:
        return np.zeros(0, dtype=bool)
    mask = np.isin(cycles.subject_ids, active_subjects)
    if gait_position is not None:
        mask &= cycles.gait_positions == gait_position
    return mask


def select_gait_subset(analysis: GaitAnalysis, gait_position) -> None:
    """Restrict every policy and the human ground truth to one gait ordinal, or pool them.

    ``gait_position`` 0 selects the first gait cycle / human step (step01), 1 the
    second (step02), and ``None`` pools every gait cycle of every episode against the
    mean of the subject's steps -- the "all gaits together" analysis. The
    selected cohort contains only subjects represented in the human data and
    every compared policy, so all policy summaries use the same subjects.

    For the chosen selection this rebuilds, per policy, the per-subject and population
    profiles from the per-cycle arrays, and swaps in the human joint/muscle ground
    truth for the matching step (step01 -> gait1, step02 -> gait2, mean of both ->
    pooled), together with the per-subject RSMs and plotting subset. In
    'mfb-retargeted' mode the kinematic ground truth is then rebuilt from the
    correspondingly filtered reference trajectories.
    """
    # Subjects whose human data has this step (joint & EMG are loaded together, so a
    # step present for muscle is present for joints too). This is the analysis set.
    human = analysis.human
    analysis.gait_position = gait_position
    active_subjects = [
        s for s in human.all_subjects if human_step_profile(human.muscle_by_step, s, gait_position) is not None
    ]
    if analysis.kinetics is not None:
        kinetic_subjects = kinetics_data.available_subjects(analysis.kinetics.cycles, active_subjects, gait_position)
        active_subjects = [subject for subject in active_subjects if subject in kinetic_subjects]
    if not active_subjects:
        raise NoSubjectsForGaitError(f"no subject has human data for {gait_selection_label(gait_position)}")

    masks = [_cycle_mask(run.cycles, active_subjects, gait_position) for run in analysis.policies]
    available_by_policy = {}
    for run, mask in zip(analysis.policies, masks, strict=True):
        available = {int(subject) for subject in np.unique(run.cycles.subject_ids[mask])} if mask.size else set()
        if analysis.kinetics is not None:
            if run.kinetics_cycles is None:
                raise RuntimeError(f"Policy {run.policy.id!r} has no loaded kinetics cycles.")
            available &= kinetics_data.available_subjects(run.kinetics_cycles, active_subjects, gait_position)
        available_by_policy[run.policy.id] = available
    active_subjects = list(
        shared_policy_subjects(
            active_subjects,
            available_by_policy,
            selection=gait_selection_label(gait_position),
        )
    )
    analysis.cohort_subjects = tuple(active_subjects)

    # --- Human ground truth = the step(s) matching this gait selection ---
    human.subjects = active_subjects
    human.joint_data = np.array([human_step_profile(human.joint_by_step, s, gait_position) for s in active_subjects])
    human.muscle_data = np.array([human_step_profile(human.muscle_by_step, s, gait_position) for s in active_subjects])
    refresh_human_rsms(human)
    choose_plot_subjects(human, analysis.style.max_plot_subjects)

    # --- Rebuild each policy from its cycles at this gait position ---
    for run in analysis.policies:
        mask = _cycle_mask(run.cycles, active_subjects, gait_position) & np.isin(
            run.cycles.subject_ids, active_subjects
        )
        run.profiles = build_policy_profiles(
            run.cycles.joint[mask],
            run.cycles.joint_ref[mask],
            run.cycles.muscle[mask],
            run.cycles.subject_ids[mask],
        )
        run.earlier_trial_profiles = _earlier_trial_profiles(run.earlier_trials, active_subjects, gait_position)

    if analysis.kinetics is not None:
        kinetics_data.select_kinetics_subset(analysis.policies, analysis.kinetics, active_subjects, gait_position)

    # --- Gait-specific MFB-retargeted kinematic ground truth ---
    if analysis.kinematics_source == "mfb-retargeted":
        use_reference_as_joint_ground_truth(analysis)


def _earlier_trial_profiles(cycles, active_subjects, gait_position):
    """Per-earlier-trial profiles for one gait position, or ``None``.

    Independent of the trial-5 rebuild: a policy may have cycles at this gait for
    one and not the other. Kept split *per earlier trial* rather than pooled, so
    ``compute_policy_within_subject_variability`` can compare trial 5 against each
    earlier trial separately and average those numbers. Trial 5 is excluded here
    because it is the tested trial and comes from the main rollouts.
    """
    if cycles is None or cycles.subject_ids is None:
        return None
    gait_mask = _cycle_mask(cycles, active_subjects, gait_position)
    if not np.any(gait_mask):
        return None

    per_trial_profiles = {}
    for trial in np.unique(cycles.trial_ids[gait_mask]):
        if int(trial) == 5:
            continue
        mask = gait_mask & (cycles.trial_ids == trial)
        if not np.any(mask):
            continue
        per_trial_profiles[int(trial)] = build_policy_profiles(
            cycles.joint[mask],
            cycles.joint_ref[mask],
            cycles.muscle[mask],
            cycles.subject_ids[mask],
        )
    return per_trial_profiles or None


def use_reference_as_joint_ground_truth(analysis: GaitAnalysis) -> None:
    """Replace the OpenSim-IK joint angles with the MFB-retargeted target.

    Every policy must provide the same retained reference profile for each
    selected subject. Averaging policy-specific segmentations would make the
    ground truth depend on which policies happened to be compared.
    """
    print(" ### --------------------Using MFB-retargeted target as kinematic ground truth-------------------- ### ")

    human = analysis.human
    references = np.array(human.joint_data, copy=True)
    expected_shape = human.joint_data.shape[1:]
    for index, subj in enumerate(human.subjects):
        profiles = []
        missing = []
        for run in analysis.policies:
            by_subject = run.profiles.per_subject_joint_ref
            if by_subject is None or subj not in by_subject:
                missing.append(run.policy.id)
            else:
                profiles.append((run.policy.id, np.asarray(by_subject[subj], dtype=float)))
        if missing:
            raise ValueError(f"MFB-retargeted reference for subject {subj} is missing from policy/policies {missing}.")
        if not profiles:
            raise ValueError("MFB-retargeted kinematics require at least one policy reference.")
        incompatible = [
            (policy_id, profile.shape) for policy_id, profile in profiles if profile.shape != expected_shape
        ]
        if incompatible:
            raise ValueError(
                f"MFB-retargeted reference for subject {subj} has incompatible "
                f"profile shape(s) {incompatible}; expected {expected_shape}."
            )
        if not all(np.isfinite(profile).all() for _, profile in profiles):
            raise ValueError(f"MFB-retargeted reference for subject {subj} contains non-finite values.")
        baseline_id, baseline = profiles[0]
        disagreements = [
            policy_id for policy_id, profile in profiles[1:] if not np.allclose(profile, baseline, rtol=0.0, atol=1e-10)
        ]
        if disagreements:
            raise ValueError(
                f"MFB-retargeted reference for subject {subj} differs between "
                f"policy {baseline_id!r} and policy/policies {disagreements}."
            )
        references[index] = baseline

    human.joint_data = references
    print(f"  Reference joint ground-truth shape: {human.joint_data.shape} (subjects, joints, time_steps)")
