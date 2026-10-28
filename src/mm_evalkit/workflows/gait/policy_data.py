"""Policy rollout decoding and gait-cycle extraction."""

from __future__ import annotations

import math

import numpy as np

from mm_evalkit.activation import (
    activation_sample_phase,
    align_activation_to_observation_timeline,
    complete_activation_gait_indexes,
    excitation_from_commands,
)
from mm_evalkit.policies import PolicySpec
from mm_evalkit.workflows.gait.channels import joint_names_myo, muscle_names_myo
from mm_evalkit.workflows.gait.cycles import (
    detect_gait_events,
    extract_valid_strides_flexible,
    get_feet_contacts,
    interpolate_gait_cycle,
    stride_phase_sequence,
)

from .constants import MOTION_SETS
from .metrics import compute_rsm
from .motion_identity import subject_from_trajectory, trial_from_trajectory
from .rollout_validation import validate_rollout_archive
from .state import ExtractedRollout, GaitChannels, PolicyCycles, PolicyProfiles, PolicyRun


def _episode_successful(policy_traj, episode_idx):
    """Check if an episode was successful based on termination flags."""

    eps_length = policy_traj[f"episode_{episode_idx}_joint_positions"].shape[0]
    ref_length = policy_traj[f"episode_{episode_idx}_traj_length"]
    return eps_length >= ref_length - 1


def _require_muscle_commands(policy_traj, n_episodes, source):
    """Excitation analyses need every episode's commands, shaped like activation."""
    for episode in range(n_episodes):
        key = f"episode_{episode}_muscle_commands"
        if key not in policy_traj:
            raise ValueError(f"{source} has no {key}; muscle_signal='excitation' needs exported muscle commands.")
        expected = policy_traj[f"episode_{episode}_muscle_activations"].shape
        if policy_traj[key].shape != expected:
            raise ValueError(f"{source} {key} has shape {policy_traj[key].shape}; expected {expected}.")


def _as_int(value):
    """Convert scalar numpy values loaded from npz files to Python ints."""

    return int(np.asarray(value).item())


def _format_event(event):
    """Compact string for one gait event, e.g. 'RHS@3' (right heel-strike at frame 3)."""
    side = "R" if event.side == "right" else "L"
    kind = "HS" if event.event == "heel_strike" else "TO"
    return f"{side}{kind}@{event.index}"


def episode_event_summary(policy_traj, episode):
    """Return (event_seq, phase_seq) strings for a whole episode.

    ``event_seq`` is the ordered gait events (right/left heel-strikes and toe-offs
    with their frame index); ``phase_seq`` is the run-length-compressed feet-contact
    phase sequence over the full episode. Computed from the touch sensors exactly as
    the gait extraction does, so it is available even for early-terminated episodes.
    """
    touch = policy_traj[f"episode_{episode}_touch_observations"]
    feet_contacts = get_feet_contacts(touch_sensors_data=touch, threshold=0.05)
    events = detect_gait_events(feet_contacts)
    event_seq = " ".join(_format_event(event) for event in events) if events else "(no events)"
    if len(feet_contacts) > 0:
        phase_seq = " -> ".join(stride_phase_sequence(feet_contacts, 0, len(feet_contacts) - 1))
    else:
        phase_seq = "(empty)"
    return event_seq, phase_seq


def extract_episode_gaits(
    channels: GaitChannels,
    policy_traj,
    episode,
    expected_n=None,
    *,
    activation_phase=None,
    muscle_signal="activation",
):
    """Extracting gait cycles from a single episode trajectory.

    ``expected_n`` is the number of gait cycles the reference clip should contain
    (~1 per 100 timesteps); it lets the flexible extractor keep the most canonical
    cycles when a rollout over-segments. ``muscle_signal`` picks exported
    activation or the excitation decoded from ``muscle_commands``.
    """

    touch = policy_traj[f"episode_{episode}_touch_observations"]
    joint_data_myo = policy_traj[f"episode_{episode}_joint_positions"] * 180 / math.pi
    joint_data_myo_ref = policy_traj[f"episode_{episode}_traj_qpos"] * 180 / math.pi
    if muscle_signal == "excitation":
        muscle_data_myo = excitation_from_commands(policy_traj[f"episode_{episode}_muscle_commands"])
    else:
        phase = activation_sample_phase(policy_traj) if activation_phase is None else activation_phase
        muscle_data_myo = align_activation_to_observation_timeline(
            policy_traj[f"episode_{episode}_muscle_activations"],
            phase,
        )

    feet_contacts = get_feet_contacts(touch_sensors_data=touch, threshold=0.05)

    # Flexible extraction: recovers cycles with a missing initial/final heel-strike
    # and merges spurious brief contacts that would otherwise split a cycle.
    gaits = extract_valid_strides_flexible(
        feet_contacts,
        side="right",
        expected_n=expected_n,
    )

    detected_indexes = [(int(gait.start), int(gait.end)) for gait in gaits]
    # A legacy post-step export has no activation for state zero. Do not fill
    # that sample and stretch an incomplete interval into a full gait.
    gait_indexes = complete_activation_gait_indexes(
        muscle_data_myo,
        detected_indexes,
    )

    episode_joint_gaits_myo = []
    episode_joint_gaits_myo_ref = []

    for joint_idx in channels.joint_indices_myo:
        joint_angle_cycle_f_interp, _ = interpolate_gait_cycle(joint_data_myo, joint_idx + 6, gait_indexes)
        joint_angle_cycle_f_interp_ref, _ = interpolate_gait_cycle(joint_data_myo_ref, joint_idx + 6, gait_indexes)
        episode_joint_gaits_myo.append(joint_angle_cycle_f_interp)
        episode_joint_gaits_myo_ref.append(joint_angle_cycle_f_interp_ref)

    episode_muscle_gaits_myo = []
    episode_gait_lengths = []  # Track gait lengths for this episode

    for muscle_idx in channels.muscle_indices_myo:
        # Use _with_lengths for the first muscle to get gait lengths
        muscle_control_cycle_f_interp, episode_gait_lengths = interpolate_gait_cycle(
            muscle_data_myo, muscle_idx, gait_indexes
        )
        episode_muscle_gaits_myo.append(muscle_control_cycle_f_interp)

    return (
        episode_joint_gaits_myo,
        episode_joint_gaits_myo_ref,
        episode_muscle_gaits_myo,
        episode_gait_lengths,
        gait_indexes,
    )


def extract_trajectory_data(
    channels: GaitChannels,
    npz_path,
    *,
    subject_based: bool,
    motion_list=None,
    muscle_signal="activation",
) -> ExtractedRollout:
    """Extract and process trajectory data from a given .npz file path.

    ``motion_list`` maps each episode's ``traj_id`` to its subject and trial.
    Subject-based Gait120 calls must pass it explicitly because main and
    trials-1-4 rollouts commonly use different lists.
    """
    policy_traj = np.load(npz_path, allow_pickle=True)
    n_episodes = validate_rollout_archive(
        policy_traj,
        npz_path,
        subject_based=subject_based,
        motion_count=len(motion_list) if motion_list is not None else None,
        expected_joint_names=tuple(joint_names_myo),
        expected_joint_width=len(joint_names_myo) + 6,
        expected_activation_width=len(muscle_names_myo),
    )
    activation_phase = activation_sample_phase(policy_traj)
    if muscle_signal == "excitation":
        _require_muscle_commands(policy_traj, n_episodes, npz_path)
    all_joint_gaits_myo = []
    all_joint_gaits_myo_ref = []
    all_muscle_gaits_myo = []
    all_gait_lengths = []  # Track original gait lengths before interpolation
    skipped_episodes = []
    # One record per episode: how many gait cycles were extracted for it (0 when the
    # episode terminated early or yielded no valid gait cycle). Feeds the per-subject /
    # per-policy / per-rollout gait-count table.
    episode_gait_records = []

    # Position of each extracted cycle within its episode (0 = first gait, 1 = second, ...),
    # kept in lock-step with the joint/muscle gait arrays so analyses can be split by gait ordinal.
    all_gait_positions = []
    if subject_based:
        all_subject_ids = []  # Track subject_id for each gait cycle
        all_trial_ids = []  # Track trial number for each gait cycle (to tell trials apart)

    for episode in range(n_episodes):
        traj_id = policy_traj[f"episode_{episode}_traj_id"]
        traj_id_int = _as_int(traj_id)
        # Number of timesteps in the reference trajectory (e.g. trial 5) for this episode.
        reference_length = _as_int(policy_traj[f"episode_{episode}_traj_length"])
        subject_id = None
        trial_id = None
        if subject_based:
            subject_id = subject_from_trajectory(traj_id_int, motion_list)
            trial_id = trial_from_trajectory(traj_id_int, motion_list)

        # Ordered gait events + compressed feet-contact phases for this episode,
        # recorded for every rollout (including early-terminated ones) so they can
        # be dumped to the per-rollout event-sequence text file.
        event_seq, phase_seq = episode_event_summary(policy_traj, episode)
        episode_length = int(policy_traj[f"episode_{episode}_joint_positions"].shape[0])

        if not _episode_successful(policy_traj, episode):
            skipped_episodes.append(
                {
                    "episode": int(episode),
                    "traj_id": traj_id_int,
                    "subject_id": subject_id,
                    "reason": "early_termination",
                    "episode_length": int(policy_traj[f"episode_{episode}_joint_positions"].shape[0]),
                    "reference_length": reference_length,
                }
            )
            episode_gait_records.append(
                {
                    "episode": int(episode),
                    "traj_id": traj_id_int,
                    "subject_id": subject_id,
                    "n_gaits": 0,
                    "gait_indexes": [],
                    "reference_length": reference_length,
                    "episode_length": episode_length,
                    "reason": "early_termination",
                    "event_seq": event_seq,
                    "phase_seq": phase_seq,
                }
            )
            continue

        (
            episode_joint_gaits_myo,
            episode_joint_gaits_myo_ref,
            episode_muscle_gaits_myo,
            episode_gait_lengths,
            episode_gait_indexes,
        ) = extract_episode_gaits(
            channels,
            policy_traj,
            episode,
            expected_n=max(1, round(reference_length / 100)),
            activation_phase=activation_phase,
            muscle_signal=muscle_signal,
        )

        if len(episode_gait_lengths) > 0:
            episode_joint_gaits_myo = (
                np.array(episode_joint_gaits_myo).reshape(len(channels.joint_names), -1, 100).transpose(1, 0, 2)
            )
            n_gaits_in_episode = episode_joint_gaits_myo.shape[0]
            episode_gait_records.append(
                {
                    "episode": int(episode),
                    "traj_id": traj_id_int,
                    "subject_id": subject_id,
                    "n_gaits": int(n_gaits_in_episode),
                    "gait_indexes": episode_gait_indexes,
                    "reference_length": reference_length,
                    "episode_length": episode_length,
                    "reason": "ok",
                    "event_seq": event_seq,
                    "phase_seq": phase_seq,
                }
            )
            all_joint_gaits_myo.extend(episode_joint_gaits_myo)
            all_gait_positions.extend(range(n_gaits_in_episode))
            if subject_based:
                all_subject_ids.extend([subject_id] * n_gaits_in_episode)
                all_trial_ids.extend([trial_id] * n_gaits_in_episode)

            episode_joint_gaits_myo_ref = (
                np.array(episode_joint_gaits_myo_ref).reshape(len(channels.joint_names), -1, 100).transpose(1, 0, 2)
            )
            all_joint_gaits_myo_ref.extend(episode_joint_gaits_myo_ref)

            episode_muscle_gaits_myo = (
                np.array(episode_muscle_gaits_myo).reshape(len(channels.muscle_names), -1, 100).transpose(1, 0, 2)
            )
            all_muscle_gaits_myo.extend(episode_muscle_gaits_myo)
            all_gait_lengths.extend(episode_gait_lengths)
        else:
            skipped_episodes.append(
                {
                    "episode": int(episode),
                    "traj_id": traj_id_int,
                    "subject_id": subject_id,
                    "reason": "no_valid_gait_cycles",
                    "episode_length": int(policy_traj[f"episode_{episode}_joint_positions"].shape[0]),
                    "reference_length": reference_length,
                }
            )
            episode_gait_records.append(
                {
                    "episode": int(episode),
                    "traj_id": traj_id_int,
                    "subject_id": subject_id,
                    "n_gaits": 0,
                    "gait_indexes": [],
                    "reference_length": reference_length,
                    "episode_length": episode_length,
                    "reason": "no_valid_gait_cycles",
                    "event_seq": event_seq,
                    "phase_seq": phase_seq,
                }
            )

    # Convert to arrays
    all_joint_gaits_arr = (
        np.array(all_joint_gaits_myo) if all_joint_gaits_myo else np.empty((0, len(channels.joint_names), 100))
    )
    all_joint_gaits_ref_arr = (
        np.array(all_joint_gaits_myo_ref) if all_joint_gaits_myo_ref else np.empty((0, len(channels.joint_names), 100))
    )
    all_muscle_gaits_arr = (
        np.array(all_muscle_gaits_myo) if all_muscle_gaits_myo else np.empty((0, len(channels.muscle_names), 100))
    )
    if not np.isfinite(all_muscle_gaits_arr).all():
        raise ValueError(
            "Activation gait extraction produced a non-finite cycle; "
            "incomplete cycles must be excluded before interpolation."
        )
    all_gait_lengths_arr = np.array(all_gait_lengths) if all_gait_lengths else np.empty(0, dtype=int)
    all_gait_positions_arr = np.array(all_gait_positions, dtype=int) if all_gait_positions else np.empty(0, dtype=int)

    print(f"  Policy Joint Gaits Shape: {all_joint_gaits_arr.shape} (gait_cycles, joints, time_steps)")
    print(f"  Policy Muscle Gaits Shape: {all_muscle_gaits_arr.shape} (gait_cycles, muscles, time_steps)")

    if subject_based:
        all_subject_ids_arr = np.array(all_subject_ids) if all_subject_ids else np.empty(0, dtype=int)
        all_trial_ids_arr = np.array(all_trial_ids, dtype=int) if all_trial_ids else np.empty(0, dtype=int)
        per_subject_avg_muscle_gaits = {}
        per_subject_avg_joint_gaits = {}
        per_subject_avg_joint_gaits_ref = {}
        per_subject_avg_muscle_rms = {}

        unique_subjects = np.unique(all_subject_ids_arr)
        for subj_id in unique_subjects:
            subj_mask = all_subject_ids_arr == subj_id
            if np.any(subj_mask):
                per_subject_avg_muscle_gaits[subj_id] = np.mean(all_muscle_gaits_arr[subj_mask], axis=0)
                per_subject_avg_joint_gaits[subj_id] = np.mean(all_joint_gaits_arr[subj_mask], axis=0)
                per_subject_avg_joint_gaits_ref[subj_id] = np.mean(all_joint_gaits_ref_arr[subj_mask], axis=0)
                per_subject_avg_muscle_rms[subj_id] = compute_rsm(per_subject_avg_muscle_gaits[subj_id])

    else:
        all_subject_ids_arr = None
        all_trial_ids_arr = None
        per_subject_avg_muscle_gaits = None
        per_subject_avg_joint_gaits = None
        per_subject_avg_joint_gaits_ref = None
        per_subject_avg_muscle_rms = None

    # Compute statistics only if data exists (avoid warnings on empty arrays)
    if all_joint_gaits_arr.shape[0] > 0:
        if subject_based:
            # Equal-subject-weighting: average each subject's cycles first, then
            # average those per-subject profiles over subjects. This gives every
            # subject the same weight in the population profile (matching the
            # one-profile-per-subject human data), rather than letting subjects
            # with more retained gait cycles dominate a cycle-pooled mean.
            avg_joint_gaits = np.mean(list(per_subject_avg_joint_gaits.values()), axis=0)
            avg_joint_gaits_ref = np.mean(list(per_subject_avg_joint_gaits_ref.values()), axis=0)
            avg_muscle_gaits = np.mean(list(per_subject_avg_muscle_gaits.values()), axis=0)
        else:
            avg_joint_gaits = np.mean(all_joint_gaits_arr, axis=0)
            avg_joint_gaits_ref = np.mean(all_joint_gaits_ref_arr, axis=0)
            avg_muscle_gaits = np.mean(all_muscle_gaits_arr, axis=0)
        std_joint_gaits = np.std(all_joint_gaits_arr, axis=0)
        std_joint_gaits_ref = np.std(all_joint_gaits_ref_arr, axis=0)
        std_muscle_gaits = np.std(all_muscle_gaits_arr, axis=0)
        population_muscle_rsm = compute_rsm(avg_muscle_gaits)
    else:
        # Return NaN arrays for empty data
        avg_joint_gaits = np.full((len(channels.joint_names), 100), np.nan)
        std_joint_gaits = np.full((len(channels.joint_names), 100), np.nan)
        avg_joint_gaits_ref = np.full((len(channels.joint_names), 100), np.nan)
        std_joint_gaits_ref = np.full((len(channels.joint_names), 100), np.nan)
        avg_muscle_gaits = np.full((len(channels.muscle_names), 100), np.nan)
        std_muscle_gaits = np.full((len(channels.muscle_names), 100), np.nan)
        population_muscle_rsm = np.full((len(channels.muscle_names), len(channels.muscle_names)), np.nan)

    policy_traj.close()
    return ExtractedRollout(
        profiles=PolicyProfiles(
            avg_joint_gaits=avg_joint_gaits,
            std_joint_gaits=std_joint_gaits,
            avg_joint_gaits_ref=avg_joint_gaits_ref,
            std_joint_gaits_ref=std_joint_gaits_ref,
            avg_muscle_gaits=avg_muscle_gaits,
            std_muscle_gaits=std_muscle_gaits,
            population_muscle_rsm=population_muscle_rsm,
            subject_ids=all_subject_ids_arr,
            per_subject_muscle=per_subject_avg_muscle_gaits,
            per_subject_joint=per_subject_avg_joint_gaits,
            per_subject_joint_ref=per_subject_avg_joint_gaits_ref,
            per_subject_muscle_rsm=per_subject_avg_muscle_rms,
        ),
        # Per-cycle arrays at every gait position, kept so the analysis can later
        # be restricted to one position by ``human_data.select_gait_subset``.
        cycles=PolicyCycles(
            joint=all_joint_gaits_arr if subject_based else None,
            joint_ref=all_joint_gaits_ref_arr if subject_based else None,
            muscle=all_muscle_gaits_arr if subject_based else None,
            subject_ids=all_subject_ids_arr if subject_based else None,
            trial_ids=all_trial_ids_arr if subject_based else None,
            gait_positions=all_gait_positions_arr if subject_based else None,
        ),
        all_gait_lengths=all_gait_lengths_arr,
        skipped_episodes=skipped_episodes,
        episode_gait_records=episode_gait_records,
    )


def load_policy_run(
    channels: GaitChannels,
    policy: PolicySpec,
    *,
    subject_based: bool,
    motion_list=None,
    muscle_signal="activation",
) -> PolicyRun:
    """Load one validated rollout and its optional earlier-trial companion."""
    if not isinstance(policy, PolicySpec):
        raise TypeError(
            "load_policy_run expects a validated PolicySpec; load YAML policies with FullBodyConfig.load_policies()."
        )

    print(f" ### --------------------Loading Policy Data: {policy.label}--------------------- ### ")
    extracted = extract_trajectory_data(
        channels,
        policy.trajectory_path,
        subject_based=subject_based,
        motion_list=motion_list,
        muscle_signal=muscle_signal,
    )

    # Optional trials-1-4 rollouts for the policy within-subject variability reference:
    # the same policy rolled out tracking this subject's earlier trials (1-4) instead of
    # the tested trial 5. Loaded and gait-segmented exactly like the trial-5 rollouts;
    # only the per-cycle arrays are kept, to be re-binned per gait position in
    # select_gait_subset (mirroring the trial-5 rebuild). Subject-based (Gait120) only.
    earlier_trials = None
    if policy.earlier_trials_path is not None:
        if not subject_based:
            raise ValueError(
                f"Policy {policy.id!r} configures Gait120 trials 1-4 rollouts for a population-only dataset."
            )
        motion_list_name = policy.earlier_trials_motion_set
        if motion_list_name not in MOTION_SETS:
            supported = ", ".join(sorted(MOTION_SETS))
            raise ValueError(
                f"Policy {policy.id!r} uses unknown trials 1-4 motion list "
                f"{motion_list_name!r}; choose one of: {supported}."
            )
        trials_1_4_motion_list = MOTION_SETS[motion_list_name]
        print(f"  Loading motion list for trials 1-4 rollouts: {motion_list_name}")
        print(f"  Motion list for trials 1-4 rollouts has {len(trials_1_4_motion_list)} entries.")
        print(f"  Loading trials 1-4 rollouts (policy within-subject variability): {policy.earlier_trials_path}")
        earlier_trials = extract_trajectory_data(
            channels,
            policy.earlier_trials_path,
            subject_based=subject_based,
            motion_list=trials_1_4_motion_list,
            muscle_signal=muscle_signal,
        ).cycles

    if extracted.all_gait_lengths.shape[0] > 0:
        print(f"  Extracted {extracted.all_gait_lengths.shape[0]} gait cycles.")
        if subject_based and extracted.profiles.subject_ids is not None:
            for subj_id in np.unique(extracted.profiles.subject_ids):
                n_gaits_subj = np.sum(extracted.profiles.subject_ids == subj_id)
                print(f"    Subject {subj_id}: {n_gaits_subj} gait cycles.")
        if extracted.skipped_episodes:
            skipped_counts = {}
            for skipped in extracted.skipped_episodes:
                skipped_counts[skipped["reason"]] = skipped_counts.get(skipped["reason"], 0) + 1
            skipped_summary = ", ".join(f"{reason}: {count}" for reason, count in sorted(skipped_counts.items()))
            print(f"  Skipped episodes: {len(extracted.skipped_episodes)} ({skipped_summary})")
    else:
        print("  No gait cycles were extracted.")

    return PolicyRun(
        policy=policy,
        cycles=extracted.cycles,
        profiles=extracted.profiles,
        all_gait_lengths=extracted.all_gait_lengths,
        skipped_episodes=extracted.skipped_episodes,
        episode_gait_records=extracted.episode_gait_records,
        earlier_trials=earlier_trials,
    )
