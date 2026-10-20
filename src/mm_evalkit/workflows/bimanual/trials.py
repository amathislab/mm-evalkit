"""Alignment and repetition extraction for one ULTRA-MoCap trial."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from mm_evalkit.activation import load_emg_signal
from mm_evalkit.datasets.ultra_mocap import find_preprocessed_emg, load_aligned_emg
from mm_evalkit.workflows.bimanual.segmentation import (
    complete_cycle_boundaries,
    condition_from_trial_stem,
    movement_from_trial_stem,
    resample_cycles,
    select_declared_critical_joint,
)

from .constants import CONTROL_FREQUENCY_HZ
from .models import Arm, MuscleGroup, TrialData


def load_trial(
    trajectory: np.lib.npyio.NpzFile,
    episode: int,
    trial_stem: str,
    joint_names: Sequence[str],
    muscle_columns: Sequence[int],
    groups: Sequence[MuscleGroup],
    arm: Arm,
    emg_roots: Sequence[Path],
    *,
    emg_signal: str = "ctrl",
) -> TrialData:
    """Align and segment one rollout episode.

    Repetitions are detected from the reference kinematics. The sEMG file and
    every mapped electrode are required. Only repetitions fully covered by all
    electrodes are used for both sEMG and policy-activation averages.
    """
    subject = trial_stem.split("_")[0]
    movement = movement_from_trial_stem(trial_stem)
    condition = condition_from_trial_stem(trial_stem)
    prefix = f"episode_{episode}"
    reference_frames = int(np.asarray(trajectory[f"{prefix}_traj_length"]).item())

    policy_qpos = np.asarray(trajectory[f"{prefix}_joint_positions"], dtype=float) * 180.0 / math.pi
    reference_qpos = np.asarray(trajectory[f"{prefix}_traj_qpos"], dtype=float) * 180.0 / math.pi
    policy_activation = load_emg_signal(trajectory, episode, emg_signal)

    rollout_frames = len(policy_qpos)
    reference_qpos = reference_qpos[:rollout_frames]
    policy_activation = policy_activation[:rollout_frames]
    early_stop_label = (
        f" [early stop {rollout_frames}/{reference_frames}]" if rollout_frames < reference_frames - 1 else ""
    )
    print(f"\nEpisode {episode}: {trial_stem}  rollout={rollout_frames}  ref={reference_frames}{early_stop_label}")

    channels = tuple(group.channel for group in groups)
    emg_path = find_preprocessed_emg(subject, trial_stem, emg_roots)
    emg = load_aligned_emg(
        emg_path,
        rollout_frames,
        CONTROL_FREQUENCY_HZ,
        required_channels=channels,
    )
    missing_frames = int(emg[list(channels)].isna().any(axis=1).sum())
    print(f"  sEMG: {emg_path.name} ({len(emg)} aligned samples, {missing_frames} incomplete)")

    selected = select_declared_critical_joint(
        reference_qpos,
        list(joint_names),
        movement,
        arm=arm,
        fs=CONTROL_FREQUENCY_HZ,
    )
    if selected is None:
        print(
            f"  no sufficiently periodic critical joint for {movement!r}; "
            "excluding the trial from cycle-level analysis."
        )
        return TrialData(
            episode=episode,
            stem=trial_stem,
            subject=subject,
            movement=movement,
            condition=condition,
            rollout_frames=rollout_frames,
            reference_frames=reference_frames,
            early_stop_label=early_stop_label,
            policy_qpos=policy_qpos,
            reference_qpos=reference_qpos,
            policy_activation=policy_activation,
            emg=emg,
            exclusion_reason="no sufficiently periodic critical joint",
            cycle_boundaries=None,
            muscle_cycle_boundaries=None,
            activation_cycles=None,
            emg_cycles=None,
        )

    critical_joint, score, period, detected_boundaries = selected
    cycle_boundaries = tuple(detected_boundaries)
    print(
        f"  segmented {len(cycle_boundaries)} {movement!r} repetitions on "
        f"{critical_joint} (periodicity {score:.2f}, "
        f"period {period / CONTROL_FREQUENCY_HZ:.1f}s)."
    )

    emg_matrix = emg[list(channels)].to_numpy(dtype=float)
    # One shared boundary set must be completely observed in both modalities.
    # This also excludes state zero from historical post-step exports.
    paired_muscle_data = np.concatenate(
        (emg_matrix, policy_activation[:, muscle_columns]),
        axis=1,
    )
    muscle_boundaries = tuple(complete_cycle_boundaries(paired_muscle_data, cycle_boundaries))
    if not muscle_boundaries:
        raise ValueError(
            f"{emg_path} covers none of the {len(cycle_boundaries)} reference repetitions for {trial_stem!r}."
        )
    if len(muscle_boundaries) != len(cycle_boundaries):
        print(f"  sEMG covers {len(muscle_boundaries)}/{len(cycle_boundaries)} complete reference repetitions.")
    emg_cycles = resample_cycles(emg_matrix, muscle_boundaries)

    activation_cycles = resample_cycles(
        policy_activation[:, muscle_columns],
        muscle_boundaries,
    )
    return TrialData(
        episode=episode,
        stem=trial_stem,
        subject=subject,
        movement=movement,
        condition=condition,
        rollout_frames=rollout_frames,
        reference_frames=reference_frames,
        early_stop_label=early_stop_label,
        policy_qpos=policy_qpos,
        reference_qpos=reference_qpos,
        policy_activation=policy_activation,
        emg=emg,
        exclusion_reason=None,
        cycle_boundaries=cycle_boundaries,
        muscle_cycle_boundaries=muscle_boundaries,
        activation_cycles=activation_cycles,
        emg_cycles=emg_cycles,
    )
