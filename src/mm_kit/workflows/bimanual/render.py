"""ULTra-MoCap adapter for the dataset-neutral video renderer."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mm_kit.activation import activation_sample_phase, align_activation_to_observation_timeline
from mm_kit.datasets.ultra_mocap import find_preprocessed_emg, load_aligned_emg
from mm_kit.paths import resolve_dataset_path
from mm_kit.rendering.models import CameraConfig, RenderClip, SignalComparison, TendonSignal
from mm_kit.workflows.bimanual.segmentation import (
    complete_cycle_boundaries,
    movement_from_trial_stem,
    select_declared_critical_joint,
)

from .constants import CONTROL_FREQUENCY_HZ, EMG_MUSCLE_GROUPS
from .muscles import load_muscle_names

CHANNELS = ("IM EMG4", "IM EMG5", "IM EMG6")
COLORS = ("#4c9aff", "#ff9f43", "#45d483")
TENDON_COLORS = (
    (0.30, 0.60, 1.00),
    (1.00, 0.46, 0.14),
    (0.28, 1.00, 0.50),
)


def prepare_clip(
    *,
    trajectory_path: str | Path,
    episode: int,
    trial_stem: str,
    dataset_dir: str | Path | None = None,
    cycles: int = 4,
    cycle_offset: int = 0,
    camera_azimuth: float = 90.0,
    camera_elevation: float = -8.0,
    camera_distance: float = 0.88,
) -> RenderClip:
    """Prepare a clock-synchronized ULTra-MoCap render clip."""
    if cycles < 1:
        raise ValueError("cycles must be at least one.")
    if cycle_offset < 0:
        raise ValueError("cycle_offset cannot be negative.")
    source = Path(trajectory_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Rollout archive does not exist: {source}")
    root = resolve_dataset_path("ultra-mocap", dataset_dir)

    with np.load(source, allow_pickle=True) as archive:
        prefix = f"episode_{episode}"
        qpos = np.asarray(archive[f"{prefix}_joint_positions"], dtype=float)
        exported = np.asarray(archive[f"{prefix}_muscle_activations"], dtype=float)
        reference = np.asarray(archive[f"{prefix}_traj_qpos"], dtype=float)[: len(qpos)]
        joint_names = [str(name) for name in archive["joint_names"]]
        activation = align_activation_to_observation_timeline(
            exported,
            activation_sample_phase(archive),
        )
    if len(activation) != len(qpos):
        raise ValueError("The rollout must contain one activation row per qpos row.")

    movement = movement_from_trial_stem(trial_stem)
    selected = select_declared_critical_joint(
        reference * 180.0 / np.pi,
        joint_names,
        movement,
        arm="right",
        fs=100.0,
    )
    if selected is None:
        raise RuntimeError(f"Could not segment periodic cycles for {trial_stem}.")

    subject = trial_stem.split("_")[0]
    emg_path = find_preprocessed_emg(subject, trial_stem, [root])
    aligned_emg = load_aligned_emg(emg_path, len(qpos), CONTROL_FREQUENCY_HZ, required_channels=CHANNELS)
    human = aligned_emg[list(CHANNELS)].to_numpy(dtype=float)
    muscle_names = load_muscle_names()
    group_columns = [[muscle_names.index(name) for name in EMG_MUSCLE_GROUPS[channel][1]] for channel in CHANNELS]
    policy = np.column_stack([activation[:, columns].mean(axis=1) for columns in group_columns])
    valid = complete_cycle_boundaries(np.concatenate((human, policy), axis=1), selected[3])
    chosen = valid[cycle_offset : cycle_offset + cycles]
    if len(chosen) != cycles:
        raise ValueError(
            f"Requested cycles {cycle_offset}:{cycle_offset + cycles}, but only {len(valid)} complete cycles exist."
        )
    start, stop = chosen[0][0], chosen[-1][1]
    qpos_clip = qpos[start : stop + 1]
    activation_clip = activation[start : stop + 1]
    human_clip = human[start : stop + 1]
    policy_clip = policy[start : stop + 1]
    x = np.arange(len(qpos_clip), dtype=float) / 100.0
    boundaries = (*((boundary[0] - start) / 100.0 for boundary in chosen), x[-1])

    labels = (
        "Biceps Brachii | BIClong + BICshort",
        "Triceps Brachii | TRIlong",
        "Middle Deltoid | DELT2",
    )
    comparisons = tuple(
        SignalComparison(
            label=label,
            human=human_clip[:, index],
            policy=policy_clip[:, index],
            color=COLORS[index],
            human_label="Recorded sEMG",
            policy_label="Artificial activation",
        )
        for index, label in enumerate(labels)
    )
    tendon_signals = tuple(
        TendonSignal(
            tendon_name=f"{muscle}_tendon",
            activation_index=muscle_names.index(muscle),
            color=TENDON_COLORS[channel_index],
        )
        for channel_index, channel in enumerate(CHANNELS)
        for muscle in EMG_MUSCLE_GROUPS[channel][1]
    )
    return RenderClip(
        qpos=qpos_clip,
        activations=activation_clip,
        sample_hz=100.0,
        x=x,
        x_label="clip time (s)",
        comparisons=comparisons,
        model_name="bimanual",
        camera=CameraConfig(
            body_names=("thorax", "humerus_r", "ulna_r", "capitate_r"),
            distance=camera_distance,
            azimuth=camera_azimuth,
            elevation=camera_elevation,
        ),
        title="Synchronized neuromuscular signals",
        subtitle=f"{trial_stem} | recorded right-arm sEMG",
        tendon_signals=tendon_signals,
        boundaries=boundaries,
        metadata={"dataset": "ultra-mocap", "trial": trial_stem, "emg_path": str(emg_path)},
    )
