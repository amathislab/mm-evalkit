"""Gait120 phase-synchronized adapter for the reusable video renderer."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mm_kit.activation import (
    activation_sample_phase,
    align_activation_to_observation_timeline,
    complete_activation_gait_indexes,
)
from mm_kit.datasets.gait120 import Gait120_muscle_map, load_gait120_emg_step
from mm_kit.paths import resolve_dataset_path
from mm_kit.rendering.models import CameraConfig, RenderClip, SignalComparison, TendonSignal
from mm_kit.rendering.timeline import resample_profile

from .channels import joint_names_myo, muscle_names_myo
from .constants import MOTION_SETS
from .cycles import extract_valid_strides_flexible, get_feet_contacts
from .motion_identity import subject_from_trajectory

DEFAULT_CHANNELS = ("vaslat_r", "bflh_r", "tibant_r", "gasmed_r")
COLORS = ("#4c9aff", "#ff9f43", "#45d483", "#b889ff", "#f4d35e", "#ef476f")


def prepare_clip(
    *,
    trajectory_path: str | Path,
    episode: int,
    motion_group: str = "GAIT120_TESTING",
    gait_index: int = 0,
    dataset_dir: str | Path | None = None,
    channels: tuple[str, ...] | list[str] = DEFAULT_CHANNELS,
    camera_azimuth: float = 105.0,
    camera_elevation: float = -12.0,
    camera_distance: float = 2.4,
) -> RenderClip:
    """Prepare one policy stride against Gait120's phase-normalized human EMG."""
    if gait_index not in {0, 1}:
        raise ValueError("gait_index must be 0 or 1 so it maps to Gait120 Step01 or Step02.")
    if motion_group not in MOTION_SETS:
        raise ValueError(f"Unknown Gait120 motion set {motion_group!r}.")
    selected_channels = tuple(channels)
    if not selected_channels or len(set(selected_channels)) != len(selected_channels):
        raise ValueError("channels must contain unique Gait120 model-muscle names.")
    missing = [name for name in selected_channels if name not in Gait120_muscle_map]
    if missing:
        raise ValueError(f"Unsupported Gait120 channel(s): {', '.join(missing)}.")

    source = Path(trajectory_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Rollout archive does not exist: {source}")
    root = resolve_dataset_path("gait120", dataset_dir)
    with np.load(source, allow_pickle=True) as archive:
        prefix = f"episode_{episode}"
        qpos = np.asarray(archive[f"{prefix}_joint_positions"], dtype=float)
        exported = np.asarray(archive[f"{prefix}_muscle_activations"], dtype=float)
        touch = np.asarray(archive[f"{prefix}_touch_observations"], dtype=float)
        traj_id = int(np.asarray(archive[f"{prefix}_traj_id"]).item())
        reference_length = int(np.asarray(archive[f"{prefix}_traj_length"]).item())
        activation = align_activation_to_observation_timeline(
            exported,
            activation_sample_phase(archive),
        )
    if len(qpos) != len(activation) or len(qpos) != len(touch):
        raise ValueError("Gait120 qpos, activation, and touch observations must share one frame axis.")

    contacts = get_feet_contacts(touch_sensors_data=touch, threshold=0.05)
    strides = extract_valid_strides_flexible(
        contacts,
        side="right",
        expected_n=round(reference_length / 100),
    )
    indexes = complete_activation_gait_indexes(
        activation,
        [(int(stride.start), int(stride.end)) for stride in strides],
    )
    if gait_index >= len(indexes):
        raise ValueError(f"Episode {episode} contains only {len(indexes)} complete activation-aligned strides.")
    start, stop = indexes[gait_index]
    qpos_clip = qpos[start:stop]
    activation_clip = activation[start:stop]
    if len(qpos_clip) < 2:
        raise ValueError("The selected Gait120 stride has fewer than two frames.")

    motion_list = MOTION_SETS[motion_group]
    subject = subject_from_trajectory(traj_id, motion_list)
    human_profiles = load_gait120_emg_step(
        root,
        subject=subject,
        step_index=gait_index,
        selected_muscle_names=selected_channels,
        muscle_map=Gait120_muscle_map,
    )
    columns = [muscle_names_myo.index(name) for name in selected_channels]
    x = np.linspace(0.0, 100.0, len(qpos_clip))
    comparisons = tuple(
        SignalComparison(
            label=f"{name} | {Gait120_muscle_map[name]}",
            human=resample_profile(human_profiles[index], len(qpos_clip)),
            policy=activation_clip[:, columns[index]],
            color=COLORS[index % len(COLORS)],
            human_label="Gait120 human EMG",
            policy_label="Policy activation",
        )
        for index, name in enumerate(selected_channels)
    )
    tendon_signals = tuple(
        TendonSignal(
            tendon_name=f"{name}_tendon",
            activation_index=columns[index],
            color=tuple(
                int(COLORS[index % len(COLORS)].lstrip("#")[offset : offset + 2], 16) / 255 for offset in (0, 2, 4)
            ),
        )
        for index, name in enumerate(selected_channels)
    )
    return RenderClip(
        qpos=qpos_clip,
        activations=activation_clip,
        sample_hz=100.0,
        x=x,
        x_label="gait cycle (%)",
        comparisons=comparisons,
        model_name="myofullbody",
        camera=CameraConfig(
            body_names=("pelvis", "femur_r", "tibia_r", "calcn_r", "toes_r"),
            distance=camera_distance,
            azimuth=camera_azimuth,
            elevation=camera_elevation,
        ),
        title="Gait120 neuromuscular comparison",
        subtitle=f"S{subject:03d} | Step{gait_index + 1:02d} | phase-aligned (not shared clock time)",
        joint_names=tuple(joint_names_myo),
        activation_names=tuple(muscle_names_myo),
        tendon_signals=tendon_signals,
        boundaries=(0.0, 100.0),
        metadata={
            "dataset": "gait120",
            "subject": f"S{subject:03d}",
            "step": f"Step{gait_index + 1:02d}",
            "motion_group": motion_group,
        },
    )
