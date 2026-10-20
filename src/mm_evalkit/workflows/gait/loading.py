"""Assemble a validated :class:`GaitAnalysis` from a configuration.

Loading is a function rather than a constructor so the state object can be built
directly in tests without touching a dataset, and so every file this run reads or
writes is visible in one place.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mm_evalkit.datasets.gait120 import Gait120_muscle_map, Gait120_muscle_names
from mm_evalkit.datasets.wang import Wang_muscle_map

from . import diagnostics, human_data, kinetics_data, policy_data
from .channels import (
    common_joint_names,
    common_muscle_names,
    joint_names_myo,
    muscle_names_myo,
)
from .config import FullBodyConfig
from .constants import MOTION_SETS
from .motion_identity import subject_from_trajectory
from .rollout_validation import validate_policy_rollouts
from .state import GaitAnalysis, GaitChannels, PlotStyle


def required_indices(requested, available, *, signal: str) -> list[int]:
    """Resolve model columns and fail once with every missing channel."""
    missing = [name for name in requested if name not in available]
    if missing:
        raise ValueError(f"MyoFullBody is missing required {signal} channel(s): {', '.join(missing)}.")
    return [available.index(name) for name in requested]


def build_channels(dataset_name: str) -> GaitChannels:
    """Select the joint and muscle channels this dataset can be compared on.

    Gait120 records every muscle that has a MyoFullBody counterpart; Wang is
    compared on the subset both datasets share.
    """
    joint_names = list(common_joint_names)
    muscle_names = list(Gait120_muscle_names if dataset_name == "Gait120" else common_muscle_names)
    return GaitChannels(
        joint_names=tuple(joint_names),
        muscle_names=tuple(muscle_names),
        joint_indices_myo=tuple(required_indices(joint_names, joint_names_myo, signal="joint")),
        muscle_indices_myo=tuple(required_indices(muscle_names, muscle_names_myo, signal="muscle")),
    )


def build_style(kinematics_source: str) -> PlotStyle:
    """Name each reference band after the ground truth it was computed from."""
    retargeted = kinematics_source == "mfb-retargeted"
    return PlotStyle(
        kin_gt_label="MFB-retargeted" if retargeted else "OpenSim-IK",
        kin_gt_pair_label=(
            "Dataset variability \n (Pairwise MFB-retargeted)"
            if retargeted
            else "Dataset variability \n (Pairwise human-human)"
        ),
        muscle_gt_pair_label="Dataset variability \n (Pairwise human-human)",
    )


def prepare_output_dir(config: FullBodyConfig) -> Path:
    """Create this run's output directory, refusing to mix in stale artifacts."""
    output_dir = config.plots_root / config.dataset_name
    if output_dir.exists():
        if not output_dir.is_dir():
            raise NotADirectoryError(f"Full-body output path is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise FileExistsError(
                f"Full-body output directory is not empty: {output_dir}. "
                "Choose a new plots directory or empty this directory before running; "
                "the analysis will not mix new results with stale artifacts."
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def requested_subjects(motion_list) -> list[int] | None:
    """Subjects named by the configured motion list, in ascending order."""
    if motion_list is None:
        return None
    return np.unique([subject_from_trajectory(index, motion_list) for index in range(len(motion_list))]).tolist()


def load_gait_analysis(config: FullBodyConfig) -> GaitAnalysis:
    """Validate every configured input, then load the rollouts and human data."""
    policy_specs = config.load_policies()
    output_dir = prepare_output_dir(config)
    channels = build_channels(config.dataset_name)
    muscle_map = Gait120_muscle_map if config.dataset_name == "Gait120" else Wang_muscle_map
    motion_list = MOTION_SETS[config.motion_set_name] if config.motion_set_name is not None else None

    validate_policy_rollouts(
        policy_specs,
        subject_based=config.subject_based,
        motion_count=(len(motion_list) if motion_list is not None else None),
        expected_joint_names=tuple(joint_names_myo),
        expected_joint_width=len(joint_names_myo) + 6,
        expected_activation_width=len(muscle_names_myo),
        expected_motion_group=config.motion_set_name,
    )
    policies = [
        policy_data.load_policy_run(
            channels,
            policy,
            subject_based=config.subject_based,
            motion_list=motion_list,
            emg_signal=config.emg_signal,
        )
        for policy in policy_specs
    ]
    cohort = human_data.load_human_cohort(
        config,
        channels,
        muscle_map,
        list(range(1, 121)) if config.dataset_name == "Gait120" and not config.subject_based
        else requested_subjects(motion_list),
        output_dir,
    )
    kinetics = kinetics_data.load_kinetics_cohort(
        policies,
        reference_dir=config.kinetics_reference_dir,
        motion_list=motion_list,
    )
    kinetics_data.write_kinetics_omission_audit(policies, output_dir)

    analysis = GaitAnalysis(
        config=config,
        channels=channels,
        style=build_style(config.kinematics_source),
        human=cohort,
        policies=policies,
        plots_dir=output_dir,
        motion_list=motion_list,
        muscle_map=muscle_map,
        kinetics=kinetics,
    )
    human_data.choose_plot_subjects(cohort, analysis.style.max_plot_subjects)
    diagnostics.build_and_save_gait_count_table(analysis)
    return analysis
