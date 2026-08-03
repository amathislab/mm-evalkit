"""Builders for a GaitAnalysis that touches no dataset.

Every field the stages read is declared on the state records, so a test can
assemble exactly the analysis it needs instead of loading a rollout or
bypassing the constructor.
"""

from pathlib import Path

import numpy as np
import pytest

from mm_evalkit.policies import PolicySpec
from mm_evalkit.workflows.gait.config import FullBodyConfig
from mm_evalkit.workflows.gait.loading import build_style
from mm_evalkit.workflows.gait.state import (
    GaitAnalysis,
    GaitChannels,
    HumanCohort,
    PolicyCycles,
    PolicyProfiles,
    PolicyRun,
)

JOINTS = ["hip", "knee", "ankle"]
MUSCLES = ["m1", "m2", "m3", "m4"]
SUBJECTS = [51, 52, 53, 54, 55, 56, 57, 58]


def policy_spec(label: str) -> PolicySpec:
    """Typed policy metadata for a fixture; the rollout file is never opened."""
    return PolicySpec(
        id=label.replace(" ", "-").replace("\n", "-"),
        label=label,
        trajectory_path=Path("unused-rollout.npz"),
        color="#2a78d6",
    )


def make_channels(joints=JOINTS, muscles=MUSCLES) -> GaitChannels:
    """Channels whose model indices are the selection's own order."""
    return GaitChannels(
        joint_names=tuple(joints),
        muscle_names=tuple(muscles),
        joint_indices_myo=tuple(range(len(joints))),
        muscle_indices_myo=tuple(range(len(muscles))),
    )


def make_config(
    *,
    dataset_name="Gait120",
    subject_based=True,
    kinematics_source="opensim-ik",
    plots_root=Path("unused-plots"),
) -> FullBodyConfig:
    """A validated-shape config built directly, without resolving any path."""
    return FullBodyConfig(
        dataset_name=dataset_name,
        dataset_dir=Path("unused-dataset"),
        policy_config_file=Path("unused-policies.yaml"),
        plots_root=plots_root,
        subject_based=subject_based,
        motion_set_name=None,
        kinematics_source=kinematics_source,
    )


def make_cohort(subjects=SUBJECTS, *, joints=JOINTS, muscles=MUSCLES, by_step=None) -> HumanCohort:
    """A cohort whose profiles are flat zeros unless per-step data is supplied."""
    joint_data = np.zeros((len(subjects), len(joints), 100))
    muscle_data = np.zeros((len(subjects), len(muscles), 100))
    empty: dict[int, dict[int, np.ndarray]] = {}
    joint_by_step, muscle_by_step = by_step if by_step is not None else (empty, empty)
    return HumanCohort(
        all_subjects=list(subjects),
        joint_names=list(joints),
        muscle_names=list(muscles),
        joint_by_step=joint_by_step,
        muscle_by_step=muscle_by_step,
        earlier_joint_by_step=empty,
        earlier_muscle_by_step=empty,
        omissions=(),
        subjects=list(subjects),
        joint_data=joint_data,
        muscle_data=muscle_data,
    )


def empty_profiles(*, joints=JOINTS, muscles=MUSCLES) -> PolicyProfiles:
    """Profiles shaped like a real policy but carrying no retained cycles."""
    return PolicyProfiles(
        avg_joint_gaits=np.zeros((len(joints), 100)),
        std_joint_gaits=np.zeros((len(joints), 100)),
        avg_joint_gaits_ref=np.zeros((len(joints), 100)),
        std_joint_gaits_ref=np.zeros((len(joints), 100)),
        avg_muscle_gaits=np.zeros((len(muscles), 100)),
        std_muscle_gaits=np.zeros((len(muscles), 100)),
        population_muscle_rsm=np.full((len(muscles), len(muscles)), np.nan),
    )


def make_run(label, *, cycles=None, profiles=None) -> PolicyRun:
    """One policy with no metrics computed yet."""
    return PolicyRun(
        policy=policy_spec(label),
        cycles=cycles or PolicyCycles(None, None, None, None, None, None),
        profiles=profiles or empty_profiles(),
        all_gait_lengths=np.empty(0, dtype=int),
        skipped_episodes=[],
        episode_gait_records=[],
    )


def make_analysis(*, policies=(), cohort=None, config=None, plots_dir=Path("unused-plots"), channels=None):
    """Assemble a GaitAnalysis from the pieces a test cares about."""
    resolved_config = config or make_config()
    return GaitAnalysis(
        config=resolved_config,
        channels=channels or make_channels(),
        style=build_style(resolved_config.kinematics_source),
        human=cohort if cohort is not None else make_cohort(),
        policies=list(policies),
        plots_dir=Path(plots_dir),
        motion_list=None,
        muscle_map={},
    )


@pytest.fixture
def analysis_factory():
    return make_analysis
