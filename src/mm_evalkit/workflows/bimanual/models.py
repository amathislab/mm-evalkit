"""Typed records passed between the ULTra-MoCap analysis stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import pandas as pd

from mm_evalkit.policies import PolicySpec

Arm = Literal["left", "right"]
TrialKey = tuple[str, str, str]
CycleBoundary = tuple[int, int]


@dataclass(frozen=True)
class AnalysisConfig:
    """Validated inputs for one analysis run."""

    policies: tuple[PolicySpec, ...]
    emg_roots: tuple[Path, ...]
    motion_group: str
    motion_paths: tuple[str, ...]
    plots_dir: Path
    episodes: tuple[int, ...] | None
    arm: Arm
    all_joints: bool
    raw_trials: bool


@dataclass(frozen=True)
class RolloutMetadata:
    """Fields shared by all episodes in one validated rollout archive."""

    path: Path
    n_episodes: int
    dt: float
    joint_names: tuple[str, ...]
    activation_count: int


@dataclass(frozen=True)
class EpisodeExclusion:
    """A scientifically intentional episode exclusion."""

    episode: int
    stage: Literal["segmentation"]
    reason: str
    trial_stem: str | None = None
    detail: str = ""


@dataclass(frozen=True)
class MuscleGroup:
    """One sEMG electrode and its anatomically corresponding actuators."""

    channel: str
    label: str
    muscles: tuple[str, ...]
    activation_columns: tuple[int, ...]
    cycle_columns: tuple[int, ...]

    def title(self, *, list_muscles: bool = False) -> str:
        """Return a concise label, optionally naming the actuator members."""
        title = f"{self.label} ({self.channel})"
        if not list_muscles:
            return title
        target = f"mean of {', '.join(self.muscles)}" if len(self.muscles) > 1 else self.muscles[0]
        return f"{title}\n{target}"


@dataclass(frozen=True)
class TrialData:
    """Aligned arrays and repetition boundaries for one rollout episode."""

    episode: int
    stem: str
    subject: str
    movement: str
    condition: str
    rollout_frames: int
    reference_frames: int
    early_stop_label: str
    policy_qpos: np.ndarray
    reference_qpos: np.ndarray
    policy_activation: np.ndarray
    emg: pd.DataFrame
    exclusion_reason: str | None
    cycle_boundaries: tuple[CycleBoundary, ...] | None
    muscle_cycle_boundaries: tuple[CycleBoundary, ...] | None
    activation_cycles: np.ndarray | None
    emg_cycles: np.ndarray | None

    @property
    def key(self) -> TrialKey:
        """Canonical identity used to pair a trial across policies."""
        return self.movement, self.condition, self.subject

    @property
    def is_segmented(self) -> bool:
        """Whether a reliable periodic segmentation was found."""
        return self.cycle_boundaries is not None


class TrialMetrics(TypedDict, total=False):
    """Metric arrays and average cycles computed for one segmented trial."""

    n_joint_cycles: int
    n_muscle_cycles: int
    joint_rmse_by: np.ndarray
    joint_rmse: np.ndarray
    muscle_corr_by: np.ndarray
    muscle_corr: np.ndarray
    muscle_rsa: np.ndarray
    muscle_rsa_pearson: np.ndarray
    joint_rmse_within_by: np.ndarray
    joint_rmse_within: np.ndarray
    muscle_corr_within_by: np.ndarray
    muscle_corr_within: np.ndarray
    muscle_rsa_within: np.ndarray
    joint_rmse_pairwise_by: np.ndarray
    muscle_corr_pairwise_by: np.ndarray
    muscle_rsa_pairwise_by: np.ndarray
    ref_cycle: np.ndarray
    ref_std: np.ndarray
    pol_cycle: np.ndarray
    pol_std: np.ndarray
    head_means: np.ndarray
    emg_cycle: np.ndarray
    emg_std: np.ndarray
    act_cycle: np.ndarray
    act_std: np.ndarray
    emg_rsm: np.ndarray
    act_rsm: np.ndarray


Records = dict[TrialKey, TrialMetrics]


@dataclass(frozen=True)
class PolicyAnalysis:
    """Trials, metrics, and exclusions produced for one policy."""

    trials: dict[TrialKey, TrialData]
    records: Records
    exclusions: tuple[EpisodeExclusion, ...]
    requested_episodes: int
