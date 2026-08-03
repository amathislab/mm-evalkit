"""The state every gait stage reads and writes, declared in one place.

Stages are plain functions that take the :class:`GaitAnalysis` below and fill in
their own record. Declaring those records here -- rather than letting each stage
invent attributes -- means the schema can be read without tracing the pipeline,
and ``slots=True`` turns a misspelled field into an immediate ``AttributeError``
instead of a silently ignored write and a downstream ``KeyError``.

Records are grouped by what produces them:

* :class:`GaitChannels` and :class:`HumanCohort` come from loading.
* :class:`PolicyRun` holds one policy's cycles, its profiles for the currently
  selected gait, and every metric scored from them.
* :class:`HumanToHumanMetrics`, :class:`HumanReferences` and
  :class:`WilcoxonResults` are the analysis-level results.

Anything rebuilt per gait selection is mutable and starts as ``None``; anything
fixed for the whole run is frozen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from mm_evalkit.arrays import FloatArray, IntArray
from mm_evalkit.datasets.gait120 import Gait120Omission, StepProfiles
from mm_evalkit.policies import PolicySpec

from .config import FullBodyConfig
from .models import EpisodeGaitRecord, SkippedEpisode

if TYPE_CHECKING:
    from collections.abc import Sequence

# One value per channel name, plus an ``all_joints`` / ``all_muscles`` aggregate.
PerChannel = dict[str, list[float]]


@dataclass(frozen=True)
class GaitChannels:
    """The joint and muscle channels compared, and their model column indices.

    ``*_indices_myo`` index the MyoFullBody channel order; ``*_indexes`` index
    the selected subset itself and exist so plots can address the same channel
    in both spaces.
    """

    joint_names: tuple[str, ...]
    muscle_names: tuple[str, ...]
    joint_indices_myo: tuple[int, ...]
    muscle_indices_myo: tuple[int, ...]

    @property
    def joint_indexes(self) -> range:
        return range(len(self.joint_names))

    @property
    def muscle_indexes(self) -> range:
        return range(len(self.muscle_names))


@dataclass(frozen=True)
class PlotStyle:
    """Presentation choices that are not part of any scientific result."""

    kin_gt_label: str
    kin_gt_pair_label: str
    muscle_gt_pair_label: str
    max_plot_subjects: int = 5
    rsa_matrix_annotate_max: int = 20


@dataclass(slots=True)
class HumanCohort:
    """The human reference: the loaded source, and the selection now in use.

    The ``*_by_step`` profiles and ``all_subjects`` are the loaded dataset and do
    not change. ``subjects``, ``joint_data``, ``muscle_data`` and the RSMs are
    rebuilt by :func:`human_data.select_gait_subset` for each gait selection, so
    they always describe the analysis currently being reported.
    """

    all_subjects: list[int]
    joint_names: list[str]
    muscle_names: list[str]
    joint_by_step: StepProfiles
    muscle_by_step: StepProfiles
    earlier_joint_by_step: StepProfiles
    earlier_muscle_by_step: StepProfiles
    omissions: tuple[Gait120Omission, ...]

    subjects: list[int]
    joint_data: FloatArray
    muscle_data: FloatArray
    muscle_rsms: dict[int, FloatArray] = field(default_factory=dict)
    muscle_rsms_avg: FloatArray | None = None
    plot_subject_positions: list[int] = field(default_factory=list)

    @property
    def n_subjects(self) -> int:
        return int(self.joint_data.shape[0])


@dataclass(frozen=True)
class PolicyCycles:
    """Every retained gait cycle of one rollout, before any gait selection.

    All arrays share a leading cycle axis. They are ``None`` for a
    population-only dataset, which has no subject identity to bin cycles by.
    """

    joint: FloatArray | None
    joint_ref: FloatArray | None
    muscle: FloatArray | None
    subject_ids: IntArray | None
    trial_ids: IntArray | None
    gait_positions: IntArray | None


@dataclass(frozen=True)
class PolicyProfiles:
    """One policy's averaged profiles for the currently selected gait.

    Population profiles weight every subject equally: each subject's cycles are
    averaged first, then those per-subject profiles are averaged. Shapes are
    ``(n_channels, 100)`` for profiles and ``(n_muscles, n_muscles)`` for RSMs.
    """

    avg_joint_gaits: FloatArray
    std_joint_gaits: FloatArray
    avg_joint_gaits_ref: FloatArray
    std_joint_gaits_ref: FloatArray
    avg_muscle_gaits: FloatArray
    std_muscle_gaits: FloatArray
    population_muscle_rsm: FloatArray
    subject_ids: IntArray | None = None
    per_subject_muscle: dict[int, FloatArray] | None = None
    per_subject_joint: dict[int, FloatArray] | None = None
    per_subject_joint_ref: dict[int, FloatArray] | None = None
    per_subject_muscle_rsm: dict[int, FloatArray] | None = None


@dataclass(frozen=True)
class KineticsCycles:
    """Phase-normalized joint-moment and vertical-GRF gait cycles.

    All four signal arrays have a shared leading cycle axis and 100 phase
    points. Moment shapes are ``(n_cycles, 6, 100)`` and GRF shapes are
    ``(n_cycles, 2, 100)``. References are the OpenSim profiles matched to the
    same subject, trial, and gait window as each policy cycle.
    """

    torque: FloatArray
    torque_ref: FloatArray
    grf: FloatArray
    grf_ref: FloatArray
    subject_ids: IntArray
    trial_ids: IntArray
    gait_positions: IntArray


@dataclass(frozen=True)
class KineticsProfiles:
    """Population and per-subject kinetic profiles for one gait selection."""

    avg_torque: FloatArray
    std_torque: FloatArray
    avg_torque_ref: FloatArray
    std_torque_ref: FloatArray
    avg_grf: FloatArray
    std_grf: FloatArray
    avg_grf_ref: FloatArray
    std_grf_ref: FloatArray
    subject_ids: IntArray
    per_subject_torque: dict[int, FloatArray]
    per_subject_torque_ref: dict[int, FloatArray]
    per_subject_grf: dict[int, FloatArray]
    per_subject_grf_ref: dict[int, FloatArray]


@dataclass(frozen=True, slots=True)
class KineticsOmission:
    """One unavailable kinetics input or incompletely covered gait window."""

    policy_id: str
    episode: int
    reference_name: str
    gait_position: int
    reason: str


@dataclass(slots=True)
class KineticsCohort:
    """Human OpenSim kinetics, fixed cycles plus the active gait selection."""

    torque_names: tuple[str, ...]
    grf_names: tuple[str, ...]
    torque_unit: str
    grf_unit: str
    cycles: KineticsCycles
    reference_paths: tuple[Path, ...]
    subjects: list[int] = field(default_factory=list)
    torque_data: FloatArray = field(default_factory=lambda: np.empty((0, 0, 100)))
    grf_data: FloatArray = field(default_factory=lambda: np.empty((0, 0, 100)))

    @property
    def n_subjects(self) -> int:
        return len(self.subjects)


@dataclass(frozen=True)
class ExtractedRollout:
    """What one rollout archive yields: profiles, cycles, and the audit trail."""

    profiles: PolicyProfiles
    cycles: PolicyCycles
    all_gait_lengths: IntArray
    skipped_episodes: list[SkippedEpisode]
    episode_gait_records: list[EpisodeGaitRecord]


@dataclass(slots=True)
class PopulationMetrics:
    """One policy's population profile scored against every human subject.

    Each per-channel dict holds one value per human subject; ``avg_*`` are those
    values averaged over the subjects that have a defined score.
    """

    joint_rmse: PerChannel
    joint_corr: PerChannel
    muscle_corr: PerChannel
    rsa_matrix: FloatArray
    rsa_matrix_pearson: FloatArray
    avg_joint_rmse: dict[str, float] = field(default_factory=dict)
    avg_joint_corr: dict[str, float] = field(default_factory=dict)
    avg_muscle_corr: dict[str, float] = field(default_factory=dict)
    avg_rsa: float = float("nan")
    avg_rsa_pearson: float = float("nan")


@dataclass(slots=True)
class PerSubjectMetrics:
    """One policy's per-subject profiles scored against the matching human.

    Per-channel lists hold one value per subject *with retained cycles*, in the
    order of ``subjects_with_rsa_data``; ``subjects_without_gait_cycles`` names
    the cohort subjects this policy could not score, so an incomplete policy is
    reported rather than silently averaged away.
    """

    joint_rmse: PerChannel
    joint_corr: PerChannel
    muscle_corr: PerChannel
    rsa_matrix: FloatArray
    rsa_matrix_pearson: FloatArray
    rsa: list[float] = field(default_factory=list)
    rsa_pearson: list[float] = field(default_factory=list)
    subjects_with_rsa_data: list[int] = field(default_factory=list)
    subjects_without_gait_cycles: list[int] = field(default_factory=list)
    avg_rsa: float = float("nan")
    avg_rsa_pearson: float = float("nan")


@dataclass(slots=True)
class SubjectReference:
    """A per-subject reference band, matched to subjects by id.

    ``subjects`` gives the subject behind each entry of the metric lists, in the
    same order, so paired tests can match this reference to a policy by id
    rather than by position.
    """

    n: int
    subjects: list[int]
    joint_rmse: PerChannel
    muscle_corr: PerChannel
    rsa_spearman: list[float]
    rsa_pearson: list[float]
    joint_corr: PerChannel | None = None
    label: str = ""


@dataclass(slots=True)
class NoiseCeiling:
    """Leave-one-subject-out RSA ceiling over the human cohort."""

    n: int
    rsa_spearman: list[float]
    rsa_pearson: list[float]


@dataclass(slots=True)
class HumanReferences:
    """The reference bands a policy is read against, for one gait selection.

    Each is ``None`` when the inputs it needs are absent rather than being
    replaced by a different reference.
    """

    within_subject: SubjectReference | None = None
    rsa_noise_ceiling: NoiseCeiling | None = None
    policy_within_subject: list[SubjectReference] | None = None


@dataclass(slots=True)
class HumanToHumanMetrics:
    """Pairwise human-to-human similarity: the between-subject reference."""

    pairwise_joint_rmse: PerChannel
    pairwise_joint_corr: PerChannel
    pairwise_muscle_corr: PerChannel
    rsa_matrix: FloatArray
    rsa_matrix_pearson: FloatArray
    avg_pairwise_joint_rmse: dict[str, float] = field(default_factory=dict)
    avg_pairwise_joint_corr: dict[str, float] = field(default_factory=dict)
    avg_pairwise_muscle_corr: dict[str, float] = field(default_factory=dict)
    avg_pairwise_muscle_rsa: float = float("nan")
    avg_pairwise_muscle_rsa_pearson: float = float("nan")


@dataclass(slots=True)
class KineticsMetrics:
    """Joint-moment and vertical-GRF scores indexed by subject."""

    subjects: list[int]
    torque_rmse: PerChannel
    torque_corr: PerChannel
    grf_rmse: PerChannel
    grf_corr: PerChannel
    avg_torque_rmse: dict[str, float] = field(default_factory=dict)
    avg_torque_corr: dict[str, float] = field(default_factory=dict)
    avg_grf_rmse: dict[str, float] = field(default_factory=dict)
    avg_grf_corr: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class KineticsHumanToHumanMetrics:
    """Pairwise reference variability for human moments and vertical GRF."""

    subject_pairs: list[tuple[int, int]]
    torque_rmse: PerChannel
    torque_corr: PerChannel
    grf_rmse: PerChannel
    grf_corr: PerChannel
    avg_torque_rmse: dict[str, float] = field(default_factory=dict)
    avg_torque_corr: dict[str, float] = field(default_factory=dict)
    avg_grf_rmse: dict[str, float] = field(default_factory=dict)
    avg_grf_corr: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class WilcoxonResults:
    """The paired-test tables written for one gait selection."""

    comparisons: pd.DataFrame
    per_channel: pd.DataFrame


@dataclass(slots=True)
class PolicyRun:
    """One policy: its rollout cycles, current-gait profiles, and metrics.

    ``cycles`` and ``earlier_trials`` are the loaded rollouts and never change.
    ``profiles``, ``earlier_trial_profiles`` and every metric field are rebuilt
    for each gait selection, so a policy that has no cycles at the selected gait
    carries ``None`` there instead of a stale result from another selection.
    """

    policy: PolicySpec
    cycles: PolicyCycles
    profiles: PolicyProfiles
    all_gait_lengths: IntArray
    skipped_episodes: list[SkippedEpisode]
    episode_gait_records: list[EpisodeGaitRecord]
    earlier_trials: PolicyCycles | None = None
    earlier_trial_profiles: dict[int, PolicyProfiles] | None = None
    population: PopulationMetrics | None = None
    per_subject: PerSubjectMetrics | None = None
    earlier_human_sensitivity: SubjectReference | None = None
    within_subject_variability: SubjectReference | None = None
    kinetics_cycles: KineticsCycles | None = None
    kinetics_profiles: KineticsProfiles | None = None
    kinetics_population: KineticsMetrics | None = None
    kinetics_per_subject: KineticsMetrics | None = None
    kinetics_omissions: tuple[KineticsOmission, ...] = ()

    @property
    def label(self) -> str:
        return self.policy.label


@dataclass(slots=True)
class GaitAnalysis:
    """Everything one gait analysis run reads, writes, and reports.

    Built by :func:`mm_evalkit.workflows.gait.loading.load_gait_analysis`, which
    validates and loads; constructing this object performs no I/O of its own.
    """

    config: FullBodyConfig
    channels: GaitChannels
    style: PlotStyle
    human: HumanCohort
    policies: list[PolicyRun]
    plots_dir: Path
    motion_list: Sequence[str] | None
    muscle_map: dict[str, str]
    kinetics: KineticsCohort | None = None

    gait_position: int | None = None
    cohort_subjects: tuple[int, ...] = ()
    human_metrics: HumanToHumanMetrics | None = None
    references: HumanReferences = field(default_factory=HumanReferences)
    wilcoxon: WilcoxonResults | None = None
    gait_count_table: pd.DataFrame | None = None
    kinetics_human_metrics: KineticsHumanToHumanMetrics | None = None

    @property
    def dataset_name(self) -> str:
        return self.config.dataset_name

    @property
    def subject_based(self) -> bool:
        return self.config.subject_based

    @property
    def kinematics_source(self) -> str:
        return self.config.kinematics_source

    def summary(self) -> dict[str, Any]:
        """Compact provenance of the current selection, for run manifests."""
        return {
            "dataset": self.dataset_name,
            "kinematics_source": self.kinematics_source,
            "gait_position": self.gait_position,
            "subjects": list(self.human.subjects),
            "policies": [run.policy.id for run in self.policies],
            "kinetics": self.kinetics is not None,
        }
