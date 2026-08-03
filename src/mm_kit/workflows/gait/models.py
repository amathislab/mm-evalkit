"""Rollout-side records produced and consumed by the gait workflow stages."""

from __future__ import annotations

from typing import Literal, TypedDict

import numpy as np
from numpy.typing import NDArray

SkipReason = Literal["early_termination", "no_valid_gait_cycles"]


class NoSubjectsForGaitError(RuntimeError):
    """The requested gait ordinal is absent from every loaded human subject."""


class SkippedEpisode(TypedDict):
    """An episode excluded for a scientifically defined rollout outcome."""

    episode: int
    traj_id: int
    subject_id: int | None
    reason: SkipReason
    episode_length: int
    reference_length: int


class EpisodeGaitRecord(TypedDict):
    """Audit record for every episode, including episodes with zero cycles."""

    episode: int
    traj_id: int
    subject_id: int | None
    n_gaits: int
    gait_indexes: list[tuple[int, int]]
    reference_length: int
    episode_length: int
    reason: Literal["ok", "early_termination", "no_valid_gait_cycles"]
    event_seq: str
    phase_seq: str


class ExtractedTrajectory(TypedDict):
    """Normalized gait cycles and their subject/episode provenance."""

    all_gait_lengths: NDArray[np.integer]
    avg_joint_gaits: NDArray[np.floating]
    std_joint_gaits: NDArray[np.floating]
    avg_joint_gaits_ref: NDArray[np.floating]
    std_joint_gaits_ref: NDArray[np.floating]
    avg_muscle_gaits: NDArray[np.floating]
    std_muscle_gaits: NDArray[np.floating]
    population_muscle_rsm: NDArray[np.floating]
    all_subject_ids: NDArray[np.integer] | None
    per_subject_avg_muscle_gaits: dict[int, NDArray[np.floating]] | None
    per_subject_avg_joint_gaits: dict[int, NDArray[np.floating]] | None
    per_subject_avg_joint_gaits_ref: dict[int, NDArray[np.floating]] | None
    per_subject_avg_muscle_rms: dict[int, NDArray[np.floating]] | None
    skipped_episodes: list[SkippedEpisode]
    episode_gait_records: list[EpisodeGaitRecord]
    all_joint_gaits_raw: NDArray[np.floating] | None
    all_joint_gaits_ref_raw: NDArray[np.floating] | None
    all_muscle_gaits_raw: NDArray[np.floating] | None
    all_subject_ids_raw: NDArray[np.integer] | None
    all_trial_ids_raw: NDArray[np.integer] | None
    all_gait_positions_raw: NDArray[np.integer] | None
