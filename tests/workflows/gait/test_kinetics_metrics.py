"""Numerical and reporting tests for the Gait120 kinetics layer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mm_evalkit.datasets.gait120_kinetics import GRF_CHANNELS, TORQUE_JOINTS
from mm_evalkit.workflows.gait.kinetics_data import select_kinetics_subset
from mm_evalkit.workflows.gait.kinetics_metrics import compute_kinetics_metrics, write_kinetics_metrics
from mm_evalkit.workflows.gait.state import KineticsCohort, KineticsCycles

from .conftest import make_analysis, make_cohort, make_run

SUBJECTS = [51, 52, 53]


def _cycles(offset: float) -> KineticsCycles:
    phase = np.linspace(-1.0, 1.0, 100)
    torque_ref = np.stack(
        [np.stack([phase + 2 * channel + subject for channel in range(6)]) for subject in range(len(SUBJECTS))]
    )
    grf_ref = np.stack(
        [np.stack([phase + channel + subject for channel in range(2)]) for subject in range(len(SUBJECTS))]
    )
    return KineticsCycles(
        torque=torque_ref + offset,
        torque_ref=torque_ref,
        grf=grf_ref + offset,
        grf_ref=grf_ref,
        subject_ids=np.asarray(SUBJECTS),
        trial_ids=np.full(len(SUBJECTS), 5),
        gait_positions=np.zeros(len(SUBJECTS), dtype=int),
    )


def _analysis(tmp_path: Path):
    human_cycles = _cycles(0.0)
    cohort = KineticsCohort(
        torque_names=TORQUE_JOINTS,
        grf_names=GRF_CHANNELS,
        torque_unit="Nm/kg",
        grf_unit="BW",
        cycles=human_cycles,
        reference_paths=(),
    )
    runs = [make_run("first"), make_run("second")]
    runs[0].kinetics_cycles = _cycles(1.0)
    runs[1].kinetics_cycles = _cycles(2.0)
    analysis = make_analysis(
        policies=runs,
        cohort=make_cohort(SUBJECTS),
        plots_dir=tmp_path,
    )
    analysis.kinetics = cohort
    select_kinetics_subset(runs, cohort, SUBJECTS, 0)
    return analysis


def test_subject_matched_metrics_preserve_subject_as_the_statistical_unit(tmp_path: Path) -> None:
    analysis = _analysis(tmp_path)

    compute_kinetics_metrics(analysis)

    human = analysis.kinetics_human_metrics
    assert human.subject_pairs == [(51, 52), (51, 53), (52, 53)]
    np.testing.assert_allclose(human.torque_rmse["all_moments"], [1.0, 2.0, 1.0])
    first = analysis.policies[0].kinetics_per_subject
    assert first.subjects == SUBJECTS
    np.testing.assert_allclose(first.torque_rmse["all_moments"], 1.0)
    np.testing.assert_allclose(first.grf_rmse["all_grf"], 1.0)
    np.testing.assert_allclose(first.torque_corr["all_moments"], 1.0)
    np.testing.assert_allclose(first.grf_corr["all_grf"], 1.0)


def test_population_profile_weights_subjects_equally(tmp_path: Path) -> None:
    analysis = _analysis(tmp_path)

    compute_kinetics_metrics(analysis)

    first = analysis.policies[0].kinetics_population
    # The population profile is mean(reference subjects) + 1. Its three
    # subject comparisons therefore have errors 2, 1, and 0.
    np.testing.assert_allclose(first.torque_rmse["all_moments"], [2.0, 1.0, 0.0], atol=1e-12)
    assert first.avg_torque_rmse["all_moments"] == pytest.approx(1.0)


def test_tidy_csv_contains_every_scope_signal_metric_and_unit(tmp_path: Path) -> None:
    analysis = _analysis(tmp_path)
    compute_kinetics_metrics(analysis)

    path = write_kinetics_metrics(analysis)

    assert path == tmp_path / "kinetics_metrics.csv"
    text = path.read_text(encoding="utf-8")
    assert "human_human,joint_moment,rmse,hip_flexion_r,subject_pair,51-52" in text
    assert "first,first,subject_matched,vertical_grf,pearson,left,subject,51" in text
