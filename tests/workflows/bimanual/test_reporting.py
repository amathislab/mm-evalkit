"""Figures and summary tables built from the ULTRA-MoCap trial records."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from mm_evalkit.workflows.bimanual.metrics import pairwise_rsa
from mm_evalkit.workflows.bimanual.rsm_plots import plot_rsm_matrices
from mm_evalkit.workflows.bimanual.tables import (
    _condition_scores,
    _format_cell,
    _mean_std,
    _trial_scores,
    plot_metrics_table,
)

from .conftest import Policy, bimanual_records


def test_pairwise_rsa_scores_every_repetition_pair():
    rng = np.random.default_rng(7)
    cycles = rng.normal(0.0, 1.0, (4, 3, 60))

    scores = pairwise_rsa(cycles)

    assert len(scores) == 6  # 4 repetitions -> 6 pairs
    assert np.isfinite(scores).all()
    assert pairwise_rsa(cycles[:1]).size == 0  # nothing to pair


def test_pairwise_rsa_omits_undefined_repetition_pairs():
    rng = np.random.default_rng(17)
    valid = rng.normal(0.0, 1.0, (2, 3, 60))
    silent = np.zeros((1, 3, 60))

    scores = pairwise_rsa(np.concatenate((valid, silent), axis=0))

    assert scores.shape == (1,)
    assert np.isfinite(scores).all()


def test_bimanual_rsm_figures(tmp_path):
    rng = np.random.default_rng(8)
    keys = [("ArmSwing", "Slow", subject) for subject in ("P11", "P12")]
    policies = [Policy("zero-shot")]
    policy_records = {"zero-shot": bimanual_records(rng, 0.0, keys, 3, 3)}
    real_records = bimanual_records(rng, 0.0, keys, 3, 3, with_reference=True)

    plot_rsm_matrices(policy_records, policies, real_records, ["Biceps", "Triceps", "Deltoid"], str(tmp_path))

    assert (tmp_path / "ArmSwing_Slow" / "rsms.png").exists()


def test_trial_and_condition_aggregation_are_different_samples():
    """'Slow' was recorded by two subjects and 'Fast' by one, so the two levels differ."""
    records = {
        ("ArmSwing", "Slow", "P11"): {"joint_rmse_by": np.array([[2.0, 2.0]])},
        ("ArmSwing", "Slow", "P12"): {"joint_rmse_by": np.array([[4.0, 4.0]])},
        ("ArmSwing", "Fast", "P11"): {"joint_rmse_by": np.array([[10.0, 10.0]])},
    }
    groups = [("ArmSwing", "Slow"), ("ArmSwing", "Fast")]

    # One sample per trial...
    assert sorted(_trial_scores(records, "joint_rmse_by")) == [2.0, 4.0, 10.0]
    assert _mean_std(_trial_scores(records, "joint_rmse_by"))[0] == pytest.approx(16.0 / 3.0)
    # ...versus one sample per condition, each first averaged over its subjects, so the
    # two-subject condition does not outweigh the one-subject one.
    assert _condition_scores(records, groups, "joint_rmse_by", per_trial=True) == [3.0, 10.0]
    assert _mean_std(_condition_scores(records, groups, "joint_rmse_by", per_trial=True))[0] == pytest.approx(6.5)


def test_trial_scores_skip_trials_without_the_metric():
    records = {("m", "c", "P11"): {"joint_rmse_by": np.array([[1.0]])}, ("m", "c", "P12"): {}}
    assert _trial_scores(records, "joint_rmse_by") == [1.0]


def test_format_cell_rounds_to_the_precision_the_spread_supports():
    assert _format_cell((7.5432, 3.3)) == "7.5 ± 3.3"
    assert _format_cell((0.9612, 0.0351)) == "0.96 ± 0.04"
    assert _format_cell((np.nan, np.nan)) == "—"
    # A value that rounds to zero must not come out as "-0.0".
    assert _format_cell((-0.004, 0.2)) == "0.0 ± 0.2"


def test_bimanual_metrics_table(tmp_path):
    rng = np.random.default_rng(5)
    joints, electrodes = 3, 3
    keys = [("ArmSwing", condition, subject) for condition in ("Slow", "Normal") for subject in ("P11", "P12")]
    policies = [Policy("zero-shot"), Policy("finetuned")]
    policy_records = {
        "zero-shot": bimanual_records(rng, 4.0, keys, joints, electrodes),
        "finetuned": bimanual_records(rng, 0.0, keys, joints, electrodes),
    }
    real_records = bimanual_records(rng, 0.0, keys, joints, electrodes, with_reference=True)

    plot_metrics_table(policy_records, policies, real_records, str(tmp_path / "metrics_summary_table"))

    assert (tmp_path / "metrics_summary_table.png").exists()
    assert (tmp_path / "metrics_summary_table.svg").exists()


def test_bimanual_metrics_table_needs_trials(tmp_path, capsys):
    plot_metrics_table({}, [], {}, str(tmp_path / "metrics_summary_table"))
    assert "skipping the metrics table" in capsys.readouterr().out
    assert not (tmp_path / "metrics_summary_table.png").exists()
