"""Focused numerical-contract tests for the full-body analysis."""

from dataclasses import replace

import numpy as np
import pytest

from mm_kit.workflows.gait import metrics
from mm_kit.workflows.gait.numerics import (
    finite_mean,
    paired_pearson,
)

from .conftest import make_analysis, make_channels, make_cohort, make_run


def test_paired_pearson_uses_only_finite_pairs():
    first = np.array([0.0, 1.0, np.nan, 2.0, np.inf])
    second = np.array([0.0, 2.0, 50.0, 4.0, 100.0])

    assert paired_pearson(first, second) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ([1.0, np.nan], [2.0, 3.0]),
        ([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]),
        ([1.0, 2.0, 3.0], [4.0, 4.0, 4.0]),
    ],
)
def test_paired_pearson_returns_nan_when_undefined(first, second):
    assert np.isnan(paired_pearson(first, second))


def test_finite_mean_excludes_nonfinite_values():
    values = np.array([[1.0, np.nan, np.inf], [3.0, np.nan, -np.inf]])

    assert finite_mean(values) == pytest.approx(2.0)
    np.testing.assert_allclose(
        finite_mean(values, axis=0),
        [2.0, np.nan, np.nan],
        equal_nan=True,
    )


def test_population_average_ignores_undefined_channels(monkeypatch):
    """A channel whose correlation is undefined drops out of the across-channels mean."""
    monkeypatch.setattr(metrics, "rsa", lambda *_args: (np.nan, np.nan))
    signal = np.arange(4, dtype=float)
    constant = np.ones_like(signal)

    run = make_run("test policy")
    run.profiles = replace(
        run.profiles,
        avg_joint_gaits=np.stack([constant, signal]),
        avg_muscle_gaits=np.stack([constant, signal]),
        population_muscle_rsm=np.full((2, 2), np.nan),
    )
    cohort = make_cohort([11], joints=["constant_joint", "signal_joint"], muscles=["constant_muscle", "signal_muscle"])
    cohort.joint_data = np.stack([np.stack([constant, signal])])
    cohort.muscle_data = np.stack([np.stack([constant, 2.0 * signal])])
    cohort.muscle_rsms = {0: np.full((2, 2), np.nan)}

    analysis = make_analysis(
        policies=[run],
        cohort=cohort,
        channels=make_channels(["constant_joint", "signal_joint"], ["constant_muscle", "signal_muscle"]),
    )

    metrics.compute_human_to_model_metrics_population(analysis)

    assert np.isnan(run.population.joint_corr["constant_joint"][0])
    assert run.population.joint_corr["all_joints"] == pytest.approx([1.0])
    assert np.isnan(run.population.muscle_corr["constant_muscle"][0])
    assert run.population.muscle_corr["all_muscles"] == pytest.approx([1.0])
