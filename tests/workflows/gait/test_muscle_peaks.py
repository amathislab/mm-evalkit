"""Peak metrics preserve amplitudes and respect the gait-cycle boundary."""

import numpy as np
import pytest

from mm_evalkit.workflows.gait.metrics import muscle_peak, muscle_peak_errors


def test_peak_height_and_percentage():
    assert muscle_peak([0, 2, 4, 1, 0]) == (4, 50)


def test_peak_time_wraps_at_cycle_boundary():
    first = np.zeros(101)
    second = np.zeros(101)
    first[98] = 3
    second[2] = 5
    assert muscle_peak_errors(first, second) == pytest.approx((2, 4))
    first[:] = 0
    second[:] = 0
    first[0] = second[-1] = 1
    assert muscle_peak_errors(first, second) == (0, 0)


def test_flat_and_missing_signals_do_not_invent_peak_times():
    assert muscle_peak([2, 2, 2])[0] == 2
    assert np.isnan(muscle_peak([2, 2, 2])[1])
    assert all(np.isnan(muscle_peak([0, np.nan, 2])))
    assert muscle_peak([0, 3, 3, 0]) == (3, pytest.approx(100 / 3))


def test_peak_metrics_and_both_plot_layouts(tmp_path, monkeypatch):
    from matplotlib.figure import Figure

    from mm_evalkit.workflows.gait import metrics
    from mm_evalkit.workflows.gait.plotting.muscles import (
        plot_muscle_metrics,
        plot_subject_based_muscle_metrics,
    )
    from tests.workflows.gait.test_pipeline import _analysis

    analysis = _analysis(tmp_path)
    analysis.human.muscle_data *= 10
    metrics.compute_human_to_human_metrics(analysis)
    metrics.compute_human_to_model_metrics_population(analysis)
    metrics.compute_human_to_model_metrics_per_subject(analysis)
    figures = []
    monkeypatch.setattr(Figure, "savefig", lambda fig, *args, **kwargs: figures.append(fig))
    plot_muscle_metrics(analysis)
    plot_subject_based_muscle_metrics(analysis)
    for figure_idx, fig in enumerate(figures):
        titles = [ax.get_title() for ax in fig.axes]
        assert [ax.get_title() for ax in fig.axes[:3]] == [
            "All muscles: correlation",
            "All muscles:\npeak height error",
            "All muscles:\npeak time error",
        ]
        correlation_ax = next(ax for ax in fig.axes if ax.get_title() == "Correlation")
        assert correlation_ax.get_ylabel() == analysis.muscle_map[analysis.channels.muscle_names[0]]
        for ax in fig.axes:
            if ax.get_title() in ("Correlation", "Peak height error", "Peak time error") or ax.get_title().startswith("All muscles:"):
                assert ax.title.get_fontweight() == "bold"
        assert "Peak height error" in titles
        assert "Peak time error" in titles
        normalized_ax = next(ax for ax in fig.axes if ax.get_title() == "Normalized profile")
        expected = analysis.human.muscle_data[0, 0]
        np.testing.assert_allclose(normalized_ax.lines[0].get_ydata(),
                                   (expected - expected.min()) / np.ptp(expected))
        subject_pos = analysis.human.plot_subject_positions[0]
        raw_title = next(title for title in titles if title.endswith(" profile") and not title.startswith("Normalized")) if figure_idx == 0 else f"Subj. {analysis.human.subjects[subject_pos]}"
        raw_ax = next(ax for ax in fig.axes if ax.get_title() == raw_title)
        raw_expected = analysis.human.muscle_data[0 if figure_idx == 0 else subject_pos, 0]
        np.testing.assert_array_equal(raw_ax.lines[0].get_ydata(), raw_expected)
        assert raw_ax.get_ylim()[1] > 1
    for run in analysis.policies:
        for record in (run.population, run.per_subject):
            assert len(record.muscle_peak_time_error["all_muscles"]) == analysis.human.n_subjects
            assert all(0 <= x <= 50 for x in record.muscle_peak_time_error["all_muscles"])
