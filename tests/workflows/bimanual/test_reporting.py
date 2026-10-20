"""Figures and summary tables built from the ULTRA-MoCap trial records."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from mm_evalkit.workflows.bimanual.aggregate_waveforms import _stack_movement_cycles
from mm_evalkit.workflows.bimanual.grids import _subjects_by_group, plot_score_grid, plot_waveform_grid
from mm_evalkit.workflows.bimanual.metrics import pairwise_rsa
from mm_evalkit.workflows.bimanual.references import compute_between_subject_reference
from mm_evalkit.workflows.bimanual.rsm_plots import plot_rsm_matrices
from mm_evalkit.workflows.bimanual.summary_plots import _summary_axes, plot_metrics_summary_bars
from mm_evalkit.workflows.bimanual.tables import (
    _format_cell,
    _group_scores,
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


def test_trial_and_movement_aggregation_are_different_samples():
    """P11 swung at two speeds and P12 at one, so the two aggregation levels differ."""
    records = {
        ("ArmSwing", "Slow", "P11"): {"joint_rmse_by": np.array([[2.0, 2.0]])},
        ("ArmSwing", "Slow", "P12"): {"joint_rmse_by": np.array([[4.0, 4.0]])},
        ("ArmSwing", "Fast", "P11"): {"joint_rmse_by": np.array([[10.0, 10.0]])},
    }
    groups = ["ArmSwing"]

    # One sample per trial...
    assert sorted(_trial_scores(records, "joint_rmse_by")) == [2.0, 4.0, 10.0]
    assert _mean_std(_trial_scores(records, "joint_rmse_by"))[0] == pytest.approx(16.0 / 3.0)
    # ...versus one sample per movement: P11's two speeds average to 6 and P12's single
    # speed stays 4, then the subjects average, so P11's extra speed does not outweigh P12.
    assert _group_scores(records, groups, "joint_rmse_by", per_trial=True) == [5.0]


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


def _grid_inputs(seed=11):
    """Two policies over two conditions, with waveforms, for the cross-policy figures."""
    rng = np.random.default_rng(seed)
    joints, electrodes = 3, 3
    keys = [("ArmSwing", condition, subject) for condition in ("Slow", "Fast") for subject in ("P11", "P12")]
    # Two policies of one family: same color, told apart by linestyle alone.
    policies = [Policy("from-scratch", color="green"), Policy("finetuned", color="green", linestyle=":")]
    policy_records = {
        "from-scratch": bimanual_records(rng, 4.0, keys, joints, electrodes, with_cycles=True),
        "finetuned": bimanual_records(rng, 0.0, keys, joints, electrodes, with_cycles=True),
    }
    real_records = bimanual_records(rng, 0.0, keys, joints, electrodes, with_reference=True, with_cycles=True)
    return policy_records, policies, real_records


def test_bimanual_metrics_summary_bars(tmp_path):
    policy_records, policies, real_records = _grid_inputs()

    plot_metrics_summary_bars(policy_records, policies, real_records, str(tmp_path / "metrics_summary_bars"))

    assert (tmp_path / "metrics_summary_bars.png").exists()
    assert (tmp_path / "metrics_summary_bars.svg").exists()


def test_bimanual_metrics_summary_bars_need_trials(tmp_path, capsys):
    plot_metrics_summary_bars({}, [], {}, str(tmp_path / "metrics_summary_bars"))
    assert "skipping the metrics summary bars" in capsys.readouterr().out
    assert not (tmp_path / "metrics_summary_bars.png").exists()


def test_summary_groups_are_movements_not_movement_conditions():
    """A movement is one group however many speeds it was recorded at."""
    _, _, real_records = _grid_inputs()  # ArmSwing x (Slow, Fast) x (P11, P12)

    subjects, groups = _summary_axes(real_records)

    assert groups == ["ArmSwing"]
    assert subjects == ["P11", "P12"]
    # ...and a subject occupies one slot in that group, not one per speed.
    assert _subjects_by_group(real_records) == {"ArmSwing": ["P11", "P12"]}


def test_movement_cycles_average_a_subjects_speeds_before_stacking():
    """The grid band stays the between-subject spread, not the between-speed spread."""
    records = {
        ("ArmSwing", "Slow", "P11"): {"ref_cycle": np.zeros((1, 4))},
        ("ArmSwing", "Fast", "P11"): {"ref_cycle": np.full((1, 4), 4.0)},
        ("ArmSwing", "Slow", "P12"): {"ref_cycle": np.full((1, 4), 4.0)},
    }

    mean, std = _stack_movement_cycles(records, "ArmSwing", ["P11", "P12"], "ref_cycle")

    # P11's two speeds average to 2 and P12 stays at 4, so the mean is 3 and the band is
    # the spread between those two subjects -- not the spread over the three trials.
    assert mean == pytest.approx(np.full((1, 4), 3.0))
    assert std == pytest.approx(np.full((1, 4), 1.0))


def test_between_subject_reference_pairs_subjects_within_one_speed():
    """Pairing across speeds would measure the speed difference, not the subject one."""
    cycles = {
        ("ArmSwing", "Slow", "P11"): 0.0,
        ("ArmSwing", "Slow", "P12"): 1.0,
        ("ArmSwing", "Fast", "P11"): 10.0,
        ("ArmSwing", "Fast", "P12"): 11.0,
    }
    records = {key: {"ref_cycle": np.full((1, 4), value)} for key, value in cycles.items()}

    reference = compute_between_subject_reference(records)

    # One pair per speed pooled under the movement -- two rows, not the six that pairing
    # all four cycles together would give.
    assert list(reference) == ["ArmSwing"]
    assert reference["ArmSwing"]["joint_rmse_by"].shape == (2, 1)
    # Both pairs are one subject apart; the 10-unit speed gap never enters the score.
    assert reference["ArmSwing"]["joint_rmse"] == pytest.approx([1.0, 1.0])


def test_waveform_grid_has_one_row_per_movement_and_column_per_channel(tmp_path):
    policy_records, policies, real_records = _grid_inputs()

    plot_waveform_grid(
        policy_records,
        policies,
        real_records,
        ["elv_angle_r", "shoulder_elv_r", "elbow_flex_r"],
        policy_field="pol_cycle",
        real_field="ref_cycle",
        ylabel="joint angle (deg)",
        real_label="Real (retargeted)",
        suptitle="joints",
        out_path=str(tmp_path / "grid_joint_waveforms"),
    )

    assert (tmp_path / "grid_joint_waveforms.png").exists()
    assert (tmp_path / "grid_joint_waveforms.svg").exists()
    # The figure is closed after saving; a report writes dozens of these.
    assert plt.get_fignums() == []


def test_score_grid_shares_one_scale_per_movement(tmp_path):
    policy_records, policies, real_records = _grid_inputs()
    channels = ["elv_angle_r", "shoulder_elv_r", "elbow_flex_r"]

    plot_score_grid(
        policy_records,
        policies,
        real_records,
        channels,
        key="joint_rmse_by",
        ylabel="RMSE (deg)",
        suptitle="joints",
        out_path=str(tmp_path / "grid_joint_scores"),
    )

    assert (tmp_path / "grid_joint_scores.png").exists()
    assert (tmp_path / "grid_joint_scores.svg").exists()


def test_grid_figures_skip_an_empty_analysis(tmp_path):
    plot_score_grid(
        {}, [], {}, ["elv_angle_r"], key="joint_rmse_by", ylabel="", suptitle="", out_path=str(tmp_path / "g")
    )
    plot_waveform_grid(
        {},
        [],
        {},
        ["elv_angle_r"],
        policy_field="pol_cycle",
        real_field="ref_cycle",
        ylabel="",
        real_label="Real",
        suptitle="",
        out_path=str(tmp_path / "w"),
    )
    assert not list(tmp_path.iterdir())
