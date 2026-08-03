"""Wiring of the ULTra-MoCap paired statistics onto the shared Wilcoxon tests."""

import csv

import matplotlib
import numpy as np

matplotlib.use("Agg")

from mm_evalkit.stats.wilcoxon import MetricSamples
from mm_evalkit.workflows.bimanual.statistics import (
    _aggregate_matched_trials_by_subject,
    run_wilcoxon_tests,
)

from .conftest import Policy, bimanual_records


def test_bimanual_run_wilcoxon_tests(tmp_path):
    rng = np.random.default_rng(4)
    joints = ["elv_angle_r", "shoulder_elv_r", "elbow_flex_r"]
    electrodes = ["Biceps Brachii (IM EMG4)", "Triceps Brachii (IM EMG5)", "Deltoid (IM EMG6)"]
    keys = [
        ("ArmSwing", condition, subject)
        for condition in ("Slow", "Normal", "Fast")
        for subject in ("P11", "P12", "P13")
    ]

    policies = [Policy("zero-shot"), Policy("finetuned")]
    policy_records = {
        "zero-shot": bimanual_records(
            rng,
            2.0,
            keys,
            len(joints),
            len(electrodes),
            with_reference=True,
        ),
        "finetuned": bimanual_records(
            rng,
            0.0,
            keys,
            len(joints),
            len(electrodes),
            with_reference=True,
        ),
    }
    real_records = bimanual_records(rng, 0.0, keys, len(joints), len(electrodes), with_reference=True)

    table = run_wilcoxon_tests(policy_records, policies, real_records, joints, electrodes, str(tmp_path))

    assert set(table["axis"]) == {"across subjects"}
    assert set(table["metric"]) == {
        "Joint tracking error [deg]",
        "Muscle-EMG correlation",
        "Muscle RSA (Spearman)",
    }
    assert set(table["comparison_group"]) == {
        "Policy vs policy",
        "Policy vs matched split-half ceiling (real)",
        "Policy vs matched rep-to-rep variability (real)",
    }
    assert len(table) == 15
    assert (table[table["condition_b"] == "P2"]["n"] == 3).all()
    assert {
        "P1 split-half real",
        "P1 rep-to-rep real",
        "P2 split-half real",
        "P2 rep-to-rep real",
    }.issubset(set(table["condition_b"]))
    with (tmp_path / "wilcoxon_per_channel_tests.csv").open(
        newline="",
        encoding="utf-8",
    ) as stream:
        channel_table = list(csv.DictReader(stream))
    assert len(channel_table) == 30
    # RSA has no per-electrode breakdown, so it reports no breadth.
    rsa = table[table["metric"] == "Muscle RSA (Spearman)"]
    assert (rsa["n_channels"] == 0).all()
    # The metrics that do have one are broken down per electrode / joint.
    muscle = table[table["metric"] == "Muscle-EMG correlation"]
    assert (muscle["n_channels"] == len(electrodes)).all()
    assert (tmp_path / "wilcoxon_tests.csv").exists()
    assert (tmp_path / "wilcoxon_per_channel_tests.csv").exists()
    assert (tmp_path / "wilcoxon_joint_tracking_error_deg.png").exists()
    assert (tmp_path / "wilcoxon_muscle_rsa_spearman.png").exists()
    assert (tmp_path / "wilcoxon_per_channel_muscle_emg_correlation.png").exists()


def test_bimanual_subject_aggregation_uses_identical_trial_sets():
    units = (
        ("ArmSwing", "Slow", "P11"),
        ("ArmSwing", "Fast", "P11"),
        ("ArmSwing", "Slow", "P12"),
    )
    first = MetricSamples(
        np.array([[1.0], [100.0], [3.0]]),
        units,
        ("joint",),
    )
    second = MetricSamples(
        np.array([[2.0], [np.nan], [4.0]]),
        units,
        ("joint",),
    )

    matched_first, matched_second = _aggregate_matched_trials_by_subject(
        first,
        second,
    )

    assert matched_first.units == ("P11", "P12")
    np.testing.assert_allclose(matched_first.values[:, 0], [1.0, 3.0])
    np.testing.assert_allclose(matched_second.values[:, 0], [2.0, 4.0])


def test_bimanual_needs_segmented_trials(tmp_path):
    assert run_wilcoxon_tests({}, [], {}, ["j"], ["m"], str(tmp_path)) is None
