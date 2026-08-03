"""Wiring of the gait metrics onto the shared Wilcoxon tests and summary table."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from mm_evalkit.workflows.gait.constants import (
    BETWEEN_SUBJECT_LABEL,
    EARLIER_HUMAN_SENSITIVITY_LABEL,
    POLICY_WITHIN_SUBJECT_LABEL,
    RSA_NOISE_CEILING_LABEL,
    WITHIN_SUBJECT_LABEL,
)
from mm_evalkit.workflows.gait.plotting.summary import (
    REFERENCE_SCOPES,
    _scoped_reference_metrics,
)
from mm_evalkit.workflows.gait.state import (
    PerSubjectMetrics,
    PopulationMetrics,
    SubjectReference,
)
from mm_evalkit.workflows.gait.statistics import compute_wilcoxon_tests

from .conftest import JOINTS, MUSCLES, SUBJECTS, make_analysis, make_run


def test_summary_references_occupy_only_their_defined_scope():
    assert REFERENCE_SCOPES == {
        WITHIN_SUBJECT_LABEL: "subject_matched",
        BETWEEN_SUBJECT_LABEL: "population",
        RSA_NOISE_CEILING_LABEL: "subject_matched",
        POLICY_WITHIN_SUBJECT_LABEL: "subject_matched",
        EARLIER_HUMAN_SENSITIVITY_LABEL: "subject_matched",
    }

    empty = (np.nan, [])
    metrics = [(1.0, [1.0]), (2.0, [2.0]), (3.0, [3.0])]
    population = _scoped_reference_metrics("population", metrics, empty_metric=empty)
    subject_matched = _scoped_reference_metrics("subject_matched", metrics, empty_metric=empty)

    assert [np.isfinite(metric[0]) for metric in population] == [True, False, True, False, True, False]
    assert [np.isfinite(metric[0]) for metric in subject_matched] == [False, True, False, True, False, True]

    noise_ceiling = _scoped_reference_metrics(
        REFERENCE_SCOPES[RSA_NOISE_CEILING_LABEL],
        [empty, empty, metrics[2]],
        empty_metric=empty,
    )
    assert [np.isfinite(metric[0]) for metric in noise_ceiling] == [False, False, False, False, False, True]

    with pytest.raises(ValueError, match="Unknown summary-table reference scope"):
        _scoped_reference_metrics("both", metrics, empty_metric=empty)


def _per_channel(rng, offset, channels, aggregate):
    return {name: list(rng.normal(offset, 1.0, len(SUBJECTS))) for name in channels} | {aggregate: []}


def _analysis_with_metrics(tmp_path):
    """Two policies whose joint error differs by a constant every subject sees."""
    rng = np.random.default_rng(3)
    analysis = make_analysis(plots_dir=tmp_path)
    analysis.gait_position = 0
    analysis.references.within_subject = SubjectReference(
        n=len(SUBJECTS),
        subjects=list(SUBJECTS),
        joint_rmse=_per_channel(rng, 4.0, JOINTS, "all_joints"),
        joint_corr=_per_channel(rng, 0.9, JOINTS, "all_joints"),
        muscle_corr=_per_channel(rng, 0.7, MUSCLES, "all_muscles"),
        rsa_spearman=list(rng.normal(0.8, 0.05, len(SUBJECTS))),
        rsa_pearson=list(rng.normal(0.8, 0.05, len(SUBJECTS))),
    )

    for index, offset in enumerate((0.0, 3.0)):
        run = make_run(f"policy {index}\nsecond line")
        run.population = PopulationMetrics(
            joint_rmse=_per_channel(rng, 6.0 + offset, JOINTS, "all_joints"),
            joint_corr=_per_channel(rng, 0.8, JOINTS, "all_joints"),
            muscle_corr=_per_channel(rng, 0.5, MUSCLES, "all_muscles"),
            rsa_matrix=rng.normal(0.6, 0.1, (1, len(SUBJECTS))),
            rsa_matrix_pearson=rng.normal(0.6, 0.1, (1, len(SUBJECTS))),
        )
        run.per_subject = PerSubjectMetrics(
            joint_rmse=_per_channel(rng, 5.0 + offset, JOINTS, "all_joints"),
            joint_corr=_per_channel(rng, 0.85, JOINTS, "all_joints"),
            muscle_corr=_per_channel(rng, 0.55, MUSCLES, "all_muscles"),
            rsa_matrix=np.full((1, len(SUBJECTS)), np.nan),
            rsa_matrix_pearson=np.full((1, len(SUBJECTS)), np.nan),
            rsa=list(rng.normal(0.65, 0.1, len(SUBJECTS))),
            subjects_with_rsa_data=list(SUBJECTS),
        )
        analysis.policies.append(run)
    return analysis


def test_compute_wilcoxon_tests(tmp_path):
    analysis = _analysis_with_metrics(tmp_path)

    table = compute_wilcoxon_tests(analysis)
    channel_table = analysis.wilcoxon.per_channel

    # Four metrics, one channel-averaged test each per comparison both conditions carry.
    assert set(table["metric"]) == {
        "Joint angle error [deg]",
        "Joint angle correlation",
        "EMG-muscle correlation",
        "EMG-muscle RSA (Spearman)",
    }
    assert set(table["axis"]) == {"across subjects"}
    assert set(table["comparison_group"]) == {
        "Policy vs policy (subject-matched)",
        "Policy vs policy (population)",
        "Subject-matched vs population",
        "Policy vs human within-subject reliability",
    }
    # The joint error was built 3 deg apart, which every subject and every joint sees.
    joint_error = table[
        (table["metric"] == "Joint angle error [deg]")
        & (table["condition_a"] == "P1 subj.")
        & (table["condition_b"] == "P2 subj.")
    ].squeeze()
    assert joint_error["median_diff"] < 0
    assert joint_error["p_holm"] < 0.05
    # Every joint carries it, and each per-joint test is powered by the 8 subjects -- the
    # three-joint marginal this replaced bottomed out at p = 0.25 and could never say so.
    assert joint_error["n_channels_sig"] == len(JOINTS)
    per_joint = channel_table[
        (channel_table["metric"] == "Joint angle error [deg]")
        & (channel_table["condition_a"] == "P1 subj.")
        & (channel_table["condition_b"] == "P2 subj.")
    ]
    assert (per_joint["n"] == len(SUBJECTS)).all()
    assert per_joint["significant"].all()
    # RSA is a summary over muscles already, so it carries no per-channel breakdown.
    assert "EMG-muscle RSA (Spearman)" not in set(channel_table["metric"])

    assert (tmp_path / "wilcoxon_tests.csv").exists()
    assert (tmp_path / "wilcoxon_per_channel_tests.csv").exists()
    assert (tmp_path / "wilcoxon_conditions.csv").exists()
    # Three views of the same tests: comparison table, condition x condition matrix,
    # and the comparison x joint grid.
    assert (tmp_path / "wilcoxon_joint_angle_error_deg.png").exists()
    assert (tmp_path / "wilcoxon_matrix_joint_angle_error_deg.png").exists()
    assert (tmp_path / "wilcoxon_per_channel_joint_angle_error_deg.png").exists()


def test_policies_without_data_are_skipped(tmp_path):
    analysis = _analysis_with_metrics(tmp_path)
    analysis.policies[1].profiles.avg_joint_gaits[:] = np.nan

    table = compute_wilcoxon_tests(analysis)

    # Only one policy is left, so the remaining comparisons are against the human bound.
    assert "P2 subj." not in set(table["condition_a"]) | set(table["condition_b"])
    assert set(table["comparison_group"]) == {
        "Subject-matched vs population",
        "Policy vs human within-subject reliability",
    }
