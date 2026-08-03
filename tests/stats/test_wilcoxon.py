"""Wilcoxon signed-rank testing of the neuromuscular metrics.

Covers the primitive (paired_wilcoxon / holm_correction) and the
two-readings-per-comparison contract: the channel-averaged test and the
per-channel breakdown, both paired over units and both on matched
units/channels only.
"""

import numpy as np
import pytest

from mm_evalkit.stats.wilcoxon import (
    MetricFamily,
    MetricSamples,
    compare_per_channel,
    compare_samples,
    holm_correction,
    paired_wilcoxon,
    wilcoxon_table,
)

JOINTS = ["hip", "knee", "ankle"]
SUBJECTS = [51, 52, 53, 54, 55, 56, 57, 58]


# --------------------------------------------------------------------------------------
# The primitive
# --------------------------------------------------------------------------------------


def test_paired_wilcoxon_matches_scipy():
    scipy_wilcoxon = pytest.importorskip("scipy.stats").wilcoxon
    a = np.array([1.0, 2.0, 3.5, 4.0, 5.0, 6.0, 7.5])
    b = np.array([1.4, 2.6, 3.1, 4.9, 5.6, 6.8, 7.2])

    result = paired_wilcoxon(a, b)
    expected = scipy_wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")

    assert result["n"] == 7
    assert result["p"] == pytest.approx(expected.pvalue)
    assert result["W"] == pytest.approx(expected.statistic)
    assert result["median_diff"] == pytest.approx(np.median(a - b))


def test_paired_wilcoxon_drops_unmeasured_pairs():
    """A NaN on either side removes the pair; it is unmeasured, not a zero difference."""
    a = np.array([1.0, np.nan, 3.0, 4.0, 5.0, 6.0])
    b = np.array([0.5, 2.0, np.nan, 3.5, 4.5, 5.5])

    result = paired_wilcoxon(a, b)

    assert result["n"] == 4
    assert result["n_nonzero"] == 4
    assert "underpowered" in result["note"]  # 4 < the 6 pairs a two-sided test needs


def test_paired_wilcoxon_effect_size_is_signed_and_bounded():
    a = np.arange(1.0, 9.0)
    assert paired_wilcoxon(a, a - 1.0)["effect_size"] == pytest.approx(1.0)
    assert paired_wilcoxon(a, a + 1.0)["effect_size"] == pytest.approx(-1.0)


def test_paired_wilcoxon_reports_degenerate_samples():
    identical = paired_wilcoxon(np.arange(5.0), np.arange(5.0))
    assert identical["n_nonzero"] == 0
    assert np.isnan(identical["p"])
    assert "identical" in identical["note"]

    empty = paired_wilcoxon([], [])
    assert empty["n"] == 0
    assert "no paired samples" in empty["note"]


def test_holm_correction_is_step_down_and_monotone():
    corrected = holm_correction([0.01, 0.02, 0.03, np.nan])
    assert corrected[0] == pytest.approx(0.03)  # 3 * 0.01
    assert corrected[1] == pytest.approx(0.04)  # 2 * 0.02
    assert corrected[2] == pytest.approx(0.04)  # max(previous, 1 * 0.03)
    assert np.isnan(corrected[3])  # a NaN is not a test and is not counted


# --------------------------------------------------------------------------------------
# Two tests per comparison
# --------------------------------------------------------------------------------------


def _samples(values, units=SUBJECTS, channels=JOINTS):
    return MetricSamples(np.asarray(values, dtype=float), tuple(units), tuple(channels))


def test_compare_samples_averages_over_channels():
    rng = np.random.default_rng(0)
    base = rng.normal(8.0, 2.0, (len(SUBJECTS), len(JOINTS)))
    outcome = compare_samples(_samples(base), _samples(base + 1.0), unit_name="subject")

    assert set(outcome) == {"across subjects"}
    # One value per subject: the channels are averaged away before pairing.
    assert outcome["across subjects"]["n"] == len(SUBJECTS)
    assert outcome["across subjects"]["median_diff"] == pytest.approx(-1.0)


def test_compare_per_channel_pairs_over_units_on_every_channel():
    """Each channel gets its own test, at the *unit* sample size -- not the channel count."""
    rng = np.random.default_rng(0)
    base = rng.normal(8.0, 2.0, (len(SUBJECTS), len(JOINTS)))
    rows = compare_per_channel(_samples(base), _samples(base + 1.0))

    assert [r["channel"] for r in rows] == JOINTS
    for row in rows:
        assert row["n"] == len(SUBJECTS)  # 8 subjects, not 3 joints
        assert row["median_diff"] == pytest.approx(-1.0)
        assert row["effect_size"] == pytest.approx(-1.0)
        # 8 paired subjects clear the resolution floor that 3 joints could never reach.
        assert row["significant"]


def test_compare_per_channel_holm_corrects_within_the_comparison():
    """The family is this comparison's channels, so Holm scales with their count."""
    rng = np.random.default_rng(7)
    base = rng.normal(8.0, 2.0, (len(SUBJECTS), len(JOINTS)))
    rows = compare_per_channel(_samples(base), _samples(base + 1.0))

    assert all(r["p_holm"] >= r["p"] for r in rows)
    assert max(r["p_holm"] for r in rows) <= len(JOINTS) * max(r["p"] for r in rows)


def test_compare_samples_pairs_by_subject_id_not_position():
    """A condition missing a subject must shrink n, never shift the pairing."""
    values = np.tile(np.arange(len(SUBJECTS), dtype=float)[:, None], (1, len(JOINTS)))
    full = _samples(values)
    # Same values, but subject 51 is absent, so the rows sit at different positions.
    partial = _samples(values[1:], units=SUBJECTS[1:])

    outcome = compare_samples(full, partial, unit_name="subject")
    rows = compare_per_channel(full, partial)

    assert outcome["across subjects"]["n"] == len(SUBJECTS) - 1
    # Matched by id, every difference is exactly zero; matched by position it would not be.
    assert outcome["across subjects"]["n_nonzero"] == 0
    assert all(r["n"] == len(SUBJECTS) - 1 and r["n_nonzero"] == 0 for r in rows)


def test_compare_per_channel_skips_metrics_without_a_breakdown():
    """RSA's single column *is* the summary over muscles, so it gets no per-muscle test."""
    rng = np.random.default_rng(1)
    a = MetricSamples.from_vector(rng.uniform(0.3, 0.9, len(SUBJECTS)), SUBJECTS, "muscle RSM")
    b = MetricSamples.from_vector(rng.uniform(0.3, 0.9, len(SUBJECTS)), SUBJECTS, "muscle RSM")

    assert compare_samples(a, b, unit_name="subject")["across subjects"]["n"] == len(SUBJECTS)
    assert compare_per_channel(a, b) == []


def test_metric_samples_reject_misaligned_metric_lists():
    with pytest.raises(ValueError, match="units"):
        MetricSamples.from_per_channel({"hip": [1.0, 2.0]}, SUBJECTS, JOINTS)


def test_metric_samples_tolerate_a_policy_without_data():
    empty = MetricSamples.from_per_channel({j: [] for j in JOINTS}, SUBJECTS, JOINTS)
    assert empty.values.shape == (len(SUBJECTS), len(JOINTS))
    assert np.isnan(empty.values).all()


def test_wilcoxon_table_corrects_within_metric_and_axis():
    rng = np.random.default_rng(2)
    base = rng.normal(8.0, 2.0, (len(SUBJECTS), len(JOINTS)))
    samples = {
        "P1": _samples(base),
        "P2": _samples(base + 1.0),
        "P3": _samples(base + 2.0),
    }
    comparisons = [("Policy vs policy", "P1", "P2"), ("Policy vs policy", "P1", "P3")]
    table, channel_table = wilcoxon_table(
        [MetricFamily("Joint angle error [deg]", "joint", samples, comparisons)], unit_name="subject"
    )

    assert len(table) == 2  # one channel-averaged test per comparison
    assert set(table["axis"]) == {"across subjects"}
    for _, block in table.groupby(["metric", "axis"]):
        # Two tests in each family, so Holm doubles the smaller raw p-value.
        assert block["p_holm"].max() >= block["p"].max()
    assert table["significant"].dtype == bool

    # ... and one row per (comparison, joint) in the per-channel table.
    assert len(channel_table) == len(comparisons) * len(JOINTS)
    assert set(channel_table["channel"]) == set(JOINTS)
    # The breadth counts on the main table are those rows collapsed.
    for _, row in table.iterrows():
        carried = channel_table[
            (channel_table["condition_a"] == row["condition_a"]) & (channel_table["condition_b"] == row["condition_b"])
        ]
        assert row["n_channels"] == len(JOINTS)
        assert row["n_channels_sig"] == int(carried["significant"].sum())


def test_wilcoxon_table_skips_comparisons_whose_condition_is_absent():
    samples = {"P1": _samples(np.zeros((len(SUBJECTS), len(JOINTS))))}
    table, channel_table = wilcoxon_table(
        [MetricFamily("m", "joint", samples, [("g", "P1", "missing")])], unit_name="subject"
    )
    assert table.empty
    assert channel_table.empty
