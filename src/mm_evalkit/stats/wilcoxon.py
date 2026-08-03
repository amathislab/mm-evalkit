"""Wilcoxon signed-rank testing of the neuromuscular metrics.

Every metric the neuromuscular analyses report -- joint RMSE, joint correlation,
muscle/EMG correlation, muscle RSA -- is a *matrix* of paired measurements: one value
per (unit, channel), where a unit is a subject. ULTRA-MoCap trials are matched between
conditions and averaged within subject before they reach this module. The same
units and channels are scored under every condition being compared -- two policies, a
policy and a human reference, the population- and subject-matched variant of the same
policy -- so comparing two conditions is a *paired* problem.

Every test here pairs over **units**, the only axis with enough samples to support an
exact test (see the resolution note below). The matrix is read in two ways:

* **across units** (``across subjects``) -- average each condition
  over the channels first, leaving one number per unit, and pair the two conditions
  unit by unit. One test per comparison. Answers "is A closer to the humans than B,
  subject for subject?".
* **per channel** (``per joint`` / ``per muscle``) -- do *not* average over the
  channels: run the same paired-over-units test separately on each joint / muscle, and
  Holm-correct across the channels of that one comparison. Answers "*which* joints and
  muscles carry the difference, and how many of them?".

The second reading is a decomposition of the first, not an independent axis -- both
pair over the same units. What it adds is breadth. The channel-averaged test cannot
tell a difference concentrated in two joints from the same difference spread thinly
over ten, because it sees the same mean either way; the per-channel breakdown separates
them, reporting ``k of N channels`` plus the names. It deliberately makes no claim
about channels as a population: these are named, non-exchangeable anatomical joints and
muscles, not a sample from some larger set, so "does this generalise over joints" is
not the question being asked.

A metric with no channel breakdown (RSA, which is already a summary over all muscles)
carries a single column and gets no per-channel report rather than a faked one.

The test itself is the two-sided **Wilcoxon signed-rank** test on the paired
differences. It is non-parametric on purpose: these samples are small (10-20 units) and
bounded/skewed (correlations piling up near 1, RMSEs with a hard floor at 0), so a
paired t-test's normality assumption is not safe. Pairing is always by subject id /
channel name -- never by position -- and only the units and channels present on *both*
sides are used, so a policy that lost a subject simply contributes a smaller n instead
of a silently misaligned comparison.

**Why the channels are not paired over.** An earlier version of this module also ran
the mirror test -- average over the units first, then pair the two conditions channel by
channel. With 10 joints that test cannot work, for arithmetic rather than empirical
reasons. A two-sided signed-rank test on 10 pairs has a smallest attainable p of
2 / 2**10 = 1.95e-3, and Holm correction across the ~40 comparisons in a metric raises
the smallest attainable *corrected* p to 0.078 -- above alpha, whatever the data. No
choice of statistic fixes it: any exact test whose null is sign-flip exchangeability
over 10 channels shares that floor. Pairing over the 19-20 units instead puts the floor
at 2 / 2**20, and Holm across the 10 channels of a single comparison leaves it near
2e-5, which is why the per-channel reading replaced it.

Reported alongside each p-value:

* ``n`` -- paired samples actually used (after dropping non-finite and unmatched ones),
  and ``n_nonzero``, the pairs with a non-zero difference, which is what the test
  actually ranks. A two-sided signed-rank test cannot reach p < 0.05 with fewer than
  6 non-zero pairs *whatever* the data, so smaller samples are flagged as underpowered
  rather than reported as a plain "not significant".
* ``median_diff`` -- the median of A - B, the effect in the metric's own units.
* ``effect_size`` -- the matched-pairs rank-biserial correlation, ``(W+ - W-) / (W+ +
  W-)``, in [-1, 1]: +1 means A exceeded B on every pair.
* ``p_holm`` -- Holm-Bonferroni corrected within each (metric, axis) family, which is
  the set of comparisons a reader scans together.
"""

from __future__ import annotations

import warnings
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

# A two-sided Wilcoxon signed-rank test on n non-zero pairs has a smallest attainable
# p-value of 2 / 2**n, so it cannot reach 0.05 before n = 6. Below that the test is
# reported but flagged: "not significant" would misrepresent an impossible verdict.
MIN_INFORMATIVE_PAIRS = 6

# Significance threshold used for the ``significant`` column and the rendered tables.
ALPHA = 0.05


class MetricSamples(NamedTuple):
    """One condition's values for one metric: ``(n_units, n_channels)`` plus labels.

    ``units`` are the subject ids of the first axis and ``channels`` the labels of the
    second (joint or muscle / electrode names). Missing
    measurements are NaN and are dropped pairwise by the tests.
    """

    values: np.ndarray
    units: tuple
    channels: tuple

    @classmethod
    def from_per_channel(cls, per_channel, units, channels):
        """Build from ``{channel_name: [one value per unit]}`` -- the metric dicts.

        A channel absent from ``per_channel`` (or present with an empty list, as for a
        policy whose trajectory file was missing) becomes an all-NaN column rather than
        an error, so a policy with no data still takes part in the report with n = 0.
        """
        values = np.full((len(units), len(channels)), np.nan)
        for c, name in enumerate(channels):
            column = per_channel.get(name)
            if column is None or len(column) == 0:
                continue
            column = np.asarray(column, dtype=float)
            if column.shape != (len(units),):
                raise ValueError(
                    f"channel '{name}' has {column.shape[0]} values but {len(units)} units were declared; "
                    "the metric list and its unit ids must be built in the same order"
                )
            values[:, c] = column
        return cls(values, tuple(units), tuple(channels))

    @classmethod
    def from_vector(cls, values, units, channel="all"):
        """Build a single-column condition from one value per unit (e.g. RSA)."""
        values = np.asarray(values, dtype=float).reshape(-1, 1)
        if values.shape[0] != len(units):
            raise ValueError(f"{values.shape[0]} values for {len(units)} units")
        return cls(values, tuple(units), (channel,))


def paired_wilcoxon(a, b):
    """Two-sided Wilcoxon signed-rank test on two aligned vectors.

    Pairs where either side is non-finite are dropped (they are unmeasured, not zero),
    and zero differences are dropped by the test itself (``zero_method="wilcox"``), so
    ``n_nonzero`` is the sample the ranks are actually built from. Returns the row of
    numbers described in the module docstring; ``p`` is NaN when there is nothing to
    test (no pairs, or every difference exactly zero).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired samples must have the same shape, got {a.shape} and {b.shape}")

    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]

    result = {
        "n": int(a.size),
        "n_nonzero": 0,
        "median_a": np.nan,
        "median_b": np.nan,
        "median_diff": np.nan,
        "W": np.nan,
        "p": np.nan,
        "effect_size": np.nan,
        "note": "",
    }
    if a.size == 0:
        result["note"] = "no paired samples"
        return result

    diff = a - b
    result["median_a"] = float(np.median(a))
    result["median_b"] = float(np.median(b))
    result["median_diff"] = float(np.median(diff))

    nonzero = diff[diff != 0.0]
    result["n_nonzero"] = int(nonzero.size)
    if nonzero.size == 0:
        result["note"] = "identical samples (all differences zero)"
        return result

    # Matched-pairs rank-biserial correlation: the signed share of the total rank mass.
    ranks = rankdata(np.abs(nonzero))
    w_plus = float(ranks[nonzero > 0].sum())
    w_minus = float(ranks[nonzero < 0].sum())
    result["effect_size"] = (w_plus - w_minus) / (w_plus + w_minus)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        statistic, p_value = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    result["W"] = float(statistic)
    result["p"] = float(p_value)

    notes = [str(w.message) for w in caught]
    if nonzero.size < MIN_INFORMATIVE_PAIRS:
        notes.insert(
            0,
            f"underpowered: {nonzero.size} non-zero pairs, at least {MIN_INFORMATIVE_PAIRS} "
            f"are needed for a two-sided p < {ALPHA}",
        )
    result["note"] = "; ".join(notes)
    return result


def _shared_unit_indices(a: MetricSamples, b: MetricSamples):
    """Positions of the units present in both conditions, in ``a``'s order."""
    b_units = set(b.units)
    shared = [u for u in a.units if u in b_units]
    return [a.units.index(u) for u in shared], [b.units.index(u) for u in shared]


def compare_samples(a: MetricSamples, b: MetricSamples, unit_name="subject"):
    """The channel-averaged Wilcoxon test for one pair of conditions.

    Restricted to the units present in both conditions, and averaging each
    condition over its own channels.

    Returns ``{f"across {unit_name}s": result}``.
    """
    index_a, index_b = _shared_unit_indices(a, b)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        by_unit = paired_wilcoxon(
            np.nanmean(a.values[index_a], axis=1),
            np.nanmean(b.values[index_b], axis=1),
        )
    return {f"across {unit_name}s": by_unit}


def compare_per_channel(a: MetricSamples, b: MetricSamples, alpha=ALPHA):
    """One paired-over-units test per shared channel, Holm-corrected across them.

    The channels of a single comparison are the family here: the reader asks "on which
    joints does A beat B?" of one comparison at a time, and correcting across ~10
    channels leaves the test its resolution (unlike correcting across every comparison
    in the metric, which is what made the old channel-paired test unusable).

    A pair of conditions sharing fewer than two channels has no breakdown to report and
    returns an empty list -- this is the RSA case, whose single column *is* the summary
    over muscles, so a "per muscle" test on it would just restate the across-units one.
    """
    index_a, index_b = _shared_unit_indices(a, b)
    b_channels = set(b.channels)
    shared_channels = [c for c in a.channels if c in b_channels]
    if len(shared_channels) < 2:
        return []

    rows = []
    for name in shared_channels:
        column_a = a.values[index_a, a.channels.index(name)]
        column_b = b.values[index_b, b.channels.index(name)]
        rows.append({"channel": name, **paired_wilcoxon(column_a, column_b)})

    for row, corrected in zip(rows, holm_correction([r["p"] for r in rows]), strict=True):
        row["p_holm"] = float(corrected) if np.isfinite(corrected) else np.nan
        row["significant"] = bool(np.isfinite(corrected) and corrected < alpha)
    return rows


def breadth_summary(rows):
    """Collapse per-channel rows into the ``k of N`` breadth counts for the main table.

    ``n_channels_sig_positive`` / ``_negative`` split the significant channels by the
    sign of their median difference, so a comparison that wins on some channels and
    loses on others is visible as such instead of averaging to "5 of 10".
    """
    tested = [r for r in rows if np.isfinite(r["p"])]
    significant = [r for r in tested if r["significant"]]
    positive = sum(1 for r in significant if r["median_diff"] > 0)
    return {
        "n_channels": len(rows),
        "n_channels_tested": len(tested),
        "n_channels_sig": len(significant),
        "n_channels_sig_positive": positive,
        "n_channels_sig_negative": len(significant) - positive,
    }


def holm_correction(p_values):
    """Holm-Bonferroni step-down correction; NaN entries stay NaN and are not counted."""
    p_values = np.asarray(p_values, dtype=float)
    corrected = np.full(p_values.shape, np.nan)
    finite = np.flatnonzero(np.isfinite(p_values))
    if finite.size == 0:
        return corrected
    order = finite[np.argsort(p_values[finite], kind="stable")]
    m = order.size
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p_values[index])
        corrected[index] = min(running, 1.0)
    return corrected


class MetricFamily(NamedTuple):
    """One metric and the comparisons to run on it.

    ``samples`` maps a *short* condition name (the label the tables show) to its
    :class:`MetricSamples`; ``comparisons`` lists ``(group, condition_a, condition_b)``
    in report order, where ``group`` is the heading the comparison belongs under (e.g.
    "Policy vs policy"). Naming the comparisons explicitly, rather than testing every
    pair of conditions, keeps the report to the contrasts that mean something.
    """

    name: str  # "Joint angle error [deg]"
    channel_name: str  # "joint" -> the across-channels test is titled "across joints"
    samples: dict
    comparisons: list


def wilcoxon_table(families, unit_name="subject", alpha=ALPHA):
    """Run every comparison of every family; return ``(table, per_channel)``.

    ``table`` carries one row per comparison: the channel-averaged test paired over
    units, Holm-corrected within each (metric, axis) family -- the set of comparisons a
    reader scans together -- plus the ``k of N`` breadth counts of that comparison's
    per-channel tests.

    ``per_channel`` carries one row per (comparison, channel): the same paired-over-units
    test run on each joint / muscle separately, Holm-corrected across the channels of
    its own comparison. It is empty for metrics with no channel breakdown.
    """
    rows, channel_rows = [], []
    for family in families:
        for group, name_a, name_b in family.comparisons:
            a, b = family.samples.get(name_a), family.samples.get(name_b)
            if a is None or b is None:
                continue

            per_channel = compare_per_channel(a, b, alpha=alpha)
            for result in per_channel:
                channel_rows.append(
                    {
                        "metric": family.name,
                        "channel_kind": family.channel_name,
                        "comparison_group": group,
                        "condition_a": name_a,
                        "condition_b": name_b,
                        **result,
                    }
                )

            outcome = compare_samples(a, b, unit_name=unit_name)
            for axis, result in outcome.items():
                rows.append(
                    {
                        "metric": family.name,
                        "axis": axis,
                        "channel_kind": family.channel_name,
                        "comparison_group": group,
                        "condition_a": name_a,
                        "condition_b": name_b,
                        **result,
                        **breadth_summary(per_channel),
                    }
                )

    table = pd.DataFrame(rows)
    channel_table = pd.DataFrame(channel_rows)
    if table.empty:
        return table, channel_table

    table["p_holm"] = np.nan
    for _, block in table.groupby(["metric", "axis"], sort=False):
        table.loc[block.index, "p_holm"] = holm_correction(block["p"].to_numpy())
    table["significant"] = table["p_holm"] < alpha
    return table, channel_table
