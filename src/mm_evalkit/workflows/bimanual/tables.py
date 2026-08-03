"""Publication summary table for ULTRA-MoCap metrics."""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba

from .constants import (
    BETWEEN_SUBJECT_HATCH,
    BETWEEN_SUBJECT_LABEL,
    METRIC_KEYS,
    REF_GREY,
    REP_TO_REP_KEYS,
    RSA_PAIRS_NOTE,
    SPLIT_HALF_KEYS,
    WITHIN_SUBJECT_HATCH,
    WITHIN_SUBJECT_LABEL,
    WITHIN_SUBJECT_VARIABILITY_HATCH,
    WITHIN_SUBJECT_VARIABILITY_LABEL,
)
from .references import (
    compute_between_subject_reference,
)
from .summary_plots import _aggregate_over_subjects, _summary_axes


def _trial_scores(records, key):
    """One score per trial: that trial's metric averaged over its joints / electrodes.

    Trials with no value for the metric (no EMG, too few repetitions to split in half)
    simply do not contribute a sample.
    """
    scores = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for metrics in records.values():
            values = metrics.get(key)
            if values is None or not len(values):
                continue
            score = float(np.nanmean(values))
            if np.isfinite(score):
                scores.append(score)
    return scores


def _condition_scores(source, groups, key, per_trial):
    """One score per movement/condition, so a condition is one sample whatever its size.

    ``per_trial`` says how ``source`` is keyed: True for a records dict keyed by
    ``(movement, condition, subject)`` (each condition is first averaged over the
    subjects that have it, which is what the aggregated bar charts draw), False for a
    reference already keyed by ``(movement, condition)``.
    """
    if per_trial:
        means, _ = _aggregate_over_subjects(source, groups, key)
        return [m for m in means if np.isfinite(m)]
    scores = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for group in groups:
            values = source.get(group, {}).get(key)
            if values is None or not len(values):
                continue
            score = float(np.nanmean(values))
            if np.isfinite(score):
                scores.append(score)
    return scores


def _mean_std(scores):
    """``(mean, std)`` of a sample, or ``(nan, nan)`` when it is empty."""
    values = np.asarray([v for v in scores if np.isfinite(v)], dtype=float)
    if not values.size:
        return np.nan, np.nan
    return float(np.mean(values)), float(np.std(values))


def _format_cell(cell):
    """``mean ± std``, rounded to the precision the spread actually supports."""
    mean, std = cell
    if not np.isfinite(mean):
        return "—"
    # Show the std to one significant figure and the mean to the same decimal place,
    # clamped to 1-2 decimals, so a cell never carries more digits than the spread supports.
    decimals = int(np.clip(-np.floor(np.log10(std)), 1, 2)) if np.isfinite(std) and std > 0 else 2
    if abs(mean) < 0.5 * 10.0**-decimals:
        mean = 0.0  # a value that rounds to zero must not print as "-0.0"
    if not np.isfinite(std):
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def plot_metrics_table(policy_records, policies, real_records, out_path_base, arm="right"):
    """Summary table of the aggregated scores -- the bimanual ``metrics_summary_table``.

    The analogue of the table the full-body analysis writes, with the same shape:
    grey hatched reference rows first, then one row per policy, and each metric split into
    the two aggregation levels the figures use.

    * **Trial** -- every trial (one subject x movement x condition) is one sample, scored on
      its repetition-averaged cycle, so the spread is over trials. This is what
      ``summary_per_*.png`` draws.
    * **Cond.** -- each movement/condition is first averaged over the subjects that have it
      and those per-condition numbers are the sample, so a condition recorded by three
      subjects does not outweigh one recorded by two. This is what ``aggregated_per_*.png``
      draws, and it is the number to quote next to those bars.

    Cells are ``mean ± std``. The between-subject reference is computed between *pairs of
    subjects* within a condition and has no per-trial form, so its Trial cell is "—"; the
    other two references are per-trial quantities and carry both.
    """
    _, groups = _summary_axes(real_records)
    if not groups or not policies:
        print("  no segmented trials to summarise; skipping the metrics table.")
        return

    metric_keys = list(METRIC_KEYS)

    def as_records(reference_keys):
        """A records-shaped view of one real-data reference, keyed like a policy's records.

        The split-half ceiling and the repetition-to-repetition variability are both scored
        per trial, just under a different key, so re-keying them lets both aggregation
        levels come from the same helpers the policy rows use.
        """
        return {
            key: {name: metrics[source] for name, source in reference_keys.items() if source in metrics}
            for key, metrics in real_records.items()
        }

    def row(label, color, hatch, records=None, group_source=None):
        """One table row: ``records`` fills both columns, ``group_source`` only Cond."""
        cells = []
        for key in metric_keys:
            if records is not None:
                cells.append(_mean_std(_trial_scores(records, key)))
                cells.append(_mean_std(_condition_scores(records, groups, key, per_trial=True)))
            else:
                cells.append((np.nan, np.nan))
                cells.append(_mean_std(_condition_scores(group_source, groups, key, per_trial=False)))
        return {"label": label, "cells": cells, "color": color, "hatch": hatch}

    rows = [
        row(WITHIN_SUBJECT_LABEL, REF_GREY, WITHIN_SUBJECT_HATCH, records=as_records(SPLIT_HALF_KEYS)),
        row(
            WITHIN_SUBJECT_VARIABILITY_LABEL,
            REF_GREY,
            WITHIN_SUBJECT_VARIABILITY_HATCH,
            records=as_records(REP_TO_REP_KEYS),
        ),
        row(
            BETWEEN_SUBJECT_LABEL,
            REF_GREY,
            BETWEEN_SUBJECT_HATCH,
            group_source=compute_between_subject_reference(real_records),
        ),
    ]
    rows += [row(policy.label, policy.color, None, records=policy_records.get(policy.id, {})) for policy in policies]

    metric_names = [f"Joint RMSE [deg] ({arm} arm)", "Muscle-EMG correlation (r)", "Muscle RSA (Spearman)"]
    n_metric_cols = 2 * len(metric_names)
    col_widths = [0.28] + [0.72 / n_metric_cols] * n_metric_cols
    cell_text = [
        [""] * (1 + n_metric_cols),  # metric names drawn spanning
        [""] + ["Trial", "Cond."] * len(metric_names),
        *[[row["label"]] + [_format_cell(c) for c in row["cells"]] for row in rows],
    ]

    label_lines = max(str(row["label"]).count("\n") + 1 for row in rows)
    header_units, row_units = 1.6, 1.4 * label_lines
    table_height_in = (9 / 72 * 1.2) * (2 * header_units + len(rows) * row_units)
    fig, ax = plt.subplots(figsize=(2.2 * n_metric_cols + 5, table_height_in / 0.77 + 1.2))
    ax.axis("off")
    ax.set_title(f"ULTRA-MoCap: policy vs real, aggregated over {len(groups)} movement/conditions", fontsize=12, pad=12)

    table = ax.table(cellText=cell_text, cellLoc="center", loc="upper center", colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    base_height = max(cell.get_height() for cell in table.get_celld().values())
    for (r, _c), cell in table.get_celld().items():
        cell.set_height(base_height * (header_units if r < 2 else row_units))

    # Two dark header rows; the group row's borders blend into the fill so each metric
    # reads as one seamless band behind its spanning label.
    for c in range(1 + n_metric_cols):
        for r in (0, 1):
            header = table[(r, c)]
            header.set_facecolor("#404040")
            header.set_text_props(color="white", fontweight="bold" if r == 0 else "normal")
            if r == 0:
                header.set_edgecolor("#404040")

    for i, row in enumerate(rows):
        cell = table[(i + 2, 0)]
        cell.set_facecolor(to_rgba(row["color"], 0.4 if row["hatch"] else 0.35))
        if row["hatch"]:
            cell.set_edgecolor("silver")
            cell.set_hatch(row["hatch"])
        cell.set_text_props(ha="left", fontsize=8)

    fig.canvas.draw()
    for m, name in enumerate(metric_names):
        left, right = table[(0, 1 + 2 * m)], table[(0, 2 + 2 * m)]
        ax.text(
            (left.get_x() + right.get_x() + right.get_width()) / 2,
            left.get_y() + left.get_height() / 2,
            name,
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=10,
        )

    ax.text(
        0.0,
        min(cell.get_y() for cell in table.get_celld().values()) - 0.04,
        "Trial = one sample per trial (subject x movement x condition); "
        "Cond. = one sample per movement/condition, each first averaged over its subjects.\n"
        "Cells are mean ± std over those samples. "
        f"RSA compares the electrode x electrode correlation matrices ({RSA_PAIRS_NOTE}).",
        transform=ax.transAxes,
        fontsize=7,
        va="top",
        ha="left",
        color="0.3",
    )

    fig.savefig(f"{out_path_base}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{out_path_base}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path_base}.png")
