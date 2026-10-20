"""Publication summary table for ULTRA-MoCap metrics."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba

from mm_evalkit.plotting import policy_fill_alpha

from .constants import (
    BETWEEN_SUBJECT_HATCH,
    BETWEEN_SUBJECT_LABEL,
    METRIC_KEYS,
    REF_GREY,
    REP_TO_REP_KEYS,
    RSA_PAIRS_NOTE,
    WITHIN_SUBJECT_VARIABILITY_HATCH,
    WITHIN_SUBJECT_VARIABILITY_LABEL,
)
from .figures import FIGURE_TITLE_SIZE, PANEL_TITLE_SIZE, SMALL_TEXT_SIZE, TABLE_CELL_SIZE, TICK_SIZE, save_figure
from .references import (
    compute_between_subject_reference,
)
from .summary_plots import _group_scores, _mean_std, _summary_axes, _trial_scores, reference_records


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
    * **Move.** -- each movement is first averaged over its speed conditions and over the
      subjects that have it, and those per-movement numbers are the sample, so a movement
      recorded by three subjects does not outweigh one recorded by two. This is what
      ``aggregated_per_*.png`` draws, and it is the number to quote next to those bars.

    Cells are ``mean ± std``. The between-subject reference is computed between *pairs of
    subjects* within a condition and has no per-trial form, so its Trial cell is "—"; the
    repetition-to-repetition reference is a per-trial quantity and carries both.
    """
    _, groups = _summary_axes(real_records)
    if not groups or not policies:
        print("  no segmented trials to summarise; skipping the metrics table.")
        return

    metric_keys = list(METRIC_KEYS)

    def row(label, color, hatch, records=None, group_source=None, linestyle="-"):
        """One table row: ``records`` fills both columns, ``group_source`` only Move."""
        cells = []
        for key in metric_keys:
            if records is not None:
                cells.append(_mean_std(_trial_scores(records, key)))
                cells.append(_mean_std(_group_scores(records, groups, key, per_trial=True)))
            else:
                cells.append((np.nan, np.nan))
                cells.append(_mean_std(_group_scores(group_source, groups, key, per_trial=False)))
        return {"label": label, "cells": cells, "color": color, "hatch": hatch, "linestyle": linestyle}

    rows = [
        row(
            WITHIN_SUBJECT_VARIABILITY_LABEL,
            REF_GREY,
            WITHIN_SUBJECT_VARIABILITY_HATCH,
            records=reference_records(real_records, REP_TO_REP_KEYS),
        ),
        row(
            BETWEEN_SUBJECT_LABEL,
            REF_GREY,
            BETWEEN_SUBJECT_HATCH,
            group_source=compute_between_subject_reference(real_records),
        ),
    ]
    rows += [
        row(policy.label, policy.color, None, records=policy_records.get(policy.id, {}), linestyle=policy.linestyle)
        for policy in policies
    ]

    metric_names = [f"Joint RMSE [deg] ({arm} arm)", "Muscle-EMG correlation (r)", "Muscle RSA (Spearman)"]
    n_metric_cols = 2 * len(metric_names)
    col_widths = [0.28] + [0.72 / n_metric_cols] * n_metric_cols
    cell_text = [
        [""] * (1 + n_metric_cols),  # metric names drawn spanning
        [""] + ["Trial", "Move."] * len(metric_names),
        *[[row["label"]] + [_format_cell(c) for c in row["cells"]] for row in rows],
    ]

    label_lines = max(str(row["label"]).count("\n") + 1 for row in rows)
    header_units, row_units = 1.6, 1.4 * label_lines
    table_height_in = (TABLE_CELL_SIZE / 72 * 1.2) * (2 * header_units + len(rows) * row_units)
    fig, ax = plt.subplots(figsize=(2.2 * n_metric_cols + 5, table_height_in / 0.77 + 1.2))
    ax.axis("off")
    ax.set_title(
        f"ULTRA-MoCap: policy vs real, aggregated over {len(groups)} movements",
        fontsize=FIGURE_TITLE_SIZE,
        pad=12,
    )

    table = ax.table(cellText=cell_text, cellLoc="center", loc="upper center", colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(TABLE_CELL_SIZE)
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
        # Reference rows are grey and told apart by hatch; policy rows carry their plot
        # color at the same fill alpha the bars use, so same-colored policies stay
        # distinguishable here too.
        alpha = 0.4 if row["hatch"] else policy_fill_alpha(row["linestyle"])
        cell.set_facecolor(to_rgba(row["color"], alpha))
        if row["hatch"]:
            cell.set_edgecolor("silver")
            cell.set_hatch(row["hatch"])
        cell.set_text_props(ha="left", fontsize=TICK_SIZE)

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
            fontsize=PANEL_TITLE_SIZE,
        )

    ax.text(
        0.0,
        min(cell.get_y() for cell in table.get_celld().values()) - 0.04,
        "Trial = one sample per trial (subject x movement x condition); "
        "Move. = one sample per movement, each first averaged over its speeds and subjects.\n"
        "Cells are mean ± std over those samples. "
        f"RSA compares the electrode x electrode correlation matrices ({RSA_PAIRS_NOTE}).",
        transform=ax.transAxes,
        fontsize=SMALL_TEXT_SIZE,
        va="top",
        ha="left",
        color="0.3",
    )

    save_figure(fig, out_path_base, dpi=200, announce="saved")
