"""Rendering of the Wilcoxon results computed by :mod:`mm_evalkit.stats.wilcoxon`.

Consumes the ``(table, channel_table)`` frames only. It applies no correction and
runs no test of its own, so a change here can alter how a result reads but never
what it is.
"""

from __future__ import annotations

import os
import re
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, to_rgba

from .wilcoxon import ALPHA


def _stars(p_value, alpha=ALPHA):
    """Conventional significance marker for a (corrected) p-value."""
    if p_value is None or not np.isfinite(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < alpha:
        return "*"
    return "n.s."


def _format_p(p_value):
    if p_value is None or not np.isfinite(p_value):
        return "—"
    if p_value < 1e-4:
        return "<1e-4"
    return f"{p_value:.4f}"


def _format_value(value, decimals=3):
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.{decimals}f}"


def _slugify(text):
    return re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_").lower() or "metric"


def _format_breadth(row, with_kind=True):
    """``k/N joints`` for the breadth counts, split by sign when they disagree."""
    total = int(row.get("n_channels_tested", 0) or 0)
    if total == 0:
        return "—"
    positive = int(row.get("n_channels_sig_positive", 0) or 0)
    negative = int(row.get("n_channels_sig_negative", 0) or 0)
    kind = f" {row.get('channel_kind', 'channel')}s" if with_kind else ""
    if positive and negative:
        return f"{positive}+/{negative}- of {total}{kind}"
    return f"{positive + negative}/{total}{kind}"


def print_wilcoxon_summary(table, channel_table=None, unit_name="subject", alpha=ALPHA):
    """Print every test grouped by metric, marking the significant ones.

    Each comparison prints its channel-averaged test followed by the breadth of its
    per-channel tests; ``channel_table`` additionally names the channels that carry it.
    """
    if table.empty:
        print("  No Wilcoxon comparisons available (not enough conditions with data).")
        return
    print("\n ### --------------------Wilcoxon signed-rank tests-------------------- ### ")
    print(
        f"  Two-sided, paired over {unit_name}s; p Holm-corrected within each (metric, axis) "
        f"family; alpha={alpha}. 'across {unit_name}s' averages each condition over its "
        "channels first; 'breadth' re-runs the same test on each channel separately, "
        "Holm-corrected across that comparison's channels."
    )
    significant_channels = {}
    if channel_table is not None and not channel_table.empty:
        for key, block in channel_table[channel_table["significant"]].groupby(
            ["metric", "condition_a", "condition_b"], sort=False
        ):
            significant_channels[key] = list(block["channel"])

    for metric, metric_block in table.groupby("metric", sort=False):
        print(f"\n  {metric}")
        for axis, axis_block in metric_block.groupby("axis", sort=False):
            print(f"    [{axis}]")
            for _, row in axis_block.iterrows():
                marker = _stars(row["p_holm"], alpha)
                note = f"   ({row['note']})" if row["note"] else ""
                print(
                    f"      {row['condition_a']} vs {row['condition_b']}: "
                    f"n={row['n']}, median diff={_format_value(row['median_diff'])}, "
                    f"r={_format_value(row['effect_size'], 2)}, "
                    f"p={_format_p(row['p'])}, p_holm={_format_p(row['p_holm'])} {marker}{note}"
                )
                breadth = _format_breadth(row)
                if breadth != "—":
                    carried = significant_channels.get((metric, row["condition_a"], row["condition_b"]), [])
                    names = f": {', '.join(carried)}" if carried else ""
                    print(f"          breadth: {breadth}{names}")


def _render_metric_matrix(metric, block, out_base, alpha=ALPHA, subtitle=""):
    """Condition x condition matrix of the same tests, one panel per axis.

    The table figure lists the comparisons; this shows them as the matrix they form, so
    a block of policies can be read at a glance instead of row by row. Cell (row, column)
    is the test of *row* against *column*: tinted by the rank-biserial effect size (red =
    the row's values are the larger ones, blue = the column's) and annotated with the
    median difference and the Holm-corrected significance. The matrix is antisymmetric by
    construction, so each comparison fills both of its cells with opposite signs.

    Non-significant cells are washed out toward white so the significant ones stand out,
    and a pair that was never compared stays blank -- distinguishable from a pair that
    was compared and came out flat.
    """
    axes_names = list(dict.fromkeys(block["axis"]))
    conditions = list(dict.fromkeys(c for _, row in block.iterrows() for c in (row["condition_a"], row["condition_b"])))
    n = len(conditions)
    if n < 2:
        return
    position = {name: i for i, name in enumerate(conditions)}

    # Cell size shrinks as conditions are added, so a 4-condition matrix stays compact and a
    # 20-condition one stays legible instead of growing to a wall of tiny cells.
    cell_size = float(np.clip(12.0 / n, 0.6, 1.1))
    annotation_size = float(np.clip(cell_size * 8.0, 5.5, 8.0))
    tick_size = float(np.clip(cell_size * 8.5, 6.0, 9.0))
    fig_width = len(axes_names) * (n * cell_size + 1.4) + 1.2
    caption_lines = 2 + (1 if subtitle else 0)
    fig_height = n * cell_size + 2.6 + caption_lines * (8 / 72 * 1.6)
    fig, panels = plt.subplots(
        1, len(axes_names), figsize=(fig_width, fig_height), squeeze=False, constrained_layout=True
    )
    panels = panels[0]

    colormap = plt.get_cmap("RdBu_r")
    scale = Normalize(vmin=-1.0, vmax=1.0)
    image = None
    for panel, axis in zip(panels, axes_names, strict=True):
        rgba = np.ones((n, n, 4))
        annotations = {}
        for _, row in block[block["axis"] == axis].iterrows():
            if not np.isfinite(row["effect_size"]):
                continue
            i, j = position[row["condition_a"]], position[row["condition_b"]]
            for (r, c), sign in (((i, j), 1.0), ((j, i), -1.0)):
                color = np.array(colormap(scale(sign * row["effect_size"])))
                if not row["significant"]:
                    # Wash a non-significant cell toward white: the effect is still
                    # visible as a hue, but it cannot be mistaken for a result.
                    color[:3] = 1.0 - 0.35 * (1.0 - color[:3])
                rgba[r, c] = color
                # A strong effect fills the cell with a dark red/blue, so the annotation
                # flips to white there rather than disappearing into the background.
                luminance = float(color[:3] @ (0.299, 0.587, 0.114))
                annotations[(r, c)] = (
                    f"{sign * row['median_diff']:+.2f}\n{_stars(row['p_holm'], alpha)}",
                    "white" if luminance < 0.5 else "0.15",
                )
        for i in range(n):
            rgba[i, i] = to_rgba("0.9")  # a condition against itself is not a test

        image = panel.imshow(rgba, interpolation="nearest")
        panel.set_xticks(range(n), conditions, fontsize=tick_size, rotation=35, ha="right")
        panel.set_yticks(range(n), conditions, fontsize=tick_size)
        panel.set_title(axis, fontsize=11, fontweight="bold")
        panel.set_xticks(np.arange(n + 1) - 0.5, minor=True)
        panel.set_yticks(np.arange(n + 1) - 0.5, minor=True)
        panel.grid(which="minor", color="white", linewidth=1.5)
        panel.tick_params(which="minor", length=0)
        for (r, c), (text, color) in annotations.items():
            panel.text(c, r, text, ha="center", va="center", fontsize=annotation_size, color=color)

    if image is not None:
        bar = fig.colorbar(plt.cm.ScalarMappable(norm=scale, cmap=colormap), ax=panels, fraction=0.03, pad=0.02)
        bar.set_label("effect size (rank-biserial): + = the row's values are the larger", fontsize=8)
        bar.ax.tick_params(labelsize=7)

    fig.suptitle(f"Wilcoxon signed-rank tests — {metric}", fontsize=13)
    caption = [subtitle] if subtitle else []
    caption.append(
        "Cell (row, column) = row tested against column; the number is the median of "
        "(row - column). Blank = never compared, grey diagonal = not a test."
    )
    caption.append(
        f"Colour washed out where the Holm-corrected p is not significant; *** p<0.001, ** p<0.01, * p<{alpha}."
    )
    fig.text(0.005, 0.002, "\n".join(caption), fontsize=8, va="bottom", ha="left", color="0.3")

    fig.savefig(f"{out_base}.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{out_base}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_base}.png")


def _render_metric_figure(metric, block, out_base, legend=None, alpha=ALPHA, subtitle=""):
    """One table figure for a metric: rows are comparisons, one column block per axis."""
    axes_names = list(dict.fromkeys(block["axis"]))
    keyed = {
        (row["axis"], row["condition_a"], row["condition_b"], row["comparison_group"]): row
        for _, row in block.iterrows()
    }
    comparisons = list(
        dict.fromkeys((row["comparison_group"], row["condition_a"], row["condition_b"]) for _, row in block.iterrows())
    )

    per_axis_cols = ["n", "Δ median", "p", "p (Holm)", "breadth"]
    n_cols = 2 + len(axes_names) * len(per_axis_cols)

    cell_text, cell_significant = [], []
    previous_group = None
    for group, name_a, name_b in comparisons:
        # The group name is only written on the first row of its block, wrapped to the
        # column width so a long heading grows the row instead of spilling over its cell.
        heading = "" if group == previous_group else textwrap.fill(group, width=26)
        row_cells = [heading, f"{name_a}\nvs {name_b}"]
        row_significant = [False, False]
        previous_group = group
        for axis in axes_names:
            row = keyed.get((axis, name_a, name_b, group))
            if row is None:
                row_cells += ["—"] * len(per_axis_cols)
                row_significant += [False] * len(per_axis_cols)
                continue
            marker = _stars(row["p_holm"], alpha)
            row_cells += [
                str(int(row["n"])),
                _format_value(row["median_diff"]),
                _format_p(row["p"]),
                f"{_format_p(row['p_holm'])} {marker}".strip(),
                _format_breadth(row, with_kind=False),
            ]
            row_significant += [False, False, False, bool(row["significant"]), False]
        cell_text.append(row_cells)
        cell_significant.append(row_significant)

    # Two header rows (axis name spanning its block, then the column names).
    header_group = [""] * n_cols
    header_sub = ["Comparison group", "A vs B"] + per_axis_cols * len(axes_names)

    col_widths = [0.16, 0.24] + [0.60 / (n_cols - 2)] * (n_cols - 2)

    # A table cell's height unit is one line of text (fontsize 8, 1.2 line spacing), so the
    # table's height is known up front: size the figure so the table fills the default axes
    # box instead of being stranded at the top of a tall empty one. The caption is then
    # anchored directly under the last row and ``bbox_inches="tight"`` crops the rest.
    header_units = 1.0
    row_units = 0.95 * max(cell.count("\n") + 1 for row in cell_text for cell in row)
    line_height_in = 8 / 72 * 1.2
    table_height_in = line_height_in * (2 * header_units + len(cell_text) * row_units)
    caption_lines = 2 + (1 if subtitle else 0) + (len(legend) + 1 if legend else 0)
    fig_height = table_height_in / 0.77 + 0.8 + caption_lines * (7 / 72 * 1.45)
    fig, ax = plt.subplots(figsize=(3.2 + 1.55 * (n_cols - 2), fig_height))
    ax.axis("off")
    ax.set_title(f"Wilcoxon signed-rank tests — {metric}", fontsize=12, pad=14)

    table = ax.table(
        cellText=[header_group, header_sub, *cell_text],
        cellLoc="center",
        loc="upper center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)

    base_height = max(cell.get_height() for cell in table.get_celld().values())
    for (r, _c), cell in table.get_celld().items():
        cell.set_height(base_height * (header_units if r < 2 else row_units))

    for c in range(n_cols):
        for r in (0, 1):
            header = table[(r, c)]
            header.set_facecolor("#404040")
            header.set_text_props(color="white", fontweight="bold" if r == 0 else "normal")
            if r == 0:
                header.set_edgecolor("#404040")

    for i, row_significant in enumerate(cell_significant):
        r = i + 2
        table[(r, 0)].set_text_props(fontsize=7.5)
        table[(r, 1)].set_text_props(fontsize=7.5)
        for c, is_significant in enumerate(row_significant):
            if is_significant:
                table[(r, c)].set_facecolor("#d8ecd8")

    # Overlay each axis name centered across its block of columns.
    fig.canvas.draw()
    for a, axis in enumerate(axes_names):
        first = table[(0, 2 + a * len(per_axis_cols))]
        last = table[(0, 1 + (a + 1) * len(per_axis_cols))]
        ax.text(
            (first.get_x() + last.get_x() + last.get_width()) / 2,
            first.get_y() + first.get_height() / 2,
            axis,
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=10,
        )

    caption = [subtitle] if subtitle else []
    caption.append(
        "n = paired samples; Δ median = median of (A - B) over them. p Holm-corrected within "
        f"this metric and axis; *** p<0.001, ** p<0.01, * p<{alpha}."
    )
    caption.append(
        "breadth = channels on which the same paired test is significant on its own, Holm-corrected "
        "across that comparison's channels ('2+/1-' = 2 favour A, 1 favours B); see the per-channel figure."
    )
    if legend:
        caption.append("")
        caption += [f"{short}: {str(full).replace(chr(10), ' ')}" for short, full in legend.items()]
    table_bottom = min(cell.get_y() for cell in table.get_celld().values())
    ax.text(
        0.0,
        table_bottom - 0.03,
        "\n".join(caption),
        transform=ax.transAxes,
        fontsize=7,
        va="top",
        ha="left",
        color="0.3",
    )

    fig.savefig(f"{out_base}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{out_base}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_base}.png")


def _render_per_channel_figure(metric, block, out_base, alpha=ALPHA, subtitle=""):
    """Comparison x channel grid of the per-channel tests -- where each effect lives.

    Every cell is one Wilcoxon test paired over units on that channel alone, so a row
    reads as the anatomy of a comparison: a difference concentrated in two joints and
    one spread over all ten look identical in the channel-averaged table and completely
    different here. Cells are tinted by the rank-biserial effect size (red = A's values
    are the larger), annotated with the median difference and the Holm-corrected
    significance, and washed out where that channel's test is not significant.
    """
    comparisons = list(dict.fromkeys(zip(block["condition_a"], block["condition_b"], strict=True)))
    channels = list(dict.fromkeys(block["channel"]))
    n_rows, n_cols = len(comparisons), len(channels)
    if not n_rows or not n_cols:
        return
    row_of = {pair: i for i, pair in enumerate(comparisons)}
    column_of = {name: i for i, name in enumerate(channels)}

    colormap = plt.get_cmap("RdBu_r")
    scale = Normalize(vmin=-1.0, vmax=1.0)
    rgba = np.ones((n_rows, n_cols, 4))
    annotations = {}
    for _, row in block.iterrows():
        if not np.isfinite(row["effect_size"]):
            continue
        r = row_of[(row["condition_a"], row["condition_b"])]
        c = column_of[row["channel"]]
        color = np.array(colormap(scale(row["effect_size"])))
        if not row["significant"]:
            color[:3] = 1.0 - 0.30 * (1.0 - color[:3])
        rgba[r, c] = color
        luminance = float(color[:3] @ (0.299, 0.587, 0.114))
        annotations[(r, c)] = (
            f"{row['median_diff']:+.2f}\n{_stars(row['p_holm'], alpha)}",
            "white" if luminance < 0.5 else "0.15",
        )

    cell_w, cell_h = 0.82, 0.62
    fig, panel = plt.subplots(figsize=(n_cols * cell_w + 4.2, n_rows * cell_h + 2.4 + (0.3 if subtitle else 0)))
    panel.imshow(rgba, interpolation="nearest", aspect="auto")
    panel.set_xticks(range(n_cols), channels, fontsize=8, rotation=35, ha="right")
    panel.set_yticks(range(n_rows), [f"{a} vs {b}" for a, b in comparisons], fontsize=7.5)
    panel.set_xticks(np.arange(n_cols + 1) - 0.5, minor=True)
    panel.set_yticks(np.arange(n_rows + 1) - 0.5, minor=True)
    panel.grid(which="minor", color="white", linewidth=1.2)
    panel.tick_params(which="minor", length=0)
    for (r, c), (text, color) in annotations.items():
        panel.text(c, r, text, ha="center", va="center", fontsize=6.0, color=color)

    bar = fig.colorbar(plt.cm.ScalarMappable(norm=scale, cmap=colormap), ax=panel, fraction=0.025, pad=0.02)
    bar.set_label("effect size (rank-biserial): + = A's values are the larger", fontsize=8)
    bar.ax.tick_params(labelsize=7)

    panel.set_title(f"Per-channel Wilcoxon signed-rank tests — {metric}", fontsize=12, pad=12)
    caption = [subtitle] if subtitle else []
    caption.append(
        "Each cell is one test paired over units on that channel alone, Holm-corrected across the "
        f"channels of its own comparison; the number is the median of (A - B). *** p<0.001, ** p<0.01, * p<{alpha}."
    )
    fig.text(0.005, 0.002, "\n".join(caption), fontsize=7, va="bottom", ha="left", color="0.3")

    fig.savefig(f"{out_base}.png", dpi=180, bbox_inches="tight")
    fig.savefig(f"{out_base}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_base}.png")


def save_wilcoxon_report(table, channel_table, out_dir, legend=None, prefix="wilcoxon", alpha=ALPHA, subtitle=""):
    """Write the tidy CSVs, the condition legend, and up to three figures per metric.

    Each metric gets ``<prefix>_<metric>`` (the comparisons as a table: exact p-values,
    n, breadth, one row per comparison), ``<prefix>_matrix_<metric>`` (the same tests as
    a condition x condition matrix, the shape to read a block of policies from at a
    glance) and, where the metric has a channel breakdown,
    ``<prefix>_per_channel_<metric>`` (comparison x joint / muscle, showing which
    channels carry each effect).

    ``legend`` maps the short condition names used in the figures to their full
    description; it is written next to the CSV and printed under the table figure, so a
    figure stays readable without the multi-line policy labels in every cell.
    """
    if table.empty:
        print("  No Wilcoxon comparisons to save.")
        return
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, f"{prefix}_tests.csv")
    table.to_csv(csv_path, index=False)
    print(f"  Saved {csv_path}")

    has_channels = channel_table is not None and not channel_table.empty
    if has_channels:
        channel_csv = os.path.join(out_dir, f"{prefix}_per_channel_tests.csv")
        channel_table.to_csv(channel_csv, index=False)
        print(f"  Saved {channel_csv}")

    if legend:
        legend_path = os.path.join(out_dir, f"{prefix}_conditions.csv")
        pd.DataFrame(
            {"condition": list(legend), "description": [str(v).replace("\n", " ") for v in legend.values()]}
        ).to_csv(legend_path, index=False)
        print(f"  Saved {legend_path}")

    for metric, block in table.groupby("metric", sort=False):
        slug = _slugify(metric)
        _render_metric_figure(
            metric,
            block,
            os.path.join(out_dir, f"{prefix}_{slug}"),
            legend=legend,
            alpha=alpha,
            subtitle=subtitle,
        )
        _render_metric_matrix(
            metric,
            block,
            os.path.join(out_dir, f"{prefix}_matrix_{slug}"),
            alpha=alpha,
            subtitle=subtitle,
        )
        if has_channels:
            channel_block = channel_table[channel_table["metric"] == metric]
            if not channel_block.empty:
                _render_per_channel_figure(
                    metric,
                    channel_block,
                    os.path.join(out_dir, f"{prefix}_per_channel_{slug}"),
                    alpha=alpha,
                    subtitle=subtitle,
                )
