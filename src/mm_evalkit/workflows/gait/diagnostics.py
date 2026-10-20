"""Gait extraction diagnostics and audit tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba

from .plotting.style import policy_fill_alpha


def build_and_save_gait_count_table(analysis):
    """Build and save the table of extracted gait cycles per subject, policy and rollout.

    Rows are subjects. Columns are policies, each split into one sub-column per
    rollout plus a ``total`` sub-column holding that subject's total gait cycles for
    the policy. A rollout is one successive episode of that subject for the policy
    (e.g. the evaluation passes), ordered as stored in the trajectory file. Each cell
    holds the number of gait cycles extracted for that (subject, rollout); it is 0
    when the episode terminated early or produced no valid gait cycle, and blank when
    that policy has no such rollout for the subject.

    The table is built from the raw per-episode extraction, before the
    common-subject restriction, so every subject and rollout is represented. Stored
    on ``analysis.gait_count_table`` and saved as a CSV under ``analysis.plots_dir``. Only
    available for subject-based (Gait120) analyses.
    """
    if not analysis.subject_based:
        print("  Skipping gait-count table: only available for subject-based (Gait120) analyses.")
        return

    # Subjects in subj_list order, plus any extra subjects seen in the extraction.
    # (subj_list is still the full pre-restriction set at this point.)
    subjects = [int(s) for s in analysis.human.subjects]
    for run in analysis.policies:
        for rec in run.episode_gait_records:
            if rec["subject_id"] is not None and int(rec["subject_id"]) not in subjects:
                subjects.append(int(rec["subject_id"]))

    columns = []
    data = {}  # numeric gait counts, for the CSV
    detail = {}  # per-cell display text (count + gait start-end spans), for the PNG
    reason = {}  # per-cell termination reason, for highlighting in the PNG
    # Parallel display metadata (one entry per column) for the rendered PNG.
    display_cols = []  # (short_policy_label, rollout_label)
    col_colors = []  # policy plot color for each column's header
    col_alphas = []  # header fill alpha for each column (distinguishes same-colored policies)
    col_is_total = []  # whether the column is a policy total
    subject_traj_len = {}  # subject -> reference trajectory length (timesteps in trial 5)
    for policy_idx, run in enumerate(analysis.policies):
        policy = run.policy
        policy_key = policy.id
        short_label = f"P{policy_idx + 1}"
        color = policy.color
        # Same fill alpha the bar plots / metrics table use, so two policies sharing a
        # color (finetuned vs from-scratch) stay distinguishable in the header tint.
        alpha = policy_fill_alpha(policy.linestyle)
        # Group this policy's per-episode records by subject, in file order = rollouts.
        per_subject_rollouts = {s: [] for s in subjects}
        for rec in run.episode_gait_records:
            if rec["subject_id"] is None:
                continue
            subj = int(rec["subject_id"])
            per_subject_rollouts.setdefault(subj, []).append(
                (int(rec["n_gaits"]), rec.get("gait_indexes", []), rec.get("reason", "ok"))
            )
            ref_len = rec.get("reference_length")
            if ref_len is not None:
                subject_traj_len[subj] = max(subject_traj_len.get(subj, 0), int(ref_len))

        max_rollouts = max((len(v) for v in per_subject_rollouts.values()), default=0)
        for r in range(max_rollouts):
            col = (policy_key, f"rollout_{r}")
            columns.append(col)
            counts, details, reasons = [], [], []
            for s in subjects:
                rolls = per_subject_rollouts[s]
                if r < len(rolls):
                    n_gaits, gait_indexes, cell_reason = rolls[r]
                    counts.append(n_gaits)
                    details.append(format_gait_cell(n_gaits, gait_indexes))
                    reasons.append(cell_reason)
                else:
                    counts.append(np.nan)
                    details.append("")
                    reasons.append("")
            data[col] = counts
            detail[col] = details
            reason[col] = reasons
            display_cols.append((short_label, f"r{r}"))
            col_colors.append(color)
            col_alphas.append(alpha)
            col_is_total.append(False)
        total_col = (policy_key, "total")
        columns.append(total_col)
        totals = [int(sum(n for n, _, _ in per_subject_rollouts[s])) for s in subjects]
        data[total_col] = totals
        detail[total_col] = [str(t) for t in totals]
        reason[total_col] = ["" for _ in subjects]  # totals aggregate rollouts; not highlighted
        display_cols.append((short_label, "total"))
        col_colors.append(color)
        col_alphas.append(alpha)
        col_is_total.append(True)

    df = pd.DataFrame(data, index=subjects)
    df.index.name = "subject"
    df.columns = pd.MultiIndex.from_tuples(columns, names=["policy", "rollout"])
    analysis.gait_count_table = df

    csv_path = f"{analysis.plots_dir}/gait_counts_per_subject_rollout.csv"
    df.to_csv(csv_path)

    row_labels = [str(s) for s in subjects]

    # Two leading info columns (plain, not tied to any policy): the reference
    # trajectory length (timesteps in trial 5) and the number of gait cycles we
    # expect from it (~1 cycle per 100 timesteps, so ~1 for a ~100-length motion
    # and ~2 for a ~200-length motion).
    info_display_cols = [("traj", "len"), ("exp", "gaits")]
    info_col_colors = ["lightgrey", "lightgrey"]
    info_col_alphas = [0.55, 0.55]  # neutral header tint for the plain info columns
    info_col_is_total = [False, False]

    cell_text = []
    cell_reasons = []
    for i, s in enumerate(subjects):
        traj_len = subject_traj_len.get(s)
        if traj_len:
            expected_gaits = str(max(1, round(traj_len / 100)))
            info_cells = [str(traj_len), expected_gaits]
        else:
            info_cells = ["?", "?"]
        cell_text.append(info_cells + [detail[col][i] for col in columns])
        # Info columns are never highlighted.
        cell_reasons.append(["", ""] + [reason[col][i] for col in columns])

    display_cols = info_display_cols + display_cols
    col_colors = info_col_colors + col_colors
    col_alphas = info_col_alphas + col_alphas
    col_is_total = info_col_is_total + col_is_total

    png_path = f"{analysis.plots_dir}/gait_counts_per_subject_rollout.png"
    render_gait_count_table_png(
        analysis, cell_text, cell_reasons, row_labels, display_cols, col_colors, col_alphas, col_is_total, png_path
    )
    print(f"  Saved gait-count table to {csv_path} and {png_path}")

    save_event_sequences_txt(analysis)


def save_event_sequences_txt(analysis):
    """Write a text file describing every rollout's gait-event sequence and final count.

    One block per rollout (episode), grouped by policy, listing the ordered gait
    events (heel-strikes / toe-offs, ``RHS@f`` = right heel-strike at frame f), the
    compressed feet-contact phase sequence, the episode/reference lengths, the number
    of gait cycles we expect (~1 per 100 timesteps), the termination status, and the
    number of gait cycles finally extracted (with their start-end spans). Saved as
    ``gait_event_sequences.txt`` under ``analysis.plots_dir``.
    """
    txt_path = Path(analysis.plots_dir) / "gait_event_sequences.txt"
    lines = []
    for policy_idx, run in enumerate(analysis.policies):
        policy = run.policy
        label = " ".join(policy.label.split())
        traj_path = policy.trajectory_path
        records = run.episode_gait_records
        lines.append("=" * 100)
        lines.append(f"Policy P{policy_idx + 1}: {label}")
        lines.append(f"  trajectory: {traj_path}")
        lines.append(f"  rollouts: {len(records)}")
        lines.append("=" * 100)
        if not records:
            lines.append("  (no rollouts — trajectory data missing)")
            lines.append("")
            continue

        rollout_counter = {}  # subject_id -> next rollout index for that subject
        for rec in records:
            subj = rec.get("subject_id")
            subj_key = int(subj) if subj is not None else None
            rollout_idx = rollout_counter.get(subj_key, 0)
            rollout_counter[subj_key] = rollout_idx + 1

            ref_len = rec.get("reference_length")
            expected = max(1, round(ref_len / 100)) if ref_len else "?"
            spans = " ".join(f"({s}-{e})" for s, e in rec.get("gait_indexes", []))
            subj_str = subj_key if subj_key is not None else "n/a"

            lines.append(
                f"[episode {rec.get('episode', '?')}] subject={subj_str} "
                f"rollout={rollout_idx} traj_id={rec.get('traj_id', '?')}"
            )
            lines.append(
                f"  episode_length={rec.get('episode_length', '?')}  "
                f"reference_length={ref_len}  expected_gaits={expected}"
            )
            lines.append(f"  status: {rec.get('reason', '?')}")
            lines.append(f"  events: {rec.get('event_seq', '(n/a)')}")
            lines.append(f"  phases: {rec.get('phase_seq', '(n/a)')}")
            final_line = f"  final_gaits: {rec.get('n_gaits', 0)}"
            if spans:
                final_line += f"  spans: {spans}"
            lines.append(final_line)
            lines.append("")

    txt_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved per-rollout event sequences to {txt_path}")


def format_gait_cell(n_gaits, gait_indexes):
    """Cell text: the gait count, then each extracted gait's (start-end) timesteps in brackets."""
    if n_gaits == 0 or not gait_indexes:
        return str(int(n_gaits))
    spans = "\n".join(f"({start}-{end})" for start, end in gait_indexes)
    return f"{int(n_gaits)}\n{spans}"


def render_gait_count_table_png(
    analysis, cell_text, cell_reasons, row_labels, display_cols, col_colors, col_alphas, col_is_total, png_path
):
    """Render the per-subject / per-policy / per-rollout gait-count table as a PNG.

    Each cell shows the gait count and, in brackets below it, the (start-end)
    timesteps of every extracted gait; each subject row label carries the reference
    trajectory length (timesteps in trial 5) in brackets. Header cells are tinted with
    each policy's plot color at the same fill alpha the bar plots use (so two policies
    sharing a color stay distinguishable); the P1, P2, ... short labels are expanded in
    a legend below the table, and total columns are drawn a little bolder.

    Rollout cells whose episode hit early termination are highlighted in light
    yellow (``cell_reasons``, same shape as ``cell_text``). Other zero-gait cells
    (ran to completion but produced no valid gait cycle) are not highlighted, and
    total columns are never highlighted.
    """
    # Fill color for early-terminated episodes; key matches episode_gait_records["reason"].
    reason_facecolor = {
        "early_termination": to_rgba("yellow", 0.35),
    }
    n_rows, n_cols = len(row_labels), len(display_cols)
    # Tallest cell (most gait spans) drives the row height so nothing is clipped.
    max_lines = max((text.count("\n") + 1 for row in cell_text for text in row), default=1)

    col_labels = [f"{short}\n{rollout}" for short, rollout in display_cols]

    fig_w = max(8.0, 0.6 * n_cols + 2.0)
    fig_h = max(3.0, 0.24 * n_rows * max_lines + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title("Extracted gait cycles per subject (traj length) / policy / rollout", fontsize=11, pad=12)

    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        rowLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6)
    table.auto_set_column_width(range(n_cols))
    table.scale(1, max(1.3, 0.85 * max_lines))

    for (row, col), cell in table.get_celld().items():
        if row == 0 and col >= 0:  # column header row
            # Per-policy fill alpha (same as the bar plots) so same-colored policies differ.
            cell.set_facecolor(to_rgba(col_colors[col], col_alphas[col]))
            cell.set_text_props(fontweight="bold", color="black")
        elif col >= 0 and col_is_total[col]:  # policy total columns
            # A faint, proportionally scaled tint of the same alpha for the body cells.
            cell.set_facecolor(to_rgba(col_colors[col], col_alphas[col] * 0.2))
            if row > 0:
                cell.set_text_props(fontweight="bold")
        # Highlight zero-gait rollout cells by cause (row-1 indexes into cell_reasons).
        if row > 0 and col >= 0:
            cell_reason = cell_reasons[row - 1][col]
            if cell_reason in reason_facecolor:
                cell.set_facecolor(reason_facecolor[cell_reason])
        if col == -1 and row > 0:  # subject row labels
            cell.set_text_props(fontweight="bold")

    # Legend expanding the P1, P2, ... short labels to the full policy descriptions.
    legend_lines = [f"P{i + 1}: {run.policy.id} ({analysis.config.emg_signal}) — {' '.join(run.policy.label.split())}" for i, run in enumerate(analysis.policies)]
    colors = [run.policy.color for run in analysis.policies]
    for i, (line, color) in enumerate(zip(legend_lines, colors, strict=True)):
        fig.text(0.01, -0.02 - 0.035 * i, line, fontsize=7, color=color, va="top", ha="left")

    # Highlight legend: what the yellow cells mean.
    y0 = -0.02 - 0.035 * len(legend_lines)
    fig.text(
        0.01,
        y0,
        "yellow cell: episode hit early termination",
        fontsize=7,
        color="black",
        va="top",
        ha="left",
        bbox={"facecolor": to_rgba("yellow", 0.35), "edgecolor": "none", "pad": 1.5},
    )

    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
