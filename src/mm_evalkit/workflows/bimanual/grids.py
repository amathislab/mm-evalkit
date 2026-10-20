"""Channel-by-motion-type grid figures for ULTRA-MoCap.

The per-condition figures put every joint of one movement/condition on one page, so
comparing the same joint across movements means flipping between pages. These lay the
whole matrix out at once: one channel (joint or muscle group) per column, one
movement per row, with that movement's speed conditions averaged together. Two views of that matrix are written -- the average cycles
themselves, and the scores those cycles produce -- because one shows *how* a policy
differs and the other *how much*.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .aggregate_waveforms import _stack_movement_cycles
from .figures import (
    AXIS_LABEL_SIZE,
    FIGURE_TITLE_SIZE,
    LEGEND_SIZE,
    PANEL_TITLE_SIZE,
    SMALL_TEXT_SIZE,
    TICK_SIZE,
    save_figure,
)
from .summary_plots import _draw_aggregated_bars, _grey_references, _summary_axes

# Enough width per channel to read a cycle, and enough height per row for a title.
CELL_WIDTH_IN = 2.5
CELL_HEIGHT_IN = 1.9


def _subjects_by_group(real_records):
    """``{movement: [subject, ...]}`` for the trials every policy kept.

    A subject appears once per movement however many speed conditions of it were kept;
    the speeds are averaged together when the cycles are stacked.
    """
    subjects: dict[str, set[str]] = {}
    for movement, _, subject in real_records:
        subjects.setdefault(movement, set()).add(subject)
    return {group: sorted(found) for group, found in subjects.items()}


def _grid(groups, channel_names, *, row_label_width=1.6):
    """A rows=movement, columns=channel figure with room for the row labels."""
    n_rows, n_cols = len(groups), len(channel_names)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(CELL_WIDTH_IN * n_cols + row_label_width, CELL_HEIGHT_IN * n_rows + 1.6),
        squeeze=False,
    )
    return fig, axes


def _finish_grid(fig, axes, groups, channel_names, handles, labels, suptitle, out_path):
    """Label the outer edges only, add one shared legend, and save both formats."""
    for column, name in enumerate(channel_names):
        axes[0][column].set_title(name, fontsize=PANEL_TITLE_SIZE)
    for row, movement in enumerate(groups):
        axes[row][0].set_ylabel(movement, fontsize=PANEL_TITLE_SIZE, fontweight="bold")
    height = fig.get_figheight()
    fig.suptitle(suptitle, fontsize=FIGURE_TITLE_SIZE, y=1 - 0.25 / height)
    fig.tight_layout(rect=(0, 0, 1, 1 - 1.15 / height))
    if handles:
        fig.legend(
            handles,
            labels,
            fontsize=LEGEND_SIZE,
            ncols=min(len(labels), 6),
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, 1 - 0.55 / height),
        )
    save_figure(fig, out_path, announce="saved")


def plot_waveform_grid(
    policy_records,
    policies,
    real_records,
    channel_names,
    *,
    policy_field,
    real_field,
    ylabel,
    real_label,
    real_on_twin=False,
    twin_ylabel="",
    suptitle,
    out_path,
):
    """One average cycle per (movement, channel) cell, real plus every policy.

    Each subject contributes one average cycle -- its speed conditions averaged together
    -- so the shaded band is the spread *between subjects*, not between repetitions or
    between speeds. ``real_on_twin`` puts the real trace on
    a right-hand axis, which is what the muscle grid needs: policy activation and
    normalized EMG are different units and must not share a scale.
    """
    subjects_by_group = _subjects_by_group(real_records)
    _, groups = _summary_axes(real_records)
    if not groups or not channel_names:
        return

    fig, axes = _grid(groups, channel_names)
    handles, labels = [], []
    for row, group in enumerate(groups):
        subjects = subjects_by_group[group]
        real = _stack_movement_cycles(real_records, group, subjects, real_field)
        series = [
            (
                policy.label,
                policy.color,
                policy.linestyle,
                _stack_movement_cycles(policy_records.get(policy.id, {}), group, subjects, policy_field),
            )
            for policy in policies
        ]
        for column in range(len(channel_names)):
            ax = axes[row][column]
            for label, color, linestyle, values in series:
                if values is None:
                    continue
                mean, std = values
                phase = np.linspace(0, 100, mean.shape[1])
                (line,) = ax.plot(phase, mean[column], color=color, lw=1.5, ls=linestyle, zorder=3, label=label)
                ax.fill_between(
                    phase,
                    mean[column] - std[column],
                    mean[column] + std[column],
                    color=color,
                    alpha=0.12,
                    zorder=2,
                )
                if row == 0 and column == 0:
                    handles.append(line)
                    labels.append(label)
            if real is not None:
                mean, std = real
                target = ax.twinx() if real_on_twin else ax
                phase = np.linspace(0, 100, mean.shape[1])
                (line,) = target.plot(phase, mean[column], color="0.4", lw=1.6, ls="--", zorder=3, label=real_label)
                target.fill_between(
                    phase, mean[column] - std[column], mean[column] + std[column], color="0.4", alpha=0.15, zorder=2
                )
                if real_on_twin:
                    # Every cell carries a twin axis, but only the rightmost column of
                    # the grid needs to spell its scale out.
                    last_column = column == len(channel_names) - 1
                    target.tick_params(axis="y", labelsize=SMALL_TEXT_SIZE, labelcolor="0.4", labelright=last_column)
                    if last_column:
                        target.set_ylabel(twin_ylabel, fontsize=TICK_SIZE, color="0.4")
                if row == 0 and column == 0:
                    handles.append(line)
                    labels.append(real_label)
            ax.tick_params(labelsize=TICK_SIZE)
            ax.spines["top"].set_visible(False)
            # The right spine carries the twin axis when the real trace lives there.
            ax.spines["right"].set_visible(real_on_twin)
            if row == len(groups) - 1:
                ax.set_xlabel("movement cycle (%)", fontsize=AXIS_LABEL_SIZE)
            else:
                ax.tick_params(labelbottom=False)

    # The left y label of each row names the movement, so the unit is named
    # once, in the figure title.
    _finish_grid(fig, axes, groups, channel_names, handles, labels, f"{suptitle} - {ylabel}", out_path)


def plot_score_grid(
    policy_records,
    policies,
    real_records,
    channel_names,
    *,
    key,
    ylabel,
    suptitle,
    out_path,
    zero_line=False,
):
    """One small bar chart per (movement, channel) cell.

    The same bars the aggregated per-channel figures draw -- one per policy, in its
    color and texture, then the grey real-data references -- but laid out so one channel
    can be followed down a column across every movement.
    """
    _, groups = _summary_axes(real_records)
    if not groups or not channel_names:
        return
    references = _grey_references(real_records)

    fig, axes = _grid(groups, channel_names)
    handles, labels = [], []
    for row, group in enumerate(groups):
        for column in range(len(channel_names)):
            ax = axes[row][column]
            _draw_aggregated_bars(
                ax,
                policy_records,
                policies,
                [group],
                key,
                column=column,
                zero_line=zero_line,
                references=references,
            )
            ax.set_xticks([])
            ax.tick_params(labelsize=TICK_SIZE)
            if row == 0 and column == 0:
                handles, labels = ax.get_legend_handles_labels()
    # One scale per row, so a bar's height means the same thing across that row's
    # channels. Shared axes share their tick formatter, so the labels of the columns
    # that repeat the scale are switched off per axes rather than blanked on the
    # formatter -- blanking would empty the first column too.
    for row in range(len(groups)):
        for column in range(1, len(channel_names)):
            axes[row][column].sharey(axes[row][0])
            axes[row][column].tick_params(labelleft=False)
        # The cells were drawn before they were joined, so the row is rescaled here to
        # fit every column's data -- otherwise the whole row keeps the first column's
        # limits and taller bars elsewhere are cut off.
        axes[row][0].autoscale(axis="y")

    _finish_grid(fig, axes, groups, channel_names, handles, labels, f"{suptitle} - {ylabel}", out_path)
