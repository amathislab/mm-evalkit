"""Per-subject and cross-policy aggregate figures for ULTra-MoCap."""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np

from .constants import (
    BETWEEN_SUBJECT_HATCH,
    BETWEEN_SUBJECT_LABEL,
    CONDITION_ORDER,
    MOVEMENT_ORDER,
    REF_GREY,
    RSA_PAIRS_NOTE,
    SUBJECT_COLORS,
    WITHIN_SUBJECT_LABEL,
    WITHIN_SUBJECT_VARIABILITY_HATCH,
    WITHIN_SUBJECT_VARIABILITY_LABEL,
)
from .references import (
    compute_between_subject_reference,
    compute_within_subject_reference,
    compute_within_subject_variability,
)


def _summary_axes(records):
    """Subjects (bar colours) and movement/condition groups (x ticks) of a summary chart."""
    subjects = sorted({s for _, _, s in records})
    groups = sorted(
        {(m, c) for m, c, _ in records},
        key=lambda mc: (
            MOVEMENT_ORDER.index(mc[0]) if mc[0] in MOVEMENT_ORDER else 99,
            CONDITION_ORDER.index(mc[1]) if mc[1] in CONDITION_ORDER else 99,
        ),
    )
    return subjects, groups


def _whiskers(stds):
    """The std array, or None when there is nothing to show (a single value per bar).

    Each trial contributes one score -- computed on its repetition-averaged cycle --
    so a per-subject bar has no spread to draw; matplotlib would otherwise stamp a
    zero-length cap on every bar.
    """
    values = np.asarray(stds, dtype=float)
    return stds if np.any(values > 0) else None


def _draw_grouped_bars(
    ax, records, subjects, groups, key, column=None, zero_line=False, references=None, per_subject_reference=None
):
    """One bar per (movement/condition, subject): that trial's average-cycle score.

    ``column`` selects one joint / EMG channel out of a ``*_by`` metric; with
    ``column=None`` the metric is already a single value per trial.

    Two kinds of reference can be overlaid. ``references`` is a list of
    ``(label, hatch, {(movement, condition): metrics})`` -- each adds one extra grey
    bar per group, told apart by its hatch texture. ``per_subject_reference`` (keyed
    like ``records``) instead draws a grey tick across each subject's own bar -- used
    for the subject-matched ceiling, where the comparison is between a subject's policy
    score and that same subject's own split-half agreement, so it belongs on that
    subject's bar rather than beside the group.
    """
    references = references or []
    x = np.arange(len(groups))
    n_slots = len(subjects) + len(references)
    # The 0.92 factor leaves a small gap between the bars inside a group.
    width = 0.8 / n_slots * 0.92

    def stats(source):
        means, stds = [], []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for entry in source:
                values = entry.get(key) if entry else None
                if values is None or not len(values):
                    means.append(np.nan)
                    stds.append(np.nan)
                    continue
                values = values if column is None else values[:, column]
                means.append(np.nanmean(values))
                stds.append(np.nanstd(values))
        return means, stds

    bars = [
        (subject, SUBJECT_COLORS[slot % len(SUBJECT_COLORS)], None, [records.get((m, c, subject)) for m, c in groups])
        for slot, subject in enumerate(subjects)
    ]
    for label, hatch, source in references:
        bars.append((label, REF_GREY, hatch, [source.get((m, c)) for m, c in groups]))

    tick_labelled = False
    for slot, (label, color, hatch, source) in enumerate(bars):
        offset = (slot - (n_slots - 1) / 2) * (0.8 / n_slots)
        means, stds = stats(source)
        ax.bar(
            x + offset,
            means,
            width,
            yerr=_whiskers(stds),
            label=label,
            color=color,
            hatch=hatch,
            edgecolor="0.45" if hatch else None,
            linewidth=0.6 if hatch else 0,
            error_kw={"elinewidth": 1.0, "capsize": 2.5, "ecolor": "0.32"},
        )

        # Subject-matched reference: a grey tick across that subject's own bar.
        if per_subject_reference is not None and slot < len(subjects):
            subject = subjects[slot]
            values, _ = stats([per_subject_reference.get((m, c, subject)) for m, c in groups])
            for pos, value in zip(x + offset, values, strict=True):
                if not np.isfinite(value):
                    continue
                ax.plot(
                    [pos - width / 2, pos + width / 2],
                    [value, value],
                    color="0.25",
                    lw=1.6,
                    solid_capstyle="butt",
                    zorder=5,
                    label=None if tick_labelled else WITHIN_SUBJECT_LABEL,
                )
                tick_labelled = True
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", color="0.9", lw=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if zero_line:
        ax.axhline(0.0, color="0.45", lw=1.0)


def _finish_summary(fig, axes, groups, suptitle, out_path):
    """Shared legend, x tick labels on the bottom panel only, save."""
    axes[-1].set_xticks(np.arange(len(groups)))
    axes[-1].set_xticklabels([f"{m}\n{c}" for m, c in groups], fontsize=8)
    # Reserve a fixed-height header (in inches, so it holds for a tall per-channel
    # figure too) for the title and, under it, the legend -- inside the panels the
    # legend collides with the tallest bars.
    height = fig.get_figheight()
    fig.suptitle(suptitle, fontsize=12, y=1 - 0.25 / height)
    fig.tight_layout(rect=(0, 0, 1, 1 - 1.05 / height))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        fontsize=8,
        ncols=len(labels),
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1 - 0.5 / height),
    )
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"saved {out_path}")


def plot_summary_bars(records, out_path, arm="right", tag=""):
    """Per-condition, per-subject bar chart of joint tracking error and muscle agreement.

    ``records`` maps ``(movement, condition, subject)`` to the metrics dict returned
    by :func:`trial_metrics`, for **one** policy -- this is the per-subject detail view,
    so with several policies it is written once per policy. Each bar is that trial's
    single score,
    computed on its repetition-averaged cycle, so there is no repetition spread to
    draw as a whisker.

    The grey tick across each bar is that same subject's own split-half agreement in
    the real data (see :func:`compute_within_subject_reference`) -- the subject-matched
    ceiling, which is the right comparison here because each bar is a policy imitating
    that specific subject. (The between-subject reference belongs on the
    subject-aggregated figures instead.)

    Joint error (deg) and muscle correlation (unitless) are different measures, so
    they get a subplot each rather than a shared axis.
    """
    subjects, groups = _summary_axes(records)
    if not subjects or not groups:
        print("  no segmented trials to summarise; skipping the summary bar charts.")
        return
    reference = compute_within_subject_reference(records)
    references = [
        (
            WITHIN_SUBJECT_VARIABILITY_LABEL,
            WITHIN_SUBJECT_VARIABILITY_HATCH,
            compute_within_subject_variability(records),
        )
    ]

    fig, axes = plt.subplots(3, 1, figsize=(max(9, 1.15 * len(groups)), 10.5), sharex=True)
    panels = [
        (axes[0], "joint_rmse", "joint RMSE (deg)", f"Joint tracking error, averaged over the {arm}-arm joints", False),
        (
            axes[1],
            "muscle_corr",
            "correlation (r)",
            "Muscle-group mean activation vs real EMG, averaged over the electrodes",
            True,
        ),
        (
            axes[2],
            "muscle_rsa",
            "RSA (Spearman r)",
            f"Muscle coordination: policy vs real electrode-by-electrode RSM ({RSA_PAIRS_NOTE})",
            True,
        ),
    ]
    for ax, key, ylabel, title, zero_line in panels:
        _draw_grouped_bars(
            ax,
            records,
            subjects,
            groups,
            key,
            zero_line=zero_line,
            references=references,
            per_subject_reference=reference,
        )
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, loc="left")
    _finish_summary(
        fig,
        axes,
        groups,
        "Policy vs real, per movement/condition and subject "
        f"(bar = the score of that trial's repetition-averaged cycle){tag}",
        out_path,
    )


def plot_summary_by_channel(records, key, channel_names, ylabel, suptitle, out_path, zero_line=False):
    """The summary bars broken out per joint / per muscle group -- one panel each.

    Same bars as :func:`plot_summary_bars` (including the grey within-subject
    reference), but instead of averaging the score across joints (or electrodes)
    every one gets its own panel, so a single badly-tracked joint is no longer
    hidden inside the average.
    """
    subjects, groups = _summary_axes(records)
    if not subjects or not groups:
        return
    reference = compute_within_subject_reference(records)
    references = [
        (
            WITHIN_SUBJECT_VARIABILITY_LABEL,
            WITHIN_SUBJECT_VARIABILITY_HATCH,
            compute_within_subject_variability(records),
        )
    ]
    n = len(channel_names)
    fig, axes = plt.subplots(n, 1, figsize=(max(9, 1.15 * len(groups)), 1.7 * n + 1.4), sharex=True, squeeze=False)
    axes = axes[:, 0]
    for idx, name in enumerate(channel_names):
        _draw_grouped_bars(
            axes[idx],
            records,
            subjects,
            groups,
            key,
            column=idx,
            zero_line=zero_line,
            references=references,
            per_subject_reference=reference,
        )
        axes[idx].set_ylabel(ylabel, fontsize=8)
        axes[idx].set_title(name, fontsize=9, loc="left")
    _finish_summary(fig, axes, groups, suptitle, out_path)


def _aggregate_over_subjects(records, groups, key, column=None):
    """Per movement/condition: mean over subjects and the spread *between* subjects.

    Each subject contributes one number -- the score of its own repetition-averaged
    cycle -- so a subject with 50 repetitions does not outweigh one with 5; the
    whisker is the standard deviation across subjects.
    """
    means, stds = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for movement, condition in groups:
            per_subject = []
            for (m, c, _), metrics in records.items():
                if (m, c) != (movement, condition):
                    continue
                values = metrics.get(key)
                if values is None or not len(values):
                    continue
                values = values if column is None else values[:, column]
                per_subject.append(np.nanmean(values))
            means.append(np.nanmean(per_subject) if per_subject else np.nan)
            stds.append(np.nanstd(per_subject) if per_subject else np.nan)
    return means, stds


def _draw_aggregated_bars(ax, policy_records, policies, groups, key, column=None, zero_line=False, references=None):
    """One bar per policy per movement/condition (whisker = between-subject spread).

    This is the cross-policy comparison: each policy's score is first averaged over
    subjects, so every policy contributes one bar per condition. ``references`` adds the
    grey benchmark bars after them, one per ``(label, hatch, {(movement, condition):
    metrics})`` entry: how far apart two *real* subjects are, and how far apart two
    repetitions of the same subject are.
    """
    references = references or []
    x = np.arange(len(groups))
    n_slots = len(policies) + len(references)
    # The 0.92 factor leaves a small gap between the bars inside a group.
    width = 0.8 / n_slots * 0.92

    for slot, policy in enumerate(policies):
        offset = (slot - (n_slots - 1) / 2) * (0.8 / n_slots)
        means, stds = _aggregate_over_subjects(policy_records.get(policy.id, {}), groups, key, column)
        ax.bar(
            x + offset,
            means,
            width,
            yerr=_whiskers(stds),
            label=policy.label,
            color=policy.color,
            error_kw={"elinewidth": 1.0, "capsize": 2.5, "ecolor": "0.32"},
        )

    for slot, (label, hatch, source) in enumerate(references, start=len(policies)):
        offset = (slot - (n_slots - 1) / 2) * (0.8 / n_slots)
        ref_means, ref_stds = [], []
        for movement, condition in groups:
            entry = source.get((movement, condition), {})
            values = entry.get(key)
            if values is None or not len(values):
                ref_means.append(np.nan)
                ref_stds.append(np.nan)
                continue
            values = values if column is None else values[:, column]
            ref_means.append(np.nanmean(values))
            ref_stds.append(np.nanstd(values))
        ax.bar(
            x + offset,
            ref_means,
            width,
            yerr=_whiskers(ref_stds),
            label=label,
            color=REF_GREY,
            hatch=hatch,
            edgecolor="0.45",
            linewidth=0.6,
            error_kw={"elinewidth": 1.0, "capsize": 2.5, "ecolor": "0.32"},
        )

    ax.tick_params(labelsize=8)
    ax.grid(axis="y", color="0.9", lw=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if zero_line:
        ax.axhline(0.0, color="0.45", lw=1.0)


def _grey_references(real_records):
    """The two grey benchmark bars of the aggregated charts, in drawing order.

    Both are computed from real data only, so they are shared by every policy: how far
    apart two repetitions of the same subject are, then how far apart two subjects are.
    """
    return [
        (
            WITHIN_SUBJECT_VARIABILITY_LABEL,
            WITHIN_SUBJECT_VARIABILITY_HATCH,
            compute_within_subject_variability(real_records),
        ),
        (BETWEEN_SUBJECT_LABEL, BETWEEN_SUBJECT_HATCH, compute_between_subject_reference(real_records)),
    ]


def plot_aggregated_bars(policy_records, policies, real_records, out_path, arm="right"):
    """Subject-aggregated, cross-policy version of :func:`plot_summary_bars`."""
    _, groups = _summary_axes(real_records)
    if not groups:
        return
    references = _grey_references(real_records)
    fig, axes = plt.subplots(3, 1, figsize=(max(9, 1.15 * len(groups)), 10.5), sharex=True)
    panels = [
        (axes[0], "joint_rmse", "joint RMSE (deg)", f"Joint tracking error, averaged over the {arm}-arm joints", False),
        (
            axes[1],
            "muscle_corr",
            "correlation (r)",
            "Muscle-group mean activation vs real EMG, averaged over the electrodes",
            True,
        ),
        (
            axes[2],
            "muscle_rsa",
            "RSA (Spearman r)",
            f"Muscle coordination: policy vs real electrode-by-electrode RSM ({RSA_PAIRS_NOTE})",
            True,
        ),
    ]
    for ax, key, ylabel, title, zero_line in panels:
        _draw_aggregated_bars(ax, policy_records, policies, groups, key, zero_line=zero_line, references=references)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, loc="left")
    _finish_summary(
        fig,
        axes,
        groups,
        "Policy vs real, aggregated over subjects (bar = mean over subjects, whisker = between-subject spread)",
        out_path,
    )


def plot_aggregated_by_channel(
    policy_records, policies, real_records, key, channel_names, ylabel, suptitle, out_path, zero_line=False
):
    """Subject-aggregated, cross-policy version of :func:`plot_summary_by_channel`."""
    _, groups = _summary_axes(real_records)
    if not groups:
        return
    references = _grey_references(real_records)
    n = len(channel_names)
    fig, axes = plt.subplots(n, 1, figsize=(max(9, 1.15 * len(groups)), 1.7 * n + 1.4), sharex=True, squeeze=False)
    axes = axes[:, 0]
    for idx, name in enumerate(channel_names):
        _draw_aggregated_bars(
            axes[idx], policy_records, policies, groups, key, column=idx, zero_line=zero_line, references=references
        )
        axes[idx].set_ylabel(ylabel, fontsize=8)
        axes[idx].set_title(name, fontsize=9, loc="left")
    _finish_summary(fig, axes, groups, suptitle, out_path)
