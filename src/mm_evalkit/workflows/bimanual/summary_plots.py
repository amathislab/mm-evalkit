"""Per-subject and cross-policy aggregate figures for ULTRA-MoCap."""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch

from mm_evalkit.plotting import policy_bar_style

from .constants import (
    BETWEEN_SUBJECT_HATCH,
    BETWEEN_SUBJECT_LABEL,
    MOVEMENT_ORDER,
    REF_GREY,
    REP_TO_REP_KEYS,
    RSA_PAIRS_NOTE,
    SUBJECT_COLORS,
    WITHIN_SUBJECT_VARIABILITY_HATCH,
    WITHIN_SUBJECT_VARIABILITY_LABEL,
)
from .figures import (
    AXIS_LABEL_SIZE,
    FIGURE_TITLE_SIZE,
    LEGEND_SIZE,
    PANEL_TITLE_SIZE,
    TICK_SIZE,
    full_page,
    save_figure,
)
from .references import (
    compute_between_subject_reference,
    compute_within_subject_variability,
)


def _summary_axes(records):
    """Subjects (bar colours) and movement groups (x ticks) of a summary chart.

    A group is a *movement*, not a movement/condition: the speed conditions of one
    movement (Slow / Normal / Fast / VeryFast, or the 90 / 180 / Max of a reach) are
    averaged together, so ArmSwing is one group however many speeds were recorded. The
    cycles are time-normalized to 0-100%, so averaging across speeds compares the shape
    of the movement rather than its duration.
    """
    subjects = sorted({s for _, _, s in records})
    groups = sorted(
        {m for m, _, _ in records},
        key=lambda m: MOVEMENT_ORDER.index(m) if m in MOVEMENT_ORDER else 99,
    )
    return subjects, groups


def _group_trials(records, group, subject=None):
    """Every trial of one movement, across its speed conditions.

    With ``subject`` given, only that subject's trials of the movement. Returns the
    metrics dicts themselves, since a group no longer maps to a single record key.
    """
    return [
        metrics
        for (movement, _, trial_subject), metrics in records.items()
        if movement == group and (subject is None or trial_subject == subject)
    ]


def _whiskers(stds):
    """The std array, or None when there is nothing to show (a single value per bar).

    Each trial contributes one score -- computed on its repetition-averaged cycle --
    so a per-subject bar has no spread to draw; matplotlib would otherwise stamp a
    zero-length cap on every bar.
    """
    values = np.asarray(stds, dtype=float)
    return stds if np.any(values > 0) else None


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


def _group_scores(source, groups, key, per_trial):
    """One score per movement, so a movement is one sample whatever its size.

    ``per_trial`` says how ``source`` is keyed: True for a records dict keyed by
    ``(movement, condition, subject)`` (each movement is first averaged over its speed
    conditions and then over the subjects that have it, which is what the aggregated bar
    charts draw), False for a reference already keyed by movement.
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


def reference_records(real_records, reference_keys):
    """A records-shaped view of one real-data reference, keyed like a policy's records.

    The repetition-to-repetition variability is scored per trial, just under a different
    key, so re-keying it lets both the table and the summary bars read it through the same
    helpers the policy rows use.
    """
    return {
        key: {name: metrics[source] for name, source in reference_keys.items() if source in metrics}
        for key, metrics in real_records.items()
    }


def _draw_grouped_bars(ax, records, subjects, groups, key, column=None, zero_line=False, references=None):
    """One bar per (movement, subject): that subject's score for the movement.

    Each of the subject's speed conditions is scored on its own repetition-averaged
    cycle and those scores are averaged, so the bar is the movement and the whisker is
    the spread across the speeds behind it.

    ``column`` selects one joint / EMG channel out of a ``*_by`` metric; with
    ``column=None`` the metric is already a single value per trial.

    ``references`` is a list of ``(label, hatch, {(movement, condition): metrics})``
    -- each adds one extra grey bar per group, told apart by its hatch texture.
    """
    references = references or []
    x = np.arange(len(groups))
    n_slots = len(subjects) + len(references)
    # The 0.92 factor leaves a small gap between the bars inside a group.
    width = 0.8 / n_slots * 0.92

    def stats(source):
        """Mean and spread of one bar per group; ``source`` is one trial list per group."""
        means, stds = [], []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for trials in source:
                scores = []
                for metrics in trials:
                    values = metrics.get(key)
                    if values is None or not len(values):
                        continue
                    values = values if column is None else values[:, column]
                    scores.append(np.nanmean(values))
                means.append(np.nanmean(scores) if scores else np.nan)
                stds.append(np.nanstd(scores) if scores else np.nan)
        return means, stds

    bars = [
        (
            subject,
            SUBJECT_COLORS[slot % len(SUBJECT_COLORS)],
            None,
            [_group_trials(records, group, subject) for group in groups],
        )
        for slot, subject in enumerate(subjects)
    ]
    for label, hatch, source in references:
        # A reference is already one entry per movement, so it contributes a one-trial list.
        bars.append((label, REF_GREY, hatch, [[source[group]] if group in source else [] for group in groups]))

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
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(axis="y", color="0.9", lw=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if zero_line:
        ax.axhline(0.0, color="0.45", lw=1.0)


def _finish_summary(fig, axes, groups, suptitle, out_path):
    """Shared legend, x tick labels on the bottom panel only, save."""
    axes[-1].set_xticks(np.arange(len(groups)))
    axes[-1].set_xticklabels(list(groups), fontsize=TICK_SIZE)
    # Reserve a fixed-height header (in inches, so it holds for a tall per-channel
    # figure too) for the title and, under it, the legend -- inside the panels the
    # legend collides with the tallest bars.
    height = fig.get_figheight()
    fig.suptitle(suptitle, fontsize=FIGURE_TITLE_SIZE, y=1 - 0.25 / height)
    fig.tight_layout(rect=(0, 0, 1, 1 - 1.15 / height))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        fontsize=LEGEND_SIZE,
        ncols=len(labels),
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1 - 0.55 / height),
    )
    save_figure(fig, out_path, announce="saved")


def plot_summary_bars(records, out_path, arm="right", tag=""):
    """Per-condition, per-subject bar chart of joint tracking error and muscle agreement.

    ``records`` maps ``(movement, condition, subject)`` to the metrics dict returned
    by :func:`trial_metrics`, for **one** policy -- this is the per-subject detail view,
    so with several policies it is written once per policy. Each bar is that trial's
    single score,
    computed on its repetition-averaged cycle, so there is no repetition spread to
    draw as a whisker.

    Joint error (deg) and muscle correlation (unitless) are different measures, so
    they get a subplot each rather than a shared axis.
    """
    subjects, groups = _summary_axes(records)
    if not subjects or not groups:
        print("  no segmented trials to summarise; skipping the summary bar charts.")
        return
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
            "Muscle-group mean signal vs real EMG, averaged over the electrodes",
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
        )
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_SIZE)
        ax.set_title(title, fontsize=PANEL_TITLE_SIZE, loc="left")
    _finish_summary(
        fig,
        axes,
        groups,
        f"Policy vs real, per movement and subject (bar = the score of that trial's repetition-averaged cycle){tag}",
        out_path,
    )


def plot_summary_by_channel(records, key, channel_names, ylabel, suptitle, out_path, zero_line=False):
    """The summary bars broken out per joint / per muscle group -- one panel each.

    Same bars as :func:`plot_summary_bars`, but instead of averaging the score across joints (or electrodes)
    every one gets its own panel, so a single badly-tracked joint is no longer
    hidden inside the average.
    """
    subjects, groups = _summary_axes(records)
    if not subjects or not groups:
        return
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
        )
        axes[idx].set_ylabel(ylabel, fontsize=AXIS_LABEL_SIZE)
        axes[idx].set_title(name, fontsize=PANEL_TITLE_SIZE, loc="left")
    _finish_summary(fig, axes, groups, suptitle, out_path)


def _aggregate_over_subjects(records, groups, key, column=None):
    """Per movement: mean over subjects and the spread *between* subjects.

    Each subject contributes one number -- its speed conditions each scored on their own
    repetition-averaged cycle, then averaged -- so a subject who performed a movement at
    three speeds does not outweigh one who performed it at a single speed, just as a
    subject with 50 repetitions does not outweigh one with 5. The whisker is the standard
    deviation across subjects.
    """
    means, stds = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for group in groups:
            per_subject = []
            for subject in sorted({s for m, _, s in records if m == group}):
                scores = []
                for metrics in _group_trials(records, group, subject):
                    values = metrics.get(key)
                    if values is None or not len(values):
                        continue
                    values = values if column is None else values[:, column]
                    scores.append(np.nanmean(values))
                if scores:
                    per_subject.append(np.nanmean(scores))
            means.append(np.nanmean(per_subject) if per_subject else np.nan)
            stds.append(np.nanstd(per_subject) if per_subject else np.nan)
    return means, stds


def _draw_aggregated_bars(ax, policy_records, policies, groups, key, column=None, zero_line=False, references=None):
    """One bar per policy per movement (whisker = between-subject spread).

    This is the cross-policy comparison: each policy's score is first averaged over the
    speed conditions of a movement and then over subjects, so every policy contributes one
    bar per movement. ``references`` adds the grey benchmark bars after them, one per
    ``(label, hatch, {movement: metrics})`` entry: how far apart two *real* subjects are, and how far apart two
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
        # Policies are identified by color and told apart by texture: two policies of the
        # same family share a color and differ only in linestyle, which sets the edge
        # style and the fill alpha (see :mod:`mm_evalkit.plotting`).
        ax.bar(
            x + offset,
            means,
            width,
            yerr=_whiskers(stds),
            label=policy.label,
            **policy_bar_style(policy),
            error_kw={"elinewidth": 1.0, "capsize": 2.5, "ecolor": "0.32"},
        )

    for slot, (label, hatch, source) in enumerate(references, start=len(policies)):
        offset = (slot - (n_slots - 1) / 2) * (0.8 / n_slots)
        ref_means, ref_stds = [], []
        for group in groups:
            entry = source.get(group, {})
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

    ax.tick_params(labelsize=TICK_SIZE)
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
            "Muscle-group mean signal vs real EMG, averaged over the electrodes",
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
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_SIZE)
        ax.set_title(title, fontsize=PANEL_TITLE_SIZE, loc="left")
    _finish_summary(
        fig,
        axes,
        groups,
        "Policy vs real, aggregated over subjects (bar = mean over subjects, whisker = between-subject spread)",
        out_path,
    )


def _mean_ci(values) -> tuple[float, float]:
    """Return a finite-sample mean and normal-approximation 95% CI half-width."""
    samples = np.asarray(values, dtype=float).ravel()
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return np.nan, np.nan
    if samples.size == 1:
        return float(samples[0]), 0.0
    return float(np.mean(samples)), float(1.96 * np.std(samples, ddof=1) / np.sqrt(samples.size))


def plot_metrics_summary_bars(policy_records, policies, real_records, out_path, arm="right"):
    """The compact cross-policy summary, in the layout the full-body report uses.

    One panel per metric; within a panel one pair of bars per policy, drawn in that
    policy's color and texture. The pair is the same two aggregation levels the metrics
    table splits every metric into, so a bar here and a cell there are the same number:

    * **filled** -- one sample per trial (subject x movement x condition), scored on its
      repetition-averaged cycle.
    * **outline** -- one sample per movement, each first averaged over its speed
      conditions and then over the subjects that recorded it, so a movement with three
      subjects does not outweigh one with two.

    Bars are mean +- 95% CI over those samples. The grey horizontal lines are the real-data
    references with their own CI band: how far apart two subjects are, and how far apart two
    single repetitions of one subject are. The between-subject reference is defined between
    *pairs of subjects* at the same speed and so has no per-trial form; its line is the
    per-movement sample.
    """
    _, groups = _summary_axes(real_records)
    if not groups or not policies:
        print("  no segmented trials to summarise; skipping the metrics summary bars.")
        return

    rep_to_rep = reference_records(real_records, REP_TO_REP_KEYS)
    between = compute_between_subject_reference(real_records)
    metrics = [
        {
            "title": f"Joint tracking error ({arm} arm)",
            "unit": "RMSE [deg]",
            "key": "joint_rmse",
            "zero_line": False,
        },
        {
            "title": "Muscle-EMG correlation",
            "unit": "correlation (r)",
            "key": "muscle_corr",
            "zero_line": True,
        },
        {
            "title": "Muscle RSA",
            "unit": f"RSA (Spearman r; {RSA_PAIRS_NOTE})",
            "key": "muscle_rsa",
            "zero_line": True,
        },
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(6.0 * len(metrics), 9), squeeze=False)
    # The scope swatches are grey: the bars encode the policy in their color, so a
    # policy-colored legend key would read as belonging to one specific policy.
    handles: dict[str, object] = {
        "Per-trial": Patch(facecolor=to_rgba("0.35", 0.8), edgecolor="0.35", linewidth=1.5),
        "Per-movement": Patch(facecolor="none", edgecolor="0.35", linewidth=1.5),
    }
    positions = np.arange(len(policies))
    width = 0.36
    for metric_index, metric in enumerate(metrics):
        ax = axes.flat[metric_index]
        key = metric["key"]
        for policy_index, policy in enumerate(policies):
            records = policy_records.get(policy.id, {})
            trial_mean, trial_ci = _mean_ci(_trial_scores(records, key))
            group_mean, group_ci = _mean_ci(_group_scores(records, groups, key, per_trial=True))
            style = policy_bar_style(policy)
            ax.bar(
                positions[policy_index] - width / 2,
                trial_mean,
                width,
                yerr=trial_ci,
                capsize=4,
                label="Per-trial",
                **style,
            )
            # The second scope is the same policy drawn as an outline, so the pair reads
            # as one policy at two aggregation levels rather than as two policies.
            ax.bar(
                positions[policy_index] + width / 2,
                group_mean,
                width,
                yerr=group_ci,
                capsize=4,
                label="Per-movement",
                **{**style, "color": "none"},
            )

        for label, samples, style in (
            (
                BETWEEN_SUBJECT_LABEL.replace("\n", " "),
                _group_scores(between, groups, key, per_trial=False),
                {"color": "0.35", "linestyle": "--"},
            ),
            (
                WITHIN_SUBJECT_VARIABILITY_LABEL.replace("\n", " "),
                _trial_scores(rep_to_rep, key),
                {"color": "0.15", "linestyle": "-."},
            ),
        ):
            mean, ci = _mean_ci(samples)
            if not np.isfinite(mean):
                continue
            line = ax.axhline(mean, linewidth=1.6, label=label, **style)
            ax.axhspan(mean - ci, mean + ci, color=style["color"], alpha=0.10)
            handles.setdefault(label, line)

        if metric["zero_line"]:
            ax.axhline(0.0, color="0.45", lw=1.0, zorder=1)
        ax.set_title(metric["title"], fontsize=full_page(PANEL_TITLE_SIZE))
        ax.set_ylabel(metric["unit"], fontsize=full_page(AXIS_LABEL_SIZE))
        ax.set_xticks(positions, [f"P{index + 1}" for index in positions], fontsize=full_page(TICK_SIZE))
        ax.tick_params(labelsize=full_page(TICK_SIZE))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25, zorder=0)

    fig.legend(
        handles.values(),
        handles.keys(),
        loc="upper center",
        ncol=len(handles),
        frameon=False,
        fontsize=full_page(LEGEND_SIZE),
    )
    fig.suptitle(
        "Aggregate ULTRA-MoCap metrics (bars: mean ± 95% CI; P1.."
        f"P{len(policies)} = " + ", ".join(policy.label.replace(chr(10), " ") for policy in policies) + ")",
        fontsize=full_page(FIGURE_TITLE_SIZE),
        y=0.94,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(fig, out_path, dpi=200, announce="saved")


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
        axes[idx].set_ylabel(ylabel, fontsize=AXIS_LABEL_SIZE)
        axes[idx].set_title(name, fontsize=PANEL_TITLE_SIZE, loc="left")
    _finish_summary(fig, axes, groups, suptitle, out_path)
