"""Population and subject-level muscle figures."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from ..constants import (
    BETWEEN_SUBJECT_HATCH,
    BETWEEN_SUBJECT_LABEL,
    WITHIN_SUBJECT_HATCH,
    WITHIN_SUBJECT_LABEL,
)
from .style import draw_grouped_hbar


def _muscle_plot_trace(values, *, normalize):
    """Optionally min-max scale one trace, preserving constant signals."""
    if not normalize:
        return values
    span = np.ptp(values)
    if span == 0:
        return values
    return (values - np.min(values)) / span


def _plot_population_profile(analysis, ax, muscle_idx, *, normalize, signal_label):
    """Draw the human cohort and population policies with one display scaling."""
    for trace in analysis.human.muscle_data[:, muscle_idx, :]:
        ax.plot(np.linspace(0, 100, len(trace)), _muscle_plot_trace(trace, normalize=normalize),
                color="lightgrey", linewidth=2.5, alpha=0.5, zorder=1)
    for run in analysis.policies:
        trace = run.profiles.avg_muscle_gaits[muscle_idx]
        ax.plot(np.linspace(0, 100, len(trace)), _muscle_plot_trace(trace, normalize=normalize),
                color=run.policy.color, linestyle=run.policy.linestyle,
                linewidth=2.5, alpha=0.8, zorder=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if normalize:
        ax.set_ylim(0, 1)
    ax.set_ylabel("Normalized signal" if normalize else signal_label, fontsize=10, labelpad=6)
    if muscle_idx == 0:
        ax.set_title("Normalized profile" if normalize else f"{signal_label} profile", fontsize=11)
    if muscle_idx == len(analysis.channels.muscle_names) - 1:
        ax.set_xlabel("Gait cycle [%]", fontsize=11)
    ax.tick_params(labelsize=9)


def _plot_peak_columns(analysis, fig, gs, muscle_idx, muscle_name, reference,
                       policy_metrics, shared_axes, n_muscles):
    """Two comparable error columns using the correlation panel's groups."""
    for offset, (field, title, label) in enumerate((
        ("muscle_peak_height_error", "Peak height error", "Absolute signal\ndifference"),
        ("muscle_peak_time_error", "Peak time error", "Gait cycle difference\n[percentage points]"),
    )):
        ax = fig.add_subplot(gs[muscle_idx, 2 + offset], sharex=shared_axes[offset])
        if shared_axes[offset] is None:
            shared_axes[offset] = ax
        ref_metrics, ref_label, ref_hatch = reference
        draw_grouped_hbar(
            analysis, ax, getattr(ref_metrics, field).get(muscle_name, []), ref_label,
            [getattr(metrics, field).get(muscle_name, []) for metrics in policy_metrics],
            show_group_labels=False, ref_hatch=ref_hatch,
        )
        ax.set_xlim(left=0)
        if offset == 1:
            ax.set_xlim(0, 50)
        if muscle_idx == 0:
            ax.set_title(title, fontsize=10, fontweight="bold", pad=10)
        if muscle_idx == n_muscles - 1:
            ax.set_xlabel(label, fontsize=10, labelpad=6)


def _plot_aggregate_column(analysis, fig, slot, reference, policy_metrics, ref_corr):
    """Stack three all-muscle metrics independently of the muscle row count."""
    ref_metrics, ref_label, ref_hatch = reference
    rows = slot.subgridspec(3, 1, hspace=0.5)
    for row, (field, title, label) in enumerate((
        ("muscle_corr", "All muscles: correlation", "Pearson Correlation"),
        ("muscle_peak_height_error", "All muscles:\npeak height error", "Absolute signal\ndifference"),
        ("muscle_peak_time_error", "All muscles:\npeak time error", "Gait cycle difference\n[percentage points]"),
    )):
        ax = fig.add_subplot(rows[row, 0])
        ref_samples = ref_corr if row == 0 else getattr(ref_metrics, field).get("all_muscles", [])
        draw_grouped_hbar(
            analysis, ax, ref_samples, ref_label,
            [getattr(metrics, field).get("all_muscles", []) for metrics in policy_metrics],
            show_group_labels=True, ref_hatch=ref_hatch,
        )
        ax.set_title(title, fontsize=10, fontweight="bold", pad=10)
        ax.set_xlabel(label, fontsize=10, labelpad=6)
        ax.tick_params(labelsize=9)
        if row > 0:
            ax.set_xlim(left=0)
        if row == 2:
            ax.set_xlim(0, 50)


def plot_muscle_metrics(analysis):
    """Plot correlation, peak errors, and muscle profiles for the population."""

    signal_label = (
        "Activation" if analysis.config.emg_signal == "activations"
        else "Muscle signal (EMG / policy)"
    )
    muscle_names = analysis.channels.muscle_names
    display_muscle_names = [analysis.muscle_map[m] for m in muscle_names]
    n_muscles = len(muscle_names)

    # Three aggregate panels in the first column; one row per muscle to the right.
    fig = plt.figure(figsize=(23, 2.1 * n_muscles + 1), layout="constrained")
    gs = GridSpec(n_muscles, 6, figure=fig, width_ratios=[1.3, 1, 1, 1, 2, 2], hspace=0.18, wspace=0.15)

    _plot_aggregate_column(
        analysis, fig, gs[:, 0],
        (analysis.human_metrics, BETWEEN_SUBJECT_LABEL, BETWEEN_SUBJECT_HATCH),
        [run.population for run in analysis.policies],
        analysis.human_metrics.pairwise_muscle_corr["all_muscles"],
    )

    # =========================
    # Per-muscle correlation and peak errors, followed by time series
    # =========================

    # Per-muscle bars share one x scale so their lengths stay comparable across muscles.
    ax_bar_ref = None
    peak_axes = [None, None]

    for muscle_idx, muscle_name in enumerate(muscle_names):
        ax_bar = fig.add_subplot(gs[muscle_idx, 1], sharex=ax_bar_ref)
        ax_bar_ref = ax_bar_ref or ax_bar
        draw_grouped_hbar(
            analysis,
            ax_bar,
            analysis.human_metrics.pairwise_muscle_corr[muscle_name],
            BETWEEN_SUBJECT_LABEL,
            [run.population.muscle_corr[muscle_name] for run in analysis.policies],
            show_group_labels=False,
            ref_hatch=BETWEEN_SUBJECT_HATCH,
        )
        ax_bar.set_ylabel(display_muscle_names[muscle_idx], fontsize=10, labelpad=8)
        if muscle_idx == 0:
            ax_bar.set_title("Correlation", fontsize=11, fontweight="bold", pad=10)
        if muscle_idx == n_muscles - 1:
            ax_bar.set_xlabel("Pearson Correlation", fontsize=11)

        _plot_peak_columns(
            analysis, fig, gs, muscle_idx, muscle_name,
            (analysis.human_metrics, BETWEEN_SUBJECT_LABEL, BETWEEN_SUBJECT_HATCH),
            [run.population for run in analysis.policies], peak_axes, n_muscles,
        )
        for column, normalize in ((4, False), (5, True)):
            ax = fig.add_subplot(gs[muscle_idx, column])
            _plot_population_profile(analysis, ax, muscle_idx,
                                     normalize=normalize, signal_label=signal_label)

    plt.savefig(f"{analysis.plots_dir}/human_to_human_muscle_metrics.svg", bbox_inches="tight")
    plt.close(fig)


def plot_subject_based_muscle_metrics(analysis):
    """Plot subject-matched correlation, peak errors, and muscle profiles."""
    if not analysis.subject_based:
        print("Subject-based plotting is disabled. Set subject_based=True to enable.")
        return

    signal_label = (
        "Activation" if analysis.config.emg_signal == "activations"
        else "Muscle signal (EMG / policy)"
    )
    muscle_names = analysis.channels.muscle_names
    display_muscle_names = [analysis.muscle_map[m] for m in muscle_names]
    n_muscles = len(muscle_names)

    # Subject-based reference = within-subject reliability (the subject-matched upper
    # bound); fall back to the between-subject pairwise reference if it is unavailable.
    wref = analysis.references.within_subject
    if wref is not None and wref.n > 0:
        ref_label, ref_hatch = WITHIN_SUBJECT_LABEL, WITHIN_SUBJECT_HATCH
        ref_all = wref.muscle_corr["all_muscles"]
        ref_per = wref.muscle_corr
    else:
        ref_label, ref_hatch = BETWEEN_SUBJECT_LABEL, BETWEEN_SUBJECT_HATCH
        ref_all = analysis.human_metrics.pairwise_muscle_corr["all_muscles"]
        ref_per = analysis.human_metrics.pairwise_muscle_corr

    # Restrict the one-column-per-subject time series to a fixed random subset.
    plot_positions = analysis.human.plot_subject_positions
    n_subjects = len(plot_positions)

    # Three aggregate panels in the first column; per-muscle bars and subject profiles to the right.
    fig = plt.figure(figsize=(20 + 2.4 * n_subjects, 2.1 * n_muscles + 1), layout="constrained")
    # fig.suptitle("Muscle Metrics per Subject")
    gs = GridSpec(
        n_muscles,
        5 + n_subjects,
        figure=fig,
        width_ratios=[3, 3, 3, 3] + [2.2] * n_subjects + [3],
        hspace=0.6,
        wspace=0.45,
    )

    _plot_aggregate_column(
        analysis, fig, gs[:, 0],
        (wref if wref is not None and wref.n > 0 else analysis.human_metrics,
         ref_label, ref_hatch),
        [run.per_subject for run in analysis.policies], ref_all,
    )

    # =========================
    # Per-muscle correlation and peak errors, followed by subject time series
    # =========================
    time_steps = 100  # Standard gait cycle length
    time = np.linspace(0, 100, time_steps)

    # Per-muscle bars share one x scale so their lengths stay comparable across muscles.
    ax_corr_ref = None
    peak_axes = [None, None]

    for muscle_idx, muscle_name in enumerate(muscle_names):
        ax_corr = fig.add_subplot(gs[muscle_idx, 1], sharex=ax_corr_ref)
        ax_corr_ref = ax_corr_ref or ax_corr
        draw_grouped_hbar(
            analysis,
            ax_corr,
            ref_per[muscle_name],
            ref_label,
            [run.per_subject.muscle_corr[muscle_name] for run in analysis.policies],
            show_group_labels=False,
            ref_hatch=ref_hatch,
        )
        # The group names live on the all-muscles panel, so the free y label carries the muscle.
        ax_corr.set_ylabel(f"{display_muscle_names[muscle_idx]}", fontdict={"fontsize": 10})
        if muscle_idx == 0:
            ax_corr.set_title("Correlation", fontsize=11, fontweight="bold", pad=10)
        if muscle_idx == n_muscles - 1:
            ax_corr.set_xlabel("Pearson Correlation", fontsize=11)

        _plot_peak_columns(
            analysis, fig, gs, muscle_idx, muscle_name,
            (wref if wref is not None and wref.n > 0 else analysis.human_metrics,
             ref_label, ref_hatch),
            [run.per_subject for run in analysis.policies], peak_axes, n_muscles,
        )
        for col_idx, subject_idx in enumerate(plot_positions):
            ax = fig.add_subplot(gs[muscle_idx, 4 + col_idx])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            subj_id = analysis.human.subjects[subject_idx]
            if muscle_idx == 0:
                ax.set_title(f"Subj. {subj_id}", fontsize=11)
            if col_idx == 0:
                ax.set_ylabel(signal_label, fontsize=11)
            if muscle_idx == n_muscles - 1:
                ax.set_xlabel("Gait cycle [%]", fontsize=11)
            ax.tick_params(labelsize=9)

            normalized_activation = _muscle_plot_trace(
                analysis.human.muscle_data[subject_idx, muscle_idx, :],
                normalize=False,
            )
            ax.plot(time, normalized_activation, color="grey", linewidth=2.5, alpha=0.5, zorder=1)
            for run in analysis.policies:
                if run.profiles.subject_ids is not None and subj_id in run.profiles.subject_ids:
                    policy_activation = run.profiles.per_subject_muscle[subj_id][muscle_idx]
                    normalized_policy_activation = _muscle_plot_trace(
                        policy_activation, normalize=False,
                    )

                    ax.plot(
                        time,
                        normalized_policy_activation,
                        color=run.policy.color,
                        linestyle=run.policy.linestyle,
                        linewidth=2.5,
                        alpha=0.8,
                        zorder=2,
                    )

        ax_normalized = fig.add_subplot(gs[muscle_idx, 4 + n_subjects])
        _plot_population_profile(analysis, ax_normalized, muscle_idx,
                                 normalize=True, signal_label=signal_label)

    fig.savefig(f"{analysis.plots_dir}/human_to_human_muscle_metrics_per_subject.svg", bbox_inches="tight")
    plt.close(fig)
