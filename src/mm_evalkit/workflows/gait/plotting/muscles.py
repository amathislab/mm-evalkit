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


def plot_muscle_metrics(analysis):
    """Generate plots for muscle metrics (activation time series and correlation)."""

    muscle_names = analysis.channels.muscle_names
    display_muscle_names = [analysis.muscle_map[m] for m in muscle_names]
    n_muscles = len(muscle_names)

    # Layout: the all-muscles correlation spans the full height of a wide first column (it
    # doubles as the legend), then one row per muscle with its bars beside its time series.
    fig = plt.figure(figsize=(14, 1.9 * n_muscles + 1))
    gs = GridSpec(n_muscles, 3, figure=fig, width_ratios=[1, 1, 2], hspace=0.6, wspace=0.3)

    # =========================
    # All-Muscles Correlation Plot (First Column)
    # =========================
    ax_all = fig.add_subplot(gs[:, 0])
    draw_grouped_hbar(
        analysis,
        ax_all,
        analysis.human_metrics.pairwise_muscle_corr["all_muscles"],
        BETWEEN_SUBJECT_LABEL,
        [run.population.muscle_corr["all_muscles"] for run in analysis.policies],
        show_group_labels=True,
        ref_hatch=BETWEEN_SUBJECT_HATCH,
    )
    ax_all.set_xlabel("Pearson Correlation")
    ax_all.set_title("All muscles")

    # =========================
    # Per-Muscle Correlation Plots (Second Column) and Time Series (Third Column)
    # =========================
    time_steps = 100  # Standard gait cycle length
    time = np.linspace(0, 100, time_steps)

    # Per-muscle bars share one x scale so their lengths stay comparable across muscles.
    ax_bar_ref = None

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
        ax_bar.set_title(f"{display_muscle_names[muscle_idx]}", fontdict={"fontsize": 9})
        if muscle_idx == n_muscles - 1:
            ax_bar.set_xlabel("Pearson Correlation")

        ax = fig.add_subplot(gs[muscle_idx, 2])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Plot average human muscle activation
        for subject_idx in range(analysis.human.muscle_data.shape[0]):
            if np.std(analysis.human.muscle_data[subject_idx, muscle_idx, :]) == 0:
                normalized_activation = analysis.human.muscle_data[subject_idx, muscle_idx, :]
            else:
                normalized_activation = (
                    analysis.human.muscle_data[subject_idx, muscle_idx, :]
                    - np.min(analysis.human.muscle_data[subject_idx, muscle_idx, :])
                ) / (
                    np.max(analysis.human.muscle_data[subject_idx, muscle_idx, :])
                    - np.min(analysis.human.muscle_data[subject_idx, muscle_idx, :])
                )
            ax.plot(time, normalized_activation, color="lightgrey", linewidth=2.5, alpha=0.5, zorder=1)

        # Plot each policy's muscle activation
        for run in analysis.policies:
            policy_activation = run.profiles.avg_muscle_gaits[muscle_idx]
            normalized_policy_activation = (
                (policy_activation - np.min(policy_activation))
                / (np.max(policy_activation) - np.min(policy_activation))
                if np.std(policy_activation) > 0
                else policy_activation
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

        ax.set_ylabel("Activation")
        if muscle_idx == n_muscles - 1:
            ax.set_xlabel("Gait cycle [%]")

    plt.savefig(f"{analysis.plots_dir}/human_to_human_muscle_metrics.svg", bbox_inches="tight")
    plt.close(fig)


def plot_subject_based_muscle_metrics(analysis):
    """Generate subject-based plots for muscle metrics (activation time series and correlation)."""
    if not analysis.subject_based:
        print("Subject-based plotting is disabled. Set subject_based=True to enable.")
        return

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

    # Layout: the all-muscles bars span the full height of a wide first column (they double as
    # the legend), then one row per muscle holding its correlation bars followed by one time
    # series per subject. The width ratios keep the first column at 1/4 of the figure.
    fig = plt.figure(figsize=(10 + 2.2 * n_subjects, 1.9 * n_muscles + 1))
    # fig.suptitle("Muscle Metrics per Subject")
    gs = GridSpec(
        n_muscles,
        2 + n_subjects,
        figure=fig,
        width_ratios=[3, 3] + [2.2] * n_subjects,
        hspace=0.6,
        wspace=0.45,
    )

    # =========================
    # All-Muscles Correlation Plot (First Column)
    # =========================
    ax_all = fig.add_subplot(gs[:, 0])
    draw_grouped_hbar(
        analysis,
        ax_all,
        ref_all,
        ref_label,
        [run.per_subject.muscle_corr["all_muscles"] for run in analysis.policies],
        show_group_labels=True,
        ref_hatch=ref_hatch,
    )
    ax_all.set_xlabel("Pearson Correlation")
    ax_all.set_title("All muscles")

    # =========================
    # Per-Muscle Correlation Plots (Second Column) and Time Series (Remaining Columns)
    # =========================
    time_steps = 100  # Standard gait cycle length
    time = np.linspace(0, 100, time_steps)

    # Per-muscle bars share one x scale so their lengths stay comparable across muscles.
    ax_corr_ref = None

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
        ax_corr.set_ylabel(f"{display_muscle_names[muscle_idx]}", fontdict={"fontsize": 9})
        if muscle_idx == 0:
            ax_corr.set_title("Correlation")
        if muscle_idx == n_muscles - 1:
            ax_corr.set_xlabel("Pearson Correlation")

        for col_idx, subject_idx in enumerate(plot_positions):
            ax = fig.add_subplot(gs[muscle_idx, 2 + col_idx])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            subj_id = analysis.human.subjects[subject_idx]
            if muscle_idx == 0:
                ax.set_title(f"Subj. {subj_id}")
            if col_idx == 0:
                ax.set_ylabel("Activation")
            if muscle_idx == n_muscles - 1:
                ax.set_xlabel("Gait cycle [%]")

            if np.std(analysis.human.muscle_data[subject_idx, muscle_idx, :]) == 0:
                normalized_activation = analysis.human.muscle_data[subject_idx, muscle_idx, :]
            else:
                normalized_activation = (
                    analysis.human.muscle_data[subject_idx, muscle_idx, :]
                    - np.min(analysis.human.muscle_data[subject_idx, muscle_idx, :])
                ) / (
                    np.max(analysis.human.muscle_data[subject_idx, muscle_idx, :])
                    - np.min(analysis.human.muscle_data[subject_idx, muscle_idx, :])
                )
            ax.plot(time, normalized_activation, color="grey", linewidth=2.5, alpha=0.5, zorder=1)
            for run in analysis.policies:
                if run.profiles.subject_ids is not None and subj_id in run.profiles.subject_ids:
                    policy_activation = run.profiles.per_subject_muscle[subj_id][muscle_idx]
                    normalized_policy_activation = (
                        (policy_activation - np.min(policy_activation))
                        / (np.max(policy_activation) - np.min(policy_activation))
                        if np.std(policy_activation) > 0
                        else policy_activation
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

    fig.savefig(f"{analysis.plots_dir}/human_to_human_muscle_metrics_per_subject.svg", bbox_inches="tight")
    plt.close(fig)
