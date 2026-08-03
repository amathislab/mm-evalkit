"""Population and subject-level kinematic figures."""

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


def plot_kinematic_metrics(analysis):
    """Generate plots for kinematic metrics (joint angle time series and RMSE)."""

    joint_names = analysis.channels.joint_names
    n_joints = len(joint_names)

    # Layout: the all-joints RMSE spans the full height of a wide first column (it doubles
    # as the legend), then one row per joint with its RMSE bars beside its time series.
    fig = plt.figure(figsize=(14, 1.9 * n_joints + 1))
    gs = GridSpec(n_joints, 3, figure=fig, width_ratios=[1, 1, 2], hspace=0.6, wspace=0.3)

    # =========================
    # All-Joints RMSE Plot (First Column)
    # =========================
    ax_all = fig.add_subplot(gs[:, 0])
    draw_grouped_hbar(
        analysis,
        ax_all,
        analysis.human_metrics.pairwise_joint_rmse["all_joints"],
        BETWEEN_SUBJECT_LABEL,
        [run.population.joint_rmse["all_joints"] for run in analysis.policies],
        show_group_labels=True,
        ref_hatch=BETWEEN_SUBJECT_HATCH,
    )
    ax_all.set_xlabel("RMSE [degrees]")
    ax_all.set_title("All joints")

    # =========================
    # Per-Joint RMSE Plots (Second Column) and Time Series (Third Column)
    # =========================
    time_steps = 100  # Standard gait cycle length
    time = np.linspace(0, 100, time_steps)

    # Per-joint bars share one x scale so their lengths stay comparable across joints.
    ax_rmse_ref = None

    for joint_idx, joint_name in enumerate(joint_names):
        ax_rmse = fig.add_subplot(gs[joint_idx, 1], sharex=ax_rmse_ref)
        ax_rmse_ref = ax_rmse_ref or ax_rmse
        draw_grouped_hbar(
            analysis,
            ax_rmse,
            analysis.human_metrics.pairwise_joint_rmse[joint_name],
            BETWEEN_SUBJECT_LABEL,
            [run.population.joint_rmse[joint_name] for run in analysis.policies],
            show_group_labels=False,
            ref_hatch=BETWEEN_SUBJECT_HATCH,
        )
        ax_rmse.set_title(f"{joint_name}")
        if joint_idx == n_joints - 1:
            ax_rmse.set_xlabel("RMSE [degrees]")

        ax = fig.add_subplot(gs[joint_idx, 2])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Plot average human joint angle
        for subject_idx in range(analysis.human.joint_data.shape[0]):
            subject_angle = analysis.human.joint_data[subject_idx, joint_idx, :]
            ax.plot(time, subject_angle, color="lightgrey", linewidth=2.5, alpha=0.5, zorder=1)

        # Plot each policy's joint angle
        for run in analysis.policies:
            policy_angle = run.profiles.avg_joint_gaits[joint_idx]
            ax.plot(
                time,
                policy_angle,
                color=run.policy.color,
                linestyle=run.policy.linestyle,
                linewidth=2.5,
                alpha=0.8,
                zorder=2,
            )

        ax.set_ylabel("Angle [deg]")
        if joint_idx == n_joints - 1:
            ax.set_xlabel("Gait cycle [%]")

    plt.savefig(f"{analysis.plots_dir}/human_to_human_kinematic_metrics.svg", bbox_inches="tight")
    plt.close(fig)


def plot_subject_based_kinematic_metrics(analysis):
    """Generate subject-based plots for kinematic metrics (joint RMSE)."""
    if not analysis.subject_based:
        print("Subject-based plotting is disabled. Set subject_based=True to enable.")
        return

    joint_names = analysis.channels.joint_names
    n_joints = len(joint_names)

    # Subject-based reference = within-subject reliability (the subject-matched upper
    # bound); fall back to the between-subject pairwise reference if it is unavailable.
    wref = analysis.references.within_subject
    if wref is not None and wref.n > 0:
        ref_label, ref_hatch = WITHIN_SUBJECT_LABEL, WITHIN_SUBJECT_HATCH
        ref_all = wref.joint_rmse["all_joints"]
        ref_per = wref.joint_rmse
    else:
        ref_label, ref_hatch = BETWEEN_SUBJECT_LABEL, BETWEEN_SUBJECT_HATCH
        ref_all = analysis.human_metrics.pairwise_joint_rmse["all_joints"]
        ref_per = analysis.human_metrics.pairwise_joint_rmse

    # Restrict the one-column-per-subject time series to a fixed random subset.
    plot_positions = analysis.human.plot_subject_positions
    n_subjects = len(plot_positions)

    # Layout: the all-joints bars span the full height of a wide first column (they double as
    # the legend), then one row per joint holding its RMSE bars followed by one time series
    # per subject. The width ratios keep the first column at 1/4 of the figure.
    fig = plt.figure(figsize=(10 + 2.2 * n_subjects, 1.9 * n_joints + 1))
    # fig.suptitle("Kinematic Metrics per Subject")
    gs = GridSpec(
        n_joints,
        2 + n_subjects,
        figure=fig,
        width_ratios=[3, 3] + [2.2] * n_subjects,
        hspace=0.6,
        wspace=0.45,
    )

    # =========================
    # All-Joints RMSE Plot (First Column)
    # =========================
    ax_all = fig.add_subplot(gs[:, 0])
    draw_grouped_hbar(
        analysis,
        ax_all,
        ref_all,
        ref_label,
        [run.per_subject.joint_rmse["all_joints"] for run in analysis.policies],
        show_group_labels=True,
        ref_hatch=ref_hatch,
    )
    ax_all.set_xlabel("RMSE [degrees]")
    ax_all.set_title("All joints")

    # =========================
    # Per-Joint RMSE Plots (Second Column) and Time Series (Remaining Columns)
    # =========================
    time_steps = 100  # Standard gait cycle length
    time = np.linspace(0, 100, time_steps)

    # Per-joint bars share one x scale so their lengths stay comparable across joints.
    ax_rmse_ref = None

    for joint_idx, joint_name in enumerate(joint_names):
        ax_rmse = fig.add_subplot(gs[joint_idx, 1], sharex=ax_rmse_ref)
        ax_rmse_ref = ax_rmse_ref or ax_rmse
        draw_grouped_hbar(
            analysis,
            ax_rmse,
            ref_per[joint_name],
            ref_label,
            [run.per_subject.joint_rmse[joint_name] for run in analysis.policies],
            show_group_labels=False,
            ref_hatch=ref_hatch,
        )
        # The group names live on the all-joints panel, so the free y label carries the joint.
        ax_rmse.set_ylabel(f"{joint_name}")
        if joint_idx == 0:
            ax_rmse.set_title("RMSE")
        if joint_idx == n_joints - 1:
            ax_rmse.set_xlabel("RMSE [degrees]")

        for col_idx, subject_idx in enumerate(plot_positions):
            ax = fig.add_subplot(gs[joint_idx, 2 + col_idx])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            subj_id = analysis.human.subjects[subject_idx]
            if joint_idx == 0:
                ax.set_title(f"Subj. {subj_id}")
            if col_idx == 0:
                ax.set_ylabel("Angle [deg]")
            if joint_idx == n_joints - 1:
                ax.set_xlabel("Gait cycle [%]")

            subject_angle = analysis.human.joint_data[subject_idx, joint_idx, :]
            ax.plot(time, subject_angle, color="grey", linewidth=2.5, alpha=0.5, zorder=1)
            for run in analysis.policies:
                if run.profiles.subject_ids is not None and subj_id in run.profiles.subject_ids:
                    policy_angle = run.profiles.per_subject_joint[subj_id][joint_idx]
                    ax.plot(
                        time,
                        policy_angle,
                        color=run.policy.color,
                        linestyle=run.policy.linestyle,
                        linewidth=2.5,
                        alpha=0.8,
                        zorder=2,
                    )

    fig.savefig(f"{analysis.plots_dir}/human_to_human_kinematic_metrics_per_subject.svg", bbox_inches="tight")
    plt.close(fig)
