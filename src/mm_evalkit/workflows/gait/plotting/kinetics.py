"""Gait120 joint-moment and vertical-GRF profiles and metric summaries."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.gridspec import GridSpec

from ..constants import BETWEEN_SUBJECT_HATCH, BETWEEN_SUBJECT_LABEL, REF_GREY
from .style import draw_grouped_hbar, policy_fill_alpha


def _plot_signal_profiles(
    analysis,
    *,
    signal: str,
    names: tuple[str, ...],
    unit: str,
    human_data: np.ndarray,
    human_metric: dict[str, np.ndarray],
    policy_metrics: list[dict[str, np.ndarray]],
    policy_profiles: list[np.ndarray],
    aggregate_name: str,
    metric_label: str,
    output_name: str,
) -> None:
    """Draw one signal family in the shared gait metric/profile layout.

    Kinematics and EMG use an aggregate bar panel, per-channel bars, and one
    waveform per row.  Moments and GRF follow that same layout so the figures
    can be compared directly instead of requiring a separate kinetics legend.
    """
    n_channels = len(names)
    phase = np.linspace(0.0, 100.0, human_data.shape[-1])
    fig = plt.figure(figsize=(14, 1.9 * n_channels + 1))
    grid = GridSpec(n_channels, 3, figure=fig, width_ratios=[1, 1, 2], hspace=0.6, wspace=0.3)

    aggregate_ax = fig.add_subplot(grid[:, 0])
    draw_grouped_hbar(
        analysis,
        aggregate_ax,
        human_metric[aggregate_name],
        BETWEEN_SUBJECT_LABEL,
        [metrics[aggregate_name] for metrics in policy_metrics],
        show_group_labels=True,
        ref_hatch=BETWEEN_SUBJECT_HATCH,
    )
    aggregate_ax.set_xlabel(metric_label)
    aggregate_ax.set_title(f"All {signal}")

    metric_ax_ref = None
    for index, name in enumerate(names):
        metric_ax = fig.add_subplot(grid[index, 1], sharex=metric_ax_ref)
        metric_ax_ref = metric_ax_ref or metric_ax
        draw_grouped_hbar(
            analysis,
            metric_ax,
            human_metric[name],
            BETWEEN_SUBJECT_LABEL,
            [metrics[name] for metrics in policy_metrics],
            show_group_labels=False,
            ref_hatch=BETWEEN_SUBJECT_HATCH,
        )
        metric_ax.set_title(name)
        if index == n_channels - 1:
            metric_ax.set_xlabel(metric_label)

        profile_ax = fig.add_subplot(grid[index, 2])
        for human_profile in human_data[:, index, :]:
            profile_ax.plot(phase, human_profile, color=REF_GREY, linewidth=2.5, alpha=0.5, zorder=1)
        for run, profile in zip(analysis.policies, policy_profiles, strict=True):
            profile_ax.plot(
                phase,
                profile[index],
                color=run.policy.color,
                linestyle=run.policy.linestyle,
                linewidth=2.5,
                alpha=0.8,
                zorder=2,
            )
        profile_ax.set_ylabel(unit)
        if index == n_channels - 1:
            profile_ax.set_xlabel("Gait cycle [%]")
        profile_ax.spines[["top", "right"]].set_visible(False)

    fig.savefig(analysis.plots_dir / output_name, bbox_inches="tight")
    plt.close(fig)


def plot_kinetics_profiles(analysis) -> None:
    """Plot joint moments in the standard gait metric/profile layout."""
    cohort = analysis.kinetics
    if cohort is None:
        return
    human = analysis.kinetics_human_metrics
    if human is None:
        raise RuntimeError("Kinetics metrics have not been computed.")
    profiles = []
    for run in analysis.policies:
        if run.kinetics_profiles is None:
            raise RuntimeError(f"Policy {run.policy.id!r} has no kinetics profiles.")
        profiles.append(run.kinetics_profiles.avg_torque)
    _plot_signal_profiles(
        analysis,
        signal="moments",
        names=cohort.torque_names,
        unit=cohort.torque_unit,
        human_data=cohort.torque_data,
        human_metric=human.torque_rmse,
        policy_metrics=[run.kinetics_population.torque_rmse for run in analysis.policies],
        policy_profiles=profiles,
        aggregate_name="all_moments",
        metric_label="RMSE [Nm/kg]",
        output_name="kinetics_profiles.svg",
    )


def plot_grf_profiles(analysis) -> None:
    """Plot bilateral vertical GRF in the standard gait metric/profile layout."""
    cohort = analysis.kinetics
    if cohort is None:
        return
    human = analysis.kinetics_human_metrics
    if human is None:
        raise RuntimeError("Kinetics metrics have not been computed.")
    profiles = []
    for run in analysis.policies:
        if run.kinetics_profiles is None:
            raise RuntimeError(f"Policy {run.policy.id!r} has no kinetics profiles.")
        profiles.append(run.kinetics_profiles.avg_grf)
    _plot_signal_profiles(
        analysis,
        signal="vertical GRF",
        names=cohort.grf_names,
        unit=cohort.grf_unit,
        human_data=cohort.grf_data,
        human_metric=human.grf_rmse,
        policy_metrics=[run.kinetics_population.grf_rmse for run in analysis.policies],
        policy_profiles=profiles,
        aggregate_name="all_grf",
        metric_label="RMSE [BW]",
        output_name="grf_profiles.svg",
    )


def _plot_signal_profiles_per_subject(
    analysis,
    *,
    signal: str,
    names: tuple[str, ...],
    unit: str,
    human_data: np.ndarray,
    human_subjects: list[int],
    human_metric: dict[str, np.ndarray],
    policy_metrics: list[dict[str, np.ndarray]],
    policy_profiles: list[dict[int, np.ndarray]],
    aggregate_name: str,
    metric_label: str,
    output_name: str,
) -> None:
    """Draw selected subject waveforms beside subject-matched metric bars."""
    if not analysis.subject_based:
        return

    requested_subjects = [analysis.human.subjects[index] for index in analysis.human.plot_subject_positions]
    subject_indexes = {subject: index for index, subject in enumerate(human_subjects)}
    subjects = [subject for subject in requested_subjects if subject in subject_indexes]
    if not subjects:
        return

    n_channels = len(names)
    phase = np.linspace(0.0, 100.0, human_data.shape[-1])
    fig = plt.figure(figsize=(10 + 2.2 * len(subjects), 1.9 * n_channels + 1))
    grid = GridSpec(
        n_channels,
        2 + len(subjects),
        figure=fig,
        width_ratios=[3, 3] + [2.2] * len(subjects),
        hspace=0.6,
        wspace=0.45,
    )

    aggregate_ax = fig.add_subplot(grid[:, 0])
    draw_grouped_hbar(
        analysis,
        aggregate_ax,
        human_metric[aggregate_name],
        BETWEEN_SUBJECT_LABEL,
        [metrics[aggregate_name] for metrics in policy_metrics],
        show_group_labels=True,
        ref_hatch=BETWEEN_SUBJECT_HATCH,
    )
    aggregate_ax.set_xlabel(metric_label)
    aggregate_ax.set_title(f"All {signal}")

    metric_ax_ref = None
    for channel_index, name in enumerate(names):
        metric_ax = fig.add_subplot(grid[channel_index, 1], sharex=metric_ax_ref)
        metric_ax_ref = metric_ax_ref or metric_ax
        draw_grouped_hbar(
            analysis,
            metric_ax,
            human_metric[name],
            BETWEEN_SUBJECT_LABEL,
            [metrics[name] for metrics in policy_metrics],
            show_group_labels=False,
            ref_hatch=BETWEEN_SUBJECT_HATCH,
        )
        metric_ax.set_ylabel(name)
        if channel_index == 0:
            metric_ax.set_title("RMSE")
        if channel_index == n_channels - 1:
            metric_ax.set_xlabel(metric_label)

        for column_index, subject in enumerate(subjects):
            profile_ax = fig.add_subplot(grid[channel_index, 2 + column_index])
            if channel_index == 0:
                profile_ax.set_title(f"Subj. {subject}")
            if column_index == 0:
                profile_ax.set_ylabel(unit)
            if channel_index == n_channels - 1:
                profile_ax.set_xlabel("Gait cycle [%]")
            profile_ax.plot(
                phase,
                human_data[subject_indexes[subject], channel_index],
                color=REF_GREY,
                linewidth=2.5,
                alpha=0.5,
                zorder=1,
            )
            for run, profiles in zip(analysis.policies, policy_profiles, strict=True):
                profile = profiles.get(subject)
                if profile is None:
                    continue
                profile_ax.plot(
                    phase,
                    profile[channel_index],
                    color=run.policy.color,
                    linestyle=run.policy.linestyle,
                    linewidth=2.5,
                    alpha=0.8,
                    zorder=2,
                )
            profile_ax.spines[["top", "right"]].set_visible(False)

    fig.savefig(analysis.plots_dir / output_name, bbox_inches="tight")
    plt.close(fig)


def plot_subject_based_kinetics_profiles(analysis) -> None:
    """Plot subject-matched joint-moment metrics and selected subject waveforms."""
    cohort = analysis.kinetics
    if cohort is None:
        return
    human = analysis.kinetics_human_metrics
    if human is None:
        raise RuntimeError("Kinetics metrics have not been computed.")
    profiles = []
    for run in analysis.policies:
        if run.kinetics_profiles is None:
            raise RuntimeError(f"Policy {run.policy.id!r} has no kinetics profiles.")
        profiles.append(run.kinetics_profiles.per_subject_torque)
    _plot_signal_profiles_per_subject(
        analysis,
        signal="moments",
        names=cohort.torque_names,
        unit=cohort.torque_unit,
        human_data=cohort.torque_data,
        human_subjects=cohort.subjects,
        human_metric=human.torque_rmse,
        policy_metrics=[run.kinetics_per_subject.torque_rmse for run in analysis.policies],
        policy_profiles=profiles,
        aggregate_name="all_moments",
        metric_label="RMSE [Nm/kg]",
        output_name="kinetics_profiles_per_subject.svg",
    )


def plot_subject_based_grf_profiles(analysis) -> None:
    """Plot subject-matched GRF metrics and selected subject waveforms."""
    cohort = analysis.kinetics
    if cohort is None:
        return
    human = analysis.kinetics_human_metrics
    if human is None:
        raise RuntimeError("Kinetics metrics have not been computed.")
    profiles = []
    for run in analysis.policies:
        if run.kinetics_profiles is None:
            raise RuntimeError(f"Policy {run.policy.id!r} has no kinetics profiles.")
        profiles.append(run.kinetics_profiles.per_subject_grf)
    _plot_signal_profiles_per_subject(
        analysis,
        signal="vertical GRF",
        names=cohort.grf_names,
        unit=cohort.grf_unit,
        human_data=cohort.grf_data,
        human_subjects=cohort.subjects,
        human_metric=human.grf_rmse,
        policy_metrics=[run.kinetics_per_subject.grf_rmse for run in analysis.policies],
        policy_profiles=profiles,
        aggregate_name="all_grf",
        metric_label="RMSE [BW]",
        output_name="grf_profiles_per_subject.svg",
    )


def _finite(values) -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    return array[np.isfinite(array)]


def _metric_bars(analysis, ax, human_values, population_values, subject_values, *, xlabel: str, title: str) -> None:
    labels = ["Human-human"]
    samples = [_finite(human_values)]
    colors = [REF_GREY]
    hatches = ["//"]
    linestyles = ["-"]
    for run, population, subject in zip(analysis.policies, population_values, subject_values, strict=True):
        labels.extend((f"{run.label}\nPopulation", f"{run.label}\nSubject-matched"))
        samples.extend((_finite(population), _finite(subject)))
        colors.extend((run.policy.color, run.policy.color))
        hatches.extend(("", ".."))
        linestyles.extend((run.policy.linestyle, run.policy.linestyle))

    y = np.arange(len(labels))[::-1]
    for index, (position, values, color, hatch, linestyle) in enumerate(
        zip(y, samples, colors, hatches, linestyles, strict=True)
    ):
        if values.size == 0:
            continue
        alpha = 0.7 if index == 0 else policy_fill_alpha(linestyle)
        ax.barh(
            position,
            np.mean(values),
            xerr=np.std(values),
            color=to_rgba(color, alpha),
            edgecolor="grey" if index == 0 else color,
            hatch=hatch,
            linestyle=linestyle,
            capsize=3,
        )
    ax.set_yticks(y, labels, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.2)


def plot_kinetics_metrics(analysis) -> None:
    """Plot aggregate RMSE and Pearson samples in both analysis scopes."""
    if analysis.kinetics is None:
        return
    human = analysis.kinetics_human_metrics
    if human is None:
        raise RuntimeError("Kinetics metrics have not been computed.")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    _metric_bars(
        analysis,
        axes[0, 0],
        human.torque_rmse["all_moments"],
        [run.kinetics_population.torque_rmse["all_moments"] for run in analysis.policies],
        [run.kinetics_per_subject.torque_rmse["all_moments"] for run in analysis.policies],
        xlabel="RMSE [Nm/kg]",
        title="Joint moments",
    )
    _metric_bars(
        analysis,
        axes[0, 1],
        human.torque_corr["all_moments"],
        [run.kinetics_population.torque_corr["all_moments"] for run in analysis.policies],
        [run.kinetics_per_subject.torque_corr["all_moments"] for run in analysis.policies],
        xlabel="Pearson r",
        title="Joint moments",
    )
    _metric_bars(
        analysis,
        axes[1, 0],
        human.grf_rmse["all_grf"],
        [run.kinetics_population.grf_rmse["all_grf"] for run in analysis.policies],
        [run.kinetics_per_subject.grf_rmse["all_grf"] for run in analysis.policies],
        xlabel="RMSE [BW]",
        title="Vertical GRF",
    )
    _metric_bars(
        analysis,
        axes[1, 1],
        human.grf_corr["all_grf"],
        [run.kinetics_population.grf_corr["all_grf"] for run in analysis.policies],
        [run.kinetics_per_subject.grf_corr["all_grf"] for run in analysis.policies],
        xlabel="Pearson r",
        title="Vertical GRF",
    )
    fig.suptitle("Gait120 kinetics metrics")
    fig.tight_layout()
    fig.savefig(analysis.plots_dir / "kinetics_metrics.svg", bbox_inches="tight")
    plt.close(fig)
