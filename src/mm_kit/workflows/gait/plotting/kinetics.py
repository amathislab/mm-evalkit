"""Gait120 joint-moment and vertical-GRF profiles and metric summaries."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba

from ..constants import REF_GREY
from .style import policy_fill_alpha


def plot_kinetics_profiles(analysis) -> None:
    """Plot the six joint moments and bilateral vertical GRF over gait phase."""
    cohort = analysis.kinetics
    if cohort is None:
        return
    phase = np.linspace(0.0, 100.0, 100)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True)
    channels = [
        *(("torque", index, name, cohort.torque_unit) for index, name in enumerate(cohort.torque_names)),
        *(("grf", index, name, cohort.grf_unit) for index, name in enumerate(cohort.grf_names)),
    ]
    for ax, (signal, index, name, unit) in zip(axes.ravel(), channels, strict=True):
        human = cohort.torque_data[:, index] if signal == "torque" else cohort.grf_data[:, index]
        human_mean = np.nanmean(human, axis=0)
        human_std = np.nanstd(human, axis=0)
        ax.fill_between(phase, human_mean - human_std, human_mean + human_std, color=REF_GREY, alpha=0.35)
        ax.plot(phase, human_mean, color="black", linewidth=2, label="OpenSim human")
        for run in analysis.policies:
            profiles = run.kinetics_profiles
            if profiles is None:
                raise RuntimeError(f"Policy {run.policy.id!r} has no kinetics profiles.")
            values = profiles.avg_torque[index] if signal == "torque" else profiles.avg_grf[index]
            ax.plot(
                phase,
                values,
                color=run.policy.color,
                linestyle=run.policy.linestyle,
                linewidth=2,
                label=run.label,
            )
        ax.set_title(name)
        ax.set_ylabel(unit)
        ax.set_xlabel("Gait cycle [%]")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))
    fig.suptitle("Gait120 kinetics profiles", y=1.02)
    fig.tight_layout()
    fig.savefig(analysis.plots_dir / "kinetics_profiles.svg", bbox_inches="tight")
    plt.close(fig)


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
