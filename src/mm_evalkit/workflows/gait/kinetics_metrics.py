"""Subject-indexed Gait120 joint-moment and vertical-GRF metrics."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .numerics import finite_mean, paired_pearson, paired_rmse
from .state import GaitAnalysis, KineticsHumanToHumanMetrics, KineticsMetrics, PerChannel


def _empty(names: tuple[str, ...], aggregate: str) -> PerChannel:
    return {**{name: [] for name in names}, aggregate: []}


def _score_pair(
    first: np.ndarray,
    second: np.ndarray,
    names: tuple[str, ...],
    rmse: PerChannel,
    corr: PerChannel,
    aggregate: str,
) -> None:
    for index, name in enumerate(names):
        rmse[name].append(paired_rmse(first[index], second[index]))
        corr[name].append(paired_pearson(first[index], second[index]))
    rmse[aggregate].append(finite_mean([rmse[name][-1] for name in names]))
    corr[aggregate].append(finite_mean([corr[name][-1] for name in names]))


def _metrics(subjects: list[int], torque_names: tuple[str, ...], grf_names: tuple[str, ...]) -> KineticsMetrics:
    return KineticsMetrics(
        subjects=list(subjects),
        torque_rmse=_empty(torque_names, "all_moments"),
        torque_corr=_empty(torque_names, "all_moments"),
        grf_rmse=_empty(grf_names, "all_grf"),
        grf_corr=_empty(grf_names, "all_grf"),
    )


def _finalize(metrics: KineticsMetrics) -> None:
    metrics.avg_torque_rmse = {channel: finite_mean(values) for channel, values in metrics.torque_rmse.items()}
    metrics.avg_torque_corr = {channel: finite_mean(values) for channel, values in metrics.torque_corr.items()}
    metrics.avg_grf_rmse = {channel: finite_mean(values) for channel, values in metrics.grf_rmse.items()}
    metrics.avg_grf_corr = {channel: finite_mean(values) for channel, values in metrics.grf_corr.items()}


def compute_kinetics_metrics(analysis: GaitAnalysis) -> None:
    """Compute human-human, population, and subject-matched kinetic scores."""
    cohort = analysis.kinetics
    if cohort is None:
        return
    torque_names = cohort.torque_names
    grf_names = cohort.grf_names

    human = KineticsHumanToHumanMetrics(
        subject_pairs=[],
        torque_rmse=_empty(torque_names, "all_moments"),
        torque_corr=_empty(torque_names, "all_moments"),
        grf_rmse=_empty(grf_names, "all_grf"),
        grf_corr=_empty(grf_names, "all_grf"),
    )
    for first in range(cohort.n_subjects):
        for second in range(first + 1, cohort.n_subjects):
            human.subject_pairs.append((cohort.subjects[first], cohort.subjects[second]))
            _score_pair(
                cohort.torque_data[first],
                cohort.torque_data[second],
                torque_names,
                human.torque_rmse,
                human.torque_corr,
                "all_moments",
            )
            _score_pair(
                cohort.grf_data[first],
                cohort.grf_data[second],
                grf_names,
                human.grf_rmse,
                human.grf_corr,
                "all_grf",
            )
    human.avg_torque_rmse = {channel: finite_mean(values) for channel, values in human.torque_rmse.items()}
    human.avg_torque_corr = {channel: finite_mean(values) for channel, values in human.torque_corr.items()}
    human.avg_grf_rmse = {channel: finite_mean(values) for channel, values in human.grf_rmse.items()}
    human.avg_grf_corr = {channel: finite_mean(values) for channel, values in human.grf_corr.items()}
    analysis.kinetics_human_metrics = human

    for run in analysis.policies:
        profiles = run.kinetics_profiles
        if profiles is None:
            raise RuntimeError(f"Policy {run.policy.id!r} has no kinetics profiles for the active gait selection.")

        population = _metrics(cohort.subjects, torque_names, grf_names)
        for position in range(cohort.n_subjects):
            _score_pair(
                profiles.avg_torque,
                cohort.torque_data[position],
                torque_names,
                population.torque_rmse,
                population.torque_corr,
                "all_moments",
            )
            _score_pair(
                profiles.avg_grf,
                cohort.grf_data[position],
                grf_names,
                population.grf_rmse,
                population.grf_corr,
                "all_grf",
            )
        _finalize(population)
        run.kinetics_population = population

        subject_metrics = _metrics(cohort.subjects, torque_names, grf_names)
        for position, subject in enumerate(cohort.subjects):
            if subject not in profiles.per_subject_torque:
                raise RuntimeError(
                    f"Policy {run.policy.id!r} has no subject-matched kinetics profile for subject {subject}."
                )
            _score_pair(
                profiles.per_subject_torque[subject],
                cohort.torque_data[position],
                torque_names,
                subject_metrics.torque_rmse,
                subject_metrics.torque_corr,
                "all_moments",
            )
            _score_pair(
                profiles.per_subject_grf[subject],
                cohort.grf_data[position],
                grf_names,
                subject_metrics.grf_rmse,
                subject_metrics.grf_corr,
                "all_grf",
            )
        _finalize(subject_metrics)
        run.kinetics_per_subject = subject_metrics


def _metric_rows(
    *,
    policy_id: str,
    label: str,
    scope: str,
    unit_ids: list[str],
    metrics: KineticsMetrics | KineticsHumanToHumanMetrics,
):
    groups = (
        ("joint_moment", "rmse", metrics.torque_rmse),
        ("joint_moment", "pearson", metrics.torque_corr),
        ("vertical_grf", "rmse", metrics.grf_rmse),
        ("vertical_grf", "pearson", metrics.grf_corr),
    )
    for signal, metric, channels in groups:
        for channel, values in channels.items():
            if len(values) != len(unit_ids):
                raise RuntimeError(
                    f"Kinetics {scope} {signal}/{metric}/{channel} has {len(values)} values for {len(unit_ids)} units."
                )
            for unit_id, value in zip(unit_ids, values, strict=True):
                yield {
                    "policy_id": policy_id,
                    "label": label,
                    "scope": scope,
                    "signal": signal,
                    "metric": metric,
                    "channel": channel,
                    "unit": "subject_pair" if scope == "human_human" else "subject",
                    "unit_id": unit_id,
                    "value": value,
                }


def write_kinetics_metrics(analysis: GaitAnalysis) -> Path | None:
    """Write every subject-level kinetic score to a tidy CSV."""
    if analysis.kinetics is None:
        return None
    human = analysis.kinetics_human_metrics
    if human is None:
        raise RuntimeError("Kinetics human-to-human metrics have not been computed.")
    rows = list(
        _metric_rows(
            policy_id="",
            label="Human-to-human",
            scope="human_human",
            unit_ids=[f"{first}-{second}" for first, second in human.subject_pairs],
            metrics=human,
        )
    )
    for run in analysis.policies:
        if run.kinetics_population is None or run.kinetics_per_subject is None:
            raise RuntimeError(f"Policy {run.policy.id!r} kinetics metrics have not been computed.")
        rows.extend(
            _metric_rows(
                policy_id=run.policy.id,
                label=run.label,
                scope="population",
                unit_ids=[str(subject) for subject in run.kinetics_population.subjects],
                metrics=run.kinetics_population,
            )
        )
        rows.extend(
            _metric_rows(
                policy_id=run.policy.id,
                label=run.label,
                scope="subject_matched",
                unit_ids=[str(subject) for subject in run.kinetics_per_subject.subjects],
                metrics=run.kinetics_per_subject,
            )
        )

    path = Path(analysis.plots_dir) / "kinetics_metrics.csv"
    fieldnames = ("policy_id", "label", "scope", "signal", "metric", "channel", "unit", "unit_id", "value")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path
