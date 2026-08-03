"""Subject-level paired statistics for the ULTRA-MoCap analysis."""

from __future__ import annotations

import warnings
from itertools import combinations

import numpy as np

from mm_evalkit.stats.report import print_wilcoxon_summary, save_wilcoxon_report
from mm_evalkit.stats.wilcoxon import MetricFamily, MetricSamples, wilcoxon_table

from .constants import (
    METRIC_KEYS,
    REP_TO_REP_KEYS,
    SPLIT_HALF_KEYS,
    WITHIN_SUBJECT_LABEL,
    WITHIN_SUBJECT_VARIABILITY_LABEL,
)

POLICY_KEYS = {
    "joint_rmse": "joint_rmse_by",
    "muscle_corr": "muscle_corr_by",
    "muscle_rsa": "muscle_rsa",
}


def _samples_from_records(records, key, channels, units):
    """Read one trial metric into a ``(trials, channels)`` matrix."""
    values = np.full((len(units), len(channels)), np.nan)
    for index, unit in enumerate(units):
        row = records.get(unit, {}).get(key)
        if row is None or not len(row):
            continue
        row = np.asarray(row, dtype=float)
        if row.ndim == 2:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                row = np.nanmean(row, axis=0)
        if row.shape != (len(channels),):
            raise ValueError(f"{key!r} has shape {row.shape}; expected {len(channels)} channel values.")
        values[index] = row
    return MetricSamples(values, tuple(units), tuple(channels))


def _aggregate_matched_trials_by_subject(
    first: MetricSamples,
    second: MetricSamples,
) -> tuple[MetricSamples, MetricSamples]:
    """Average identical finite trial pairs within each subject.

    Movement-condition trials are repeated observations of a subject, not
    independent samples. Matching is performed separately for every channel:
    a trial contributes to both condition means only when that channel is
    finite on both sides.
    """
    if first.units != second.units:
        raise ValueError("Conditions must declare identical trial units.")
    if first.channels != second.channels:
        raise ValueError("Conditions must declare identical metric channels.")

    subjects = tuple(dict.fromkeys(unit[2] for unit in first.units))
    first_values = np.full((len(subjects), len(first.channels)), np.nan)
    second_values = np.full_like(first_values, np.nan)

    for subject_index, subject in enumerate(subjects):
        trial_indices = np.asarray(
            [index for index, unit in enumerate(first.units) if unit[2] == subject],
            dtype=int,
        )
        for channel_index in range(len(first.channels)):
            first_trials = first.values[trial_indices, channel_index]
            second_trials = second.values[trial_indices, channel_index]
            paired = np.isfinite(first_trials) & np.isfinite(second_trials)
            if paired.any():
                first_values[subject_index, channel_index] = np.mean(first_trials[paired])
                second_values[subject_index, channel_index] = np.mean(second_trials[paired])

    return (
        MetricSamples(first_values, subjects, first.channels),
        MetricSamples(second_values, subjects, second.channels),
    )


def run_wilcoxon_tests(
    policy_records,
    policies,
    real_records,
    joint_names,
    muscle_labels,
    plots_dir,
):
    """Run paired tests over subjects after matching their available trials."""
    units = sorted(real_records)
    if not units:
        return None

    channels = {
        "joint_rmse": tuple(joint_names),
        "muscle_corr": tuple(muscle_labels),
        "muscle_rsa": ("electrode RSM",),
    }
    samples = {key: {} for key in METRIC_KEYS}
    legend = {}

    def add(name, description, records, keys):
        legend[name] = description
        for metric in METRIC_KEYS:
            samples[metric][name] = _samples_from_records(
                records,
                keys[metric],
                channels[metric],
                units,
            )

    policy_names = []
    for slot, policy in enumerate(policies):
        records = policy_records.get(policy.id, {})
        if not records:
            continue
        name = f"P{slot + 1}"
        add(name, policy.label, records, POLICY_KEYS)
        policy_names.append(name)

    ceiling = "Split-half (real)"
    add(
        ceiling,
        " ".join(WITHIN_SUBJECT_LABEL.split()),
        real_records,
        SPLIT_HALF_KEYS,
    )
    rep_to_rep = "Rep-to-rep (real)"
    add(
        rep_to_rep,
        " ".join(WITHIN_SUBJECT_VARIABILITY_LABEL.split()),
        real_records,
        REP_TO_REP_KEYS,
    )

    comparisons = [
        ("Policy vs policy", first, second, first, second) for first, second in combinations(policy_names, 2)
    ]
    for name in policy_names:
        split_half_name = f"{name} split-half real"
        rep_to_rep_name = f"{name} rep-to-rep real"
        legend[split_half_name] = legend[ceiling]
        legend[rep_to_rep_name] = legend[rep_to_rep]
        comparisons.extend(
            (
                (
                    "Policy vs matched split-half ceiling (real)",
                    name,
                    split_half_name,
                    name,
                    ceiling,
                ),
                (
                    "Policy vs matched rep-to-rep variability (real)",
                    name,
                    rep_to_rep_name,
                    name,
                    rep_to_rep,
                ),
            )
        )
    if not comparisons:
        return None

    families = []
    for metric_name, channel_name, metric_key in (
        ("Joint tracking error [deg]", "joint", "joint_rmse"),
        ("Muscle-EMG correlation", "electrode", "muscle_corr"),
        ("Muscle RSA (Spearman)", "electrode", "muscle_rsa"),
    ):
        for group, display_a, display_b, source_a, source_b in comparisons:
            matched_a, matched_b = _aggregate_matched_trials_by_subject(
                samples[metric_key][source_a],
                samples[metric_key][source_b],
            )
            families.append(
                MetricFamily(
                    metric_name,
                    channel_name,
                    {display_a: matched_a, display_b: matched_b},
                    [(group, display_a, display_b)],
                )
            )

    table, channel_table = wilcoxon_table(families, unit_name="subject")
    print_wilcoxon_summary(table, channel_table, unit_name="subject")
    n_subjects = len({unit[2] for unit in units})
    save_wilcoxon_report(
        table,
        channel_table,
        plots_dir,
        legend=legend,
        subtitle=(
            f"ULTRA-MoCap: {n_subjects} subjects; trials matched within subject; "
            f"{len(joint_names)} joints, {len(muscle_labels)} electrodes."
        ),
    )
    return table
