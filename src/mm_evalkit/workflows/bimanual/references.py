"""Real-data reliability and between-subject reference calculations."""

from __future__ import annotations

from itertools import combinations

import numpy as np

from mm_evalkit.stats.rsa import rsa

from .metrics import _row_correlations, _row_rmse

_REAL_RECORD_FIELDS = (
    "ref_cycle",
    "ref_std",
    "emg_cycle",
    "emg_std",
    "emg_rsm",
    "joint_rmse_within_by",
    "joint_rmse_within",
    "muscle_corr_within_by",
    "muscle_corr_within",
    "muscle_rsa_within",
    "joint_rmse_pairwise_by",
    "muscle_corr_pairwise_by",
    "muscle_rsa_pairwise_by",
)


def select_reference_record(entries):
    """Return the first of several identical real-data records.

    Cross-policy boundaries are validated before reporting. This final check
    ensures the derived real-data fields are also identical, rather than
    choosing whichever policy happened to retain more repetitions.
    """
    if not entries:
        raise ValueError("At least one policy record is required.")
    baseline_policy, baseline = entries[0]
    for policy, candidate in entries[1:]:
        for field in _REAL_RECORD_FIELDS:
            if field not in baseline and field not in candidate:
                continue
            if field not in baseline or field not in candidate:
                raise ValueError(
                    f"Policies {baseline_policy.id!r} and {policy.id!r} disagree on real-data field {field!r}."
                )
            if not np.array_equal(
                np.asarray(baseline[field]),
                np.asarray(candidate[field]),
                equal_nan=True,
            ):
                raise ValueError(
                    f"Policies {baseline_policy.id!r} and {policy.id!r} contain different real-data field {field!r}."
                )
    return baseline


def compute_within_subject_reference(records):
    """Each subject's own split-half agreement, per trial.

    Returned in the same shape as a trial's metrics and keyed the same way as
    ``records``, so the per-subject bars can draw it across each subject's own bar.
    """
    reference = {}
    for key, metrics in records.items():
        entry = {}
        if "joint_rmse_within_by" in metrics:
            entry["joint_rmse_by"] = metrics["joint_rmse_within_by"]
            entry["joint_rmse"] = metrics["joint_rmse_within"]
        if "muscle_corr_within_by" in metrics:
            entry["muscle_corr_by"] = metrics["muscle_corr_within_by"]
            entry["muscle_corr"] = metrics["muscle_corr_within"]
        if "muscle_rsa_within" in metrics:
            entry["muscle_rsa"] = metrics["muscle_rsa_within"]
        if entry:
            reference[key] = entry
    return reference


def compute_within_subject_variability(records):
    """Each subject's repetition-to-repetition agreement, averaged over subjects.

    Per trial the metric is computed between **every pair of that subject's individual
    repetitions** and averaged over the pairs (see :func:`trial_metrics`); the value
    for a movement/condition is then the mean over the subjects that have that trial.
    Returned keyed by ``(movement, condition)`` in the same shape as a trial's metrics,
    so the bars draw it as one extra grey bar per group beside the between-subject one.

    Each row of the returned ``*_by`` arrays is one subject, so the bar's whisker is
    the spread of this quantity *between* subjects.

    This is a single-repetition quantity -- how far one repetition of the same person
    is from another. It is deliberately not the ceiling the policy bars are up against:
    those are scored on the average of every repetition, and the matched ceiling for
    that is the split-half tick (see :func:`compute_within_subject_reference`).
    """
    by_group = {}
    for (movement, condition, _), metrics in records.items():
        by_group.setdefault((movement, condition), []).append(metrics)

    reference = {}
    for group, trials in by_group.items():
        entry = {}
        for source, target in (
            ("joint_rmse_pairwise_by", "joint_rmse"),
            ("muscle_corr_pairwise_by", "muscle_corr"),
            ("muscle_rsa_pairwise_by", "muscle_rsa"),
        ):
            rows = [metrics[source] for metrics in trials if source in metrics]
            if not rows:
                continue
            stacked = np.asarray(rows)  # (n_subjects, n_channels)
            entry[f"{target}_by"] = stacked
            entry[target] = np.nanmean(stacked, axis=1)
        if entry:
            reference[group] = entry
    return reference


def compute_between_subject_reference(records):
    """How far two *real* subjects are from each other, per movement/condition.

    The policy bars say how well the policy matches its own subject; on their own
    they don't say whether that is good. This computes the same two scores between
    every pair of subjects using only real data -- reference kinematics vs reference
    kinematics, real EMG vs real EMG -- on the per-subject average cycles. Both are
    phase-aligned by construction (each cycle runs peak-to-peak of the movement's
    critical joint) and time-normalized to the same 100 points.

    It is computed on trial-average cycles, which is exactly how the policy is now
    scored, so the two are directly comparable.

    Returns ``{(movement, condition): metrics}`` in the same shape as a trial's
    metrics, so the bars are drawn by the same code. Movement/conditions with fewer
    than two subjects are absent (no pair to compare).
    """
    by_group = {}
    for (movement, condition, _), metrics in records.items():
        by_group.setdefault((movement, condition), []).append(metrics)

    reference = {}
    for group, trials in by_group.items():
        entry = {}
        joints = [m["ref_cycle"] for m in trials if "ref_cycle" in m]
        if len(joints) > 1:
            # RMSE over the cycle between the two subjects, per joint.
            rmse = np.array([_row_rmse(a, b) for a, b in combinations(joints, 2)])
            entry["joint_rmse_by"] = rmse
            entry["joint_rmse"] = np.nanmean(rmse, axis=1)
        emgs = [m["emg_cycle"] for m in trials if "emg_cycle" in m]
        if len(emgs) > 1:
            corr = np.array([_row_correlations(a, b) for a, b in combinations(emgs, 2)])
            if not np.isnan(corr).all():
                entry["muscle_corr_by"] = corr
                entry["muscle_corr"] = np.nanmean(corr, axis=1)
            # RSA between two subjects: how much their muscle *coordination* differs,
            # the reference for the policy's RSA bar.
            rsms = [m["emg_rsm"] for m in trials if "emg_rsm" in m]
            pair_rsa = [rsa(a, b)[0] for a, b in combinations(rsms, 2)]
            pair_rsa = [s for s in pair_rsa if np.isfinite(s)]
            if pair_rsa:
                entry["muscle_rsa"] = np.asarray(pair_rsa, dtype=float)
        if entry:
            reference[group] = entry
    return reference


def merge_real_records(policy_records, policies):
    """Return one validated real-data record per shared trial.

    ``validate_comparable_trials`` has already required identical trial and
    repetition sets. This function verifies the derived real fields and then
    uses configuration order only to choose an equivalent representation.
    """
    if not policies:
        return {}
    keys = set(policy_records.get(policies[0].id, {}))
    for policy in policies[1:]:
        current = set(policy_records.get(policy.id, {}))
        if current != keys:
            raise ValueError(
                "Policy metric records must have identical trial keys before real-data records are merged."
            )
    merged = {}
    for key in keys:
        entries = [
            (policy, policy_records[policy.id][key]) for policy in policies if key in policy_records.get(policy.id, {})
        ]
        merged[key] = select_reference_record(entries)
    return merged
