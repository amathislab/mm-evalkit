"""Cycle-level neuromuscular metrics for ULTRA-MoCap."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np

from mm_evalkit.stats.rsa import compute_rsm, rsa
from mm_evalkit.workflows.bimanual.segmentation import average_cycles, resample_cycles

from .constants import SPLIT_HALF_RMSE_FACTOR
from .models import MuscleGroup, TrialData, TrialMetrics
from .muscles import combine_groups


def _cycle_mean_std(cycles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std over repetitions, quiet on all-NaN channels."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(cycles, axis=0), np.nanstd(cycles, axis=0)


def _finite_mean(values, axis=None):
    """Mean over finite values, returning NaN when none are available."""
    values = np.asarray(values, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(
            np.where(np.isfinite(values), values, np.nan),
            axis=axis,
        )


def _row_correlations(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson r per channel over samples observed in both arrays."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError(
            f"Correlation inputs must have the same (channels, points) shape, got {a.shape} and {b.shape}."
        )
    out = np.full(len(a), np.nan)
    for channel in range(len(a)):
        x, y = a[channel], b[channel]
        finite = np.isfinite(x) & np.isfinite(y)
        x, y = x[finite], y[finite]
        if len(x) >= 2 and np.std(x) > 1e-9 and np.std(y) > 1e-9:
            out[channel] = np.corrcoef(x, y)[0, 1]
    return out


def _row_rmse(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """RMSE per channel over samples observed in both arrays."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError(f"RMSE inputs must have the same (channels, points) shape, got {a.shape} and {b.shape}.")
    out = np.full(len(a), np.nan)
    for channel in range(len(a)):
        finite = np.isfinite(a[channel]) & np.isfinite(b[channel])
        if finite.any():
            difference = a[channel, finite] - b[channel, finite]
            out[channel] = np.sqrt(np.mean(difference**2))
    return out


def _pairs(n):
    """Index pairs (i, j), i < j, or empty when there is nothing to pair."""
    return np.triu_indices(n, k=1) if n >= 2 else (np.array([], int), np.array([], int))


def pairwise_rmse(cycles: np.ndarray) -> np.ndarray:
    """RMSE between every pair of cycles ``(n, C, points)`` -> ``(n_pairs, C)``."""
    cycles = np.asarray(cycles, dtype=float)
    i, j = _pairs(len(cycles))
    if not len(i):
        return np.empty((0, cycles.shape[1] if cycles.ndim == 3 else 0))
    return np.stack([_row_rmse(cycles[first], cycles[second]) for first, second in zip(i, j, strict=True)])


def pairwise_corr(cycles: np.ndarray) -> np.ndarray:
    """Pearson r between every pair of cycles ``(n, C, points)`` -> ``(n_pairs, C)``."""
    cycles = np.asarray(cycles, dtype=float)
    i, j = _pairs(len(cycles))
    if not len(i):
        return np.empty((0, cycles.shape[1] if cycles.ndim == 3 else 0))
    return np.stack([_row_correlations(cycles[first], cycles[second]) for first, second in zip(i, j, strict=True)])


def pairwise_rsa(cycles: np.ndarray) -> np.ndarray:
    """Spearman RSA between every pair of cycles ``(n, C, points)`` -> ``(n_pairs,)``.

    The RSA analogue of :func:`pairwise_corr`: each individual repetition is reduced to
    its own electrode x electrode RSM and the pairs of RSMs are compared, so this is how
    much the *coordination* varies from one repetition to the next.
    """
    cycles = np.asarray(cycles, dtype=float)
    i, j = _pairs(len(cycles))
    if not len(i):
        return np.empty(0)
    rsms = [compute_rsm(cycle) for cycle in cycles]
    scores = [rsa(rsms[a], rsms[b])[0] for a, b in zip(i, j, strict=True)]
    return np.asarray([score for score in scores if np.isfinite(score)], dtype=float)


def split_half(cycles: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Average the odd and the even repetitions separately -> two average cycles.

    The reference for a score computed on a trial's average cycle has to be built
    the same way, otherwise it is not comparable: a single repetition is much
    noisier than an average of twenty, so scoring one repetition against another
    would understate what the subject's own data can achieve and flatter the
    policy. Splitting alternately rather than front/back keeps any drift over the
    course of the trial in both halves.

    Returns ``(odd_mean, even_mean)``, or ``None`` with fewer than two
    repetitions.
    """
    cycles = np.asarray(cycles, dtype=float)
    if len(cycles) < 2:
        return None
    # Equal sample counts make the two reliability estimates comparable.
    paired = cycles[: 2 * (len(cycles) // 2)]
    return _finite_mean(paired[0::2], axis=0), _finite_mean(
        paired[1::2],
        axis=0,
    )


def spearman_brown(r):
    """Reliability of a full-length average, from the correlation of its two halves.

    Each half averages only half the repetitions, so it carries ~sqrt(2) more noise
    than the full average the policy is scored against; ``2r / (1 + r)`` is the
    standard step up to full length. Applied only to positive correlations, where
    the formula is meaningful.
    """
    r = np.asarray(r, dtype=float)
    return np.where(r > 0, 2.0 * r / (1.0 + r), r)


def trial_metrics(
    trial: TrialData,
    groups: Sequence[MuscleGroup],
    joint_selection: Sequence[int],
) -> TrialMetrics:
    """Reduce one trial to its average cycles and score them.

    Joint metrics use every detected reference repetition. Muscle metrics use
    the subset fully covered by all required EMG channels; policy activation
    and EMG always use that identical subset.

    Muscles are handled per **group**: the group's heads are collapsed into their
    mean (see :func:`combine_groups`) and it is that mean which is scored against the
    group's electrode -- one score per electrode rather than one per head.

    The returned dict holds both the scores summarised by the bar charts and the
    average cycles (mean and std) the waveform figures draw, so a trial is only
    segmented and averaged once even when several policies are compared.
    """
    boundaries = trial.cycle_boundaries
    ref_qpos, policy_qpos = trial.reference_qpos, trial.policy_qpos
    act_cycles, emg_cycles = trial.activation_cycles, trial.emg_cycles
    if boundaries is None or act_cycles is None or emg_cycles is None or not len(emg_cycles):
        raise ValueError("trial_metrics requires segmented joint and paired muscle cycles.")

    ref_mean, ref_std, _ = average_cycles(
        ref_qpos[:, joint_selection],
        boundaries,
    )
    pol_mean, pol_std, _ = average_cycles(
        policy_qpos[:, joint_selection],
        boundaries,
    )
    combo_cycles = combine_groups(act_cycles, groups)
    act_mean, act_std = _cycle_mean_std(combo_cycles)
    emg_mean, emg_std = _cycle_mean_std(emg_cycles)

    # --- Tracking scores, computed ON the repetition-averaged cycle ---
    # The trial is first reduced to one average cycle per signal (exactly what the
    # waveform figures show) and the score is computed on those averages -- not one
    # score per repetition averaged afterwards, which is a different and much
    # noisier quantity. A trial therefore contributes a single number per joint /
    # per electrode, so the bars carry no repetition whisker.
    #
    # ``*_by`` keeps one column per joint / per muscle group (the per-channel
    # figures); the plain key is the same score averaged across them. The leading
    # axis of length 1 is what the summary code aggregates over.
    metrics: TrialMetrics = {
        "n_joint_cycles": len(boundaries),
        "n_muscle_cycles": len(emg_cycles),
    }
    metrics["joint_rmse_by"] = _row_rmse(pol_mean, ref_mean)[None, :]
    metrics["joint_rmse"] = np.array([_finite_mean(metrics["joint_rmse_by"])])
    corr = _row_correlations(act_mean, emg_mean)
    if not np.isnan(corr).all():
        metrics["muscle_corr_by"] = corr[None, :]
        metrics["muscle_corr"] = np.array([_finite_mean(corr)])
    metrics["emg_rsm"] = compute_rsm(emg_mean)
    metrics["act_rsm"] = compute_rsm(act_mean)
    spearman, pearson = rsa(metrics["act_rsm"], metrics["emg_rsm"])
    if np.isfinite(spearman):
        metrics["muscle_rsa"] = np.array([spearman])
        metrics["muscle_rsa_pearson"] = np.array([pearson])

    # --- Within-subject reference: the real data against ITSELF, built the same
    # way. Half the repetitions are averaged against the other half (see
    # :func:`split_half`) and the result stepped up to full length, so it is the
    # agreement the subject's own data reaches when scored exactly like the policy
    # -- the subject-matched ceiling for this trial.
    ref_cycles = resample_cycles(ref_qpos[:, joint_selection], boundaries)
    halves = split_half(ref_cycles)
    if halves is not None:
        rmse = _row_rmse(*halves) * SPLIT_HALF_RMSE_FACTOR
        metrics["joint_rmse_within_by"] = rmse[None, :]
        metrics["joint_rmse_within"] = np.array([_finite_mean(rmse)])
    halves = split_half(emg_cycles)
    if halves is not None:
        within_corr = spearman_brown(_row_correlations(*halves))
        if not np.isnan(within_corr).all():
            metrics["muscle_corr_within_by"] = within_corr[None, :]
            metrics["muscle_corr_within"] = np.array([_finite_mean(within_corr)])
        # RSA is already a correlation between derived matrices, so it does
        # not receive the Spearman-Brown correction used above.
        within_rsa, _ = rsa(compute_rsm(halves[0]), compute_rsm(halves[1]))
        if np.isfinite(within_rsa):
            metrics["muscle_rsa_within"] = np.array([within_rsa])

    # --- Within-subject variability: this subject's *individual* repetitions against
    # each other, every pair, averaged over the pairs. Unlike the split-half ceiling
    # above this is a single-repetition quantity -- how far one repetition of the same
    # person is from another -- so it is looser than the ceiling the policy bars (scored
    # on the average of every repetition) are actually up against. Averaged over
    # subjects by :func:`compute_within_subject_variability`.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pair_rmse = pairwise_rmse(ref_cycles)
        if len(pair_rmse):
            metrics["joint_rmse_pairwise_by"] = _finite_mean(
                pair_rmse,
                axis=0,
            )
        pair_corr = pairwise_corr(emg_cycles)
        if len(pair_corr) and not np.isnan(pair_corr).all():
            metrics["muscle_corr_pairwise_by"] = _finite_mean(
                pair_corr,
                axis=0,
            )
        pair_rsa = pairwise_rsa(emg_cycles)
        if len(pair_rsa):
            metrics["muscle_rsa_pairwise_by"] = np.array([_finite_mean(pair_rsa)])

    # Average cycles: drawn by the waveform figures, averaged over subjects by the
    # aggregated figures, and compared across subjects by the between-subject reference.
    metrics["ref_cycle"], metrics["ref_std"] = ref_mean, ref_std
    metrics["pol_cycle"], metrics["pol_std"] = pol_mean, pol_std
    metrics["head_means"] = act_cycles.mean(axis=0)  # (n_muscles, 100), thin head lines
    metrics["act_cycle"], metrics["act_std"] = act_mean, act_std
    metrics["emg_cycle"], metrics["emg_std"] = emg_mean, emg_std
    return metrics
