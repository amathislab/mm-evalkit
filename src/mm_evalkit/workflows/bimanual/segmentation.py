"""Movement-cycle segmentation for ULTra-MoCap bimanual trials.

Each ULTra-MoCap recording is ~50 s of a single movement repeated many times
(e.g. a dozen shoulder rotations). To average the muscle / joint pattern over
*one* canonical movement, the individual repetitions first have to be isolated.

This module segments a trial into repetitions from the periodicity of a
movement-specific "critical" joint angle -- the joint whose oscillation defines
the movement (elbow flexion angle for Elbow Flexion, shoulder rotation for
Shoulder Rotation, shoulder elevation for the reaches / arm swing). Peaks of
that (smoothed) angle mark the repetition boundaries; each repetition is then
time-normalized to a common length so cycles can be averaged (mean +/- std).

Typical use::

    selected = select_declared_critical_joint(
        ref_qpos, joint_names, "ShoulderRotation"
    )
    boundaries = selected[3]
    mean, std, n = average_cycles(muscle_data, boundaries)  # n = number of reps
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, savgol_filter

# Declared coordinate that defines one repetition for each movement.
MOVEMENT_CRITICAL_JOINT = {
    "ArmSwing": "shoulder_elv",
    "CrossbodyReach": "shoulder_elv",
    "ElbowFlexion": "elbow_flex",
    "ShoulderRotation": "shoulder_rot",
    "OverheadReach": "shoulder_elv",
}


def movement_from_trial_stem(stem: str) -> str:
    """Movement token from a trial stem, e.g. ``P11_ArmSwing_Slow`` -> ``ArmSwing``."""
    parts = stem.split("_")
    return parts[1] if len(parts) > 1 else stem


def condition_from_trial_stem(stem: str) -> str:
    """Condition token from a trial stem, e.g. ``P11_ArmSwing_Slow`` -> ``Slow``."""
    parts = stem.split("_")
    return parts[2] if len(parts) > 2 else ""


def critical_joint_name(movement: str, arm: str = "right") -> str | None:
    """Return the declared model coordinate for one movement and arm."""
    base = MOVEMENT_CRITICAL_JOINT.get(movement)
    if base is None:
        return None
    return f"{base}_{'l' if arm == 'left' else 'r'}"


def estimate_period_samples(
    x: np.ndarray, fs: float, min_period_s: float = 0.6, max_period_s: float = 20.0
) -> int | None:
    """Dominant period of ``x`` (in samples) via autocorrelation, or None.

    The first autocorrelation peak within [min_period_s, max_period_s] gives the
    repetition period without assuming a fixed movement speed. Returns None for a
    flat signal or when the search window is empty (too short a recording).
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    if np.allclose(x, 0.0):
        return None
    autocorr = np.correlate(x, x, mode="full")[len(x) - 1 :]
    lo = max(1, int(min_period_s * fs))
    hi = min(int(max_period_s * fs), len(autocorr) - 1)
    if hi <= lo:
        return None
    return lo + int(np.argmax(autocorr[lo:hi]))


def _smooth(x: np.ndarray, fs: float) -> np.ndarray:
    """Light Savitzky-Golay smoothing (~150 ms window) to steady peak detection."""
    x = np.asarray(x, dtype=float)
    window = max(5, int(0.15 * fs))
    window += 1 - (window % 2)  # force odd
    if len(x) <= window:
        return x
    return savgol_filter(x, window, 2)


def segment_cycles(
    critical_signal: np.ndarray,
    fs: float = 100.0,
    gap_fraction: float = 0.6,
    prominence_fraction: float = 0.25,
) -> list[tuple[int, int]]:
    """Repetition boundaries from the periodicity of ``critical_signal``.

    The signal is smoothed, its dominant period estimated, and peaks at least
    ``gap_fraction`` * period apart (and prominent enough relative to the signal
    range) are taken as cycle delimiters. Returns one ``(start, end)`` index pair
    per repetition (consecutive peak to peak), i.e. ``len(...)`` == number of
    complete movements found. Empty if fewer than two peaks are detected (e.g. an
    early-terminated rollout that never completes a full movement).
    """
    x = _smooth(critical_signal, fs)
    period = estimate_period_samples(x, fs)
    if period is None:
        return []
    signal_range = float(np.nanmax(x) - np.nanmin(x))
    if signal_range <= 0:
        return []
    peaks, _ = find_peaks(
        x,
        distance=max(1, int(gap_fraction * period)),
        prominence=prominence_fraction * signal_range,
    )
    if len(peaks) < 2:
        return []
    return [(int(peaks[i]), int(peaks[i + 1])) for i in range(len(peaks) - 1)]


def periodicity_strength(x: np.ndarray, fs: float) -> tuple[float, int | None]:
    """How periodic ``x`` is: normalized autocorrelation at its dominant period.

    Returns ``(score, period_samples)`` where ``score`` in [0, 1] is the
    autocorrelation at the estimated period divided by the zero-lag energy (1 =
    perfectly periodic). A flat or aperiodic signal scores ~0.
    """
    x = _smooth(x, fs)
    x = np.asarray(x, dtype=float) - np.mean(x)
    period = estimate_period_samples(x, fs)
    energy = float(np.dot(x, x))
    if period is None or period >= len(x) or energy <= 0:
        return 0.0, None
    score = float(np.dot(x[:-period], x[period:])) / energy
    return max(0.0, score), period


def cycle_shape_consistency(signal: np.ndarray, boundaries: list[tuple[int, int]]) -> float:
    """How well the segmented cycles of ``signal`` superimpose, in [-1, 1].

    Each time-normalized cycle is correlated with the mean of the *others* (so a
    cycle is never compared against itself) and the correlations are averaged. Near
    1 means the boundaries really do delimit repetitions of one movement; a low
    value means the segmentation is mixing different things together, even if the
    cycles happen to have similar durations. Returns 0.0 for fewer than 2 cycles.
    """
    cycles = resample_cycles(np.asarray(signal, dtype=float)[:, None], boundaries)
    if len(cycles) < 2:
        return 0.0
    scores = []
    for k in range(len(cycles)):
        others = np.delete(cycles, k, axis=0).mean(axis=0)[0]
        this = cycles[k, 0]
        if np.std(this) > 1e-12 and np.std(others) > 1e-12:
            scores.append(float(np.corrcoef(this, others)[0, 1]))
    return float(np.mean(scores)) if scores else 0.0


def select_declared_critical_joint(
    qpos_deg: np.ndarray,
    joint_names: list[str],
    movement: str,
    *,
    arm: str = "right",
    fs: float = 100.0,
    min_range_deg: float = 8.0,
    max_duration_cv: float = 0.25,
    min_cycles: int = 3,
    min_shape_consistency: float = 0.8,
):
    """Segment the coordinate declared for ``movement``, or return ``None``.

    Automatically switching coordinates changes the definition of a
    repetition between trials. The declared signal must have sufficient range,
    at least ``min_cycles`` cycles, consistent durations, and consistent
    time-normalized shapes.
    """
    name = critical_joint_name(movement, arm)
    if name is None:
        raise ValueError(f"No critical joint is declared for movement {movement!r}.")
    if name not in joint_names:
        raise ValueError(f"Movement {movement!r} requires critical joint {name!r}, which is absent from the rollout.")

    signal = np.asarray(qpos_deg[:, joint_names.index(name)], dtype=float)
    if np.ptp(signal) < min_range_deg:
        return None
    boundaries = segment_cycles(signal, fs)
    if len(boundaries) < min_cycles:
        return None

    durations = np.asarray(
        [end - start for start, end in boundaries],
        dtype=float,
    )
    if durations.mean() <= 0:
        return None
    if durations.std() / durations.mean() > max_duration_cv:
        return None
    if cycle_shape_consistency(signal, boundaries) < min_shape_consistency:
        return None

    score, period = periodicity_strength(signal, fs)
    return name, score, period, boundaries


def complete_cycle_boundaries(
    data: np.ndarray,
    boundaries: list[tuple[int, int]],
    *,
    require_finite: bool = True,
) -> list[tuple[int, int]]:
    """Return boundaries fully covered by ``data``.

    A boundary crossing the end of a shorter modality is incomplete and must
    not be stretched into a full repetition.  When ``require_finite`` is true,
    cycles containing missing samples are also excluded; timestamp-aligned EMG
    uses those NaNs to represent portions that were never recorded.
    """
    values = np.asarray(data)
    valid = []
    for start, end in boundaries:
        if start < 0 or end <= start or end > len(values):
            continue
        segment = values[start:end]
        if len(segment) < 2:
            continue
        if require_finite and not np.isfinite(segment).all():
            continue
        valid.append((start, end))
    return valid


def resample_cycles(
    data: np.ndarray,
    boundaries: list[tuple[int, int]],
    n_points: int = 100,
    *,
    require_finite: bool = True,
) -> np.ndarray:
    """Time-normalize each cycle of ``data`` (T, C) to ``(n_cycles, C, n_points)``.

    Only cycles fully covered by the signal are retained. Incomplete or missing
    EMG is never stretched into an apparently complete repetition.
    """
    data = np.asarray(data, dtype=float)
    n_channels = data.shape[1]
    query = np.linspace(0.0, 1.0, n_points)
    cycles = []
    valid_boundaries = complete_cycle_boundaries(data, boundaries, require_finite=require_finite)
    for start, end in valid_boundaries:
        segment = data[start:end]
        base = np.linspace(0.0, 1.0, len(segment))
        cycles.append(np.stack([np.interp(query, base, segment[:, c]) for c in range(n_channels)]))
    if not cycles:
        return np.empty((0, n_channels, n_points))
    return np.asarray(cycles)


def average_cycles(
    data: np.ndarray, boundaries: list[tuple[int, int]], n_points: int = 100
) -> tuple[np.ndarray, np.ndarray, int]:
    """Mean, std (both ``(C, n_points)``) and cycle count over the resampled cycles."""
    cycles = resample_cycles(data, boundaries, n_points)
    n_channels = np.asarray(data).shape[1]
    if len(cycles) == 0:
        nan = np.full((n_channels, n_points), np.nan)
        return nan, nan.copy(), 0
    return cycles.mean(axis=0), cycles.std(axis=0), len(cycles)


def plot_average_cycles(
    channel_names: list[str],
    series: list[tuple[str, str, np.ndarray, np.ndarray]],
    title: str,
    out_path: str,
    ylabel: str = "",
    twin_series: list[tuple[str, str, np.ndarray, np.ndarray]] | None = None,
    twin_ylabel: str = "",
    per_channel_series: list[list[tuple[str, str, np.ndarray]]] | None = None,
) -> None:
    """One subplot per channel; each ``series`` drawn as mean line + std band.

    ``series`` / ``twin_series`` are lists of ``(label, color, mean, std)`` -- or
    ``(label, color, mean, std, linestyle)`` to override the line style, which is how
    several policies are told apart on one axes -- with ``mean``/``std`` shaped
    ``(len(channel_names), n_points)``. ``twin_series`` is drawn on a right-hand
    y-axis (used to overlay real EMG, whose units differ from the policy activation
    on the left axis). The x-axis is the normalized movement cycle (0-100%).

    ``per_channel_series`` is one list of ``(label, color, mean)`` per channel,
    each a single 1-D trace drawn thin and underneath the main series with no band
    -- used to keep the individual muscle heads visible behind the fitted
    combination of their group, where the traces differ from subplot to subplot.
    Subplots that have them get their own legend.
    """
    n = len(channel_names)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.7 * nrows), squeeze=False)
    n_points = series[0][2].shape[1] if series else 100
    phase = np.linspace(0, 100, n_points)

    for idx, name in enumerate(channel_names):
        ax = axes[idx // ncols][idx % ncols]
        handles, faint_handles = [], []
        faint = per_channel_series[idx] if per_channel_series and idx < len(per_channel_series) else []
        for label, color, values in faint:
            (line,) = ax.plot(phase, values, color=color, lw=0.9, alpha=0.9, zorder=1, label=label)
            faint_handles.append(line)
        for spec in series:
            label, color, mean, std = spec[:4]
            linestyle = spec[4] if len(spec) > 4 else "-"
            (line,) = ax.plot(phase, mean[idx], color=color, lw=1.6, ls=linestyle, zorder=3, label=label)
            ax.fill_between(
                phase,
                mean[idx] - std[idx],
                mean[idx] + std[idx],
                color=color,
                alpha=0.2 if len(series) < 3 else 0.12,
                zorder=2,
            )
            handles.append(line)
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=7)
        if idx % ncols == 0:
            ax.set_ylabel(ylabel, fontsize=8)
        if idx // ncols == nrows - 1:
            ax.set_xlabel("movement cycle (%)", fontsize=8)

        if twin_series is not None:
            ax_twin = ax.twinx()
            for spec in twin_series:
                label, color, mean, std = spec[:4]
                linestyle = spec[4] if len(spec) > 4 else "--"
                (line,) = ax_twin.plot(phase, mean[idx], color=color, lw=1.6, ls=linestyle, label=label)
                ax_twin.fill_between(phase, mean[idx] - std[idx], mean[idx] + std[idx], color=color, alpha=0.15)
                handles.append(line)
            ax_twin.tick_params(axis="y", labelsize=7, labelcolor="0.4")
            if idx % ncols == ncols - 1 or idx == n - 1:
                ax_twin.set_ylabel(twin_ylabel, fontsize=8, color="0.4")

        if idx == 0 or faint_handles:
            entries = (handles if idx == 0 else []) + faint_handles
            ax.legend(entries, [h.get_label() for h in entries], fontsize=6.5, loc="best")

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
