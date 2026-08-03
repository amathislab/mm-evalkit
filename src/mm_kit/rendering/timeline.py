"""Small alignment helpers shared by dataset render adapters."""

from __future__ import annotations

import numpy as np


def normalize_shape(values: np.ndarray) -> np.ndarray:
    """Independently min-max normalize one trace for shape comparison."""
    signal = np.asarray(values, dtype=float)
    finite = signal[np.isfinite(signal)]
    if not finite.size or float(np.ptp(finite)) < 1e-12:
        return np.zeros_like(signal)
    return np.clip((signal - float(finite.min())) / float(np.ptp(finite)), 0.0, 1.0)


def resample_profile(values: np.ndarray, n_samples: int) -> np.ndarray:
    """Interpolate a one-dimensional profile onto a target phase grid."""
    signal = np.asarray(values, dtype=float)
    if signal.ndim != 1 or len(signal) < 2 or not np.isfinite(signal).all():
        raise ValueError("A resampled profile must be a finite vector with at least two samples.")
    if n_samples < 2:
        raise ValueError("n_samples must be at least two.")
    return np.interp(
        np.linspace(0.0, 1.0, n_samples),
        np.linspace(0.0, 1.0, len(signal)),
        signal,
    )


def interpolate_row(values: np.ndarray, position: float) -> np.ndarray:
    """Linearly interpolate a frame matrix at a fractional sample position."""
    matrix = np.asarray(values, dtype=float)
    low = min(int(np.floor(position)), len(matrix) - 1)
    high = min(low + 1, len(matrix) - 1)
    blend = position - low
    return (1.0 - blend) * matrix[low] + blend * matrix[high]
