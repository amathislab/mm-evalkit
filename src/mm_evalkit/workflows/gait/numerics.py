"""Small numerical primitives shared by the full-body metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _finite_pairs(
    first: ArrayLike,
    second: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Flatten equally shaped inputs and retain samples finite in both."""
    first_array = np.asarray(first, dtype=float)
    second_array = np.asarray(second, dtype=float)
    if first_array.shape != second_array.shape:
        raise ValueError(f"Paired metrics require equal shapes, got {first_array.shape} and {second_array.shape}.")
    first_flat = first_array.ravel()
    second_flat = second_array.ravel()
    valid = np.isfinite(first_flat) & np.isfinite(second_flat)
    return first_flat[valid], second_flat[valid]


def finite_mean(
    values: ArrayLike,
    axis: int | tuple[int, ...] | None = None,
):
    """Average finite values, returning NaN where none exist."""
    array = np.asarray(values, dtype=float)
    valid = np.isfinite(array)
    if axis is None:
        finite = array[valid]
        return float(np.mean(finite)) if finite.size else float("nan")

    count = np.sum(valid, axis=axis)
    total = np.sum(np.where(valid, array, 0.0), axis=axis)
    return np.divide(
        total,
        count,
        out=np.full(np.shape(total), np.nan, dtype=float),
        where=count > 0,
    )


def paired_pearson(first: ArrayLike, second: ArrayLike) -> float:
    """Pearson correlation over finite pairs, or NaN when undefined."""
    first_finite, second_finite = _finite_pairs(first, second)
    if first_finite.size < 2:
        return float("nan")
    first_centered = first_finite - np.mean(first_finite)
    second_centered = second_finite - np.mean(second_finite)
    denominator = np.linalg.norm(first_centered) * np.linalg.norm(second_centered)
    if denominator == 0.0:
        return float("nan")
    return float(
        np.clip(
            np.dot(first_centered, second_centered) / denominator,
            -1.0,
            1.0,
        )
    )


def paired_rmse(first: ArrayLike, second: ArrayLike) -> float:
    """RMSE over finite paired samples, or NaN when none remain."""
    first_finite, second_finite = _finite_pairs(first, second)
    if first_finite.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(first_finite - second_finite))))
