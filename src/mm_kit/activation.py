"""Align exported muscle activation with the rollout observation timeline."""

from __future__ import annotations

from typing import Any

import numpy as np

PRE_STEP = "pre_step"
POST_STEP = "post_step"
_VALID_PHASES = frozenset({PRE_STEP, POST_STEP})


def activation_sample_phase(archive: Any) -> str:
    """Return the activation timing declared by a rollout archive.

    Historical exports omitted this metadata and sampled ``data.act`` directly
    after ``env.step``. Missing metadata therefore has one documented meaning:
    ``post_step``.
    """
    key = "activation_sample_phase"
    if key not in archive:
        return POST_STEP

    value = np.asarray(archive[key])
    if value.size != 1:
        raise ValueError(f"Rollout field {key!r} must contain one scalar value.")
    phase = value.item()
    if not isinstance(phase, (str, np.str_)) or str(phase) not in _VALID_PHASES:
        choices = ", ".join(sorted(_VALID_PHASES))
        raise ValueError(f"Rollout field {key!r} must be one of {choices}; got {phase!r}.")
    return str(phase)


def align_activation_to_observation_timeline(
    activation: np.ndarray,
    phase: str,
) -> np.ndarray:
    """Place activation rows on the pre-step state timeline.

    Current ``pre_step`` exports already align with qpos. Legacy ``post_step``
    row ``i`` belongs to state ``i + 1``; state zero is unobserved and remains
    NaN, while the final post-step row has no exported state and is discarded.
    """
    values = np.asarray(activation, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"Muscle activation must have shape (frames, muscles); got {values.shape}.")
    if phase == PRE_STEP:
        return values.copy()
    if phase != POST_STEP:
        choices = ", ".join(sorted(_VALID_PHASES))
        raise ValueError(f"Activation sample phase must be one of {choices}; got {phase!r}.")

    aligned = np.full(values.shape, np.nan, dtype=float)
    if len(values) > 1:
        aligned[1:] = values[:-1]
    return aligned


def complete_activation_gait_indexes(
    activation: np.ndarray,
    gait_indexes: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Keep gait intervals fully covered by finite aligned activation."""
    values = np.asarray(activation, dtype=float)
    retained: list[tuple[int, int]] = []
    for start, end in gait_indexes:
        if start < 0 or end <= start or end > len(values):
            continue
        segment = values[start:end]
        if len(segment) >= 2 and np.isfinite(segment).all():
            retained.append((start, end))
    return retained
