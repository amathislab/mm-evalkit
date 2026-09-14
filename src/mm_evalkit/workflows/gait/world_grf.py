"""Read interval-labelled world-frame foot forces from policy exports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from mm_evalkit.arrays import FloatArray

_METADATA = (
    "grf_coordinate_frame",
    "grf_units",
    "grf_sides",
    "grf_sample_phase",
    "grf_control_reduction",
    "gravity_m_s2",
    "model_mass_kg",
)


def load_world_grf(archive: Any, prefix: str, rows: int, source: str | Path) -> FloatArray | None:
    """Return vertical GRF [BW], shape [rows, left/right], or None for legacy files.

    Samples retain the last native solver interval of each control step, whose
    end maps to ``kinetics_reference_steps``. Vertical means opposite the saved
    model gravity. Invalid or partial new-format data never falls back to touch
    sensors. Raw physics-frequency samples remain available in the export for
    impulse, impact-peak and contact-event analyses.
    """
    force_key = f"{prefix}_ground_reaction_forces"
    if not any(key.startswith("grf_") for key in archive) and force_key not in archive:
        return None
    start_key, end_key = (f"{prefix}_kinetics_interval_{bound}_times" for bound in ("start", "end"))
    missing = [key for key in (*_METADATA, force_key, start_key, end_key) if key not in archive]
    if missing:
        raise ValueError(f"World GRF export {source} is missing fields: {missing}.")
    expected = {
        "grf_coordinate_frame": "world",
        "grf_units": "N",
        "grf_sample_phase": "solver_interval",
        "grf_control_reduction": "last_physics_step",
    }
    for key, value in expected.items():
        actual = np.asarray(archive[key])
        if actual.shape != () or actual.item() != value:
            raise ValueError(f"World GRF {key} must be {value!r} in {source}.")
    sides = np.asarray(archive["grf_sides"])
    if sides.shape != (2,) or set(sides.tolist()) != {"left", "right"}:
        raise ValueError(f"World GRF grf_sides must contain left and right exactly once in {source}.")
    gravity = np.asarray(archive["gravity_m_s2"], dtype=float)
    mass = np.asarray(archive["model_mass_kg"], dtype=float)
    if gravity.shape != (3,) or not np.isfinite(gravity).all() or np.linalg.norm(gravity) <= 0:
        raise ValueError(f"World GRF requires finite nonzero gravity_m_s2 in {source}.")
    if mass.shape != () or not np.isfinite(mass) or mass <= 0:
        raise ValueError(f"World GRF requires positive model_mass_kg in {source}.")
    force = np.asarray(archive[force_key], dtype=float)
    if force.shape != (rows, 2, 3) or not np.isfinite(force).all():
        raise ValueError(f"World GRF {force_key} must be finite with shape {(rows, 2, 3)} in {source}.")
    start, end = (np.asarray(archive[key], dtype=float) for key in (start_key, end_key))
    if (
        start.shape != (rows,)
        or end.shape != (rows,)
        or not np.isfinite(start).all()
        or not np.isfinite(end).all()
        or np.any(start >= end)
        or np.any(np.diff(end) <= 0)
        or np.any(start[1:] < end[:-1])
    ):
        raise ValueError(f"World GRF solver intervals must be finite, ordered and non-overlapping in {source}.")
    magnitude = np.linalg.norm(gravity)
    vertical = force @ (-gravity / magnitude)
    order = [sides.tolist().index(side) for side in ("left", "right")]
    return vertical[:, order] / (float(mass) * magnitude)
