"""Strict reader for Gait120 inverse-dynamics and force-plate references.

Kinetic validation compares model actuator moments with OpenSim inverse
dynamics, and model contact forces with corrected Gait120 force-plate vertical
ground-reaction forces. The reference files contain full, time-indexed trials
rather than the already phase-normalized Step01/Step02 profiles used by
:mod:`mm_evalkit.datasets.gait120`. This module owns that dataset schema and
converts every retained left-heel-strike window to the common 100-point gait
grid.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mm_evalkit.arrays import FloatArray

TORQUE_JOINTS = (
    "hip_flexion_r",
    "hip_flexion_l",
    "knee_angle_r",
    "knee_angle_l",
    "ankle_angle_r",
    "ankle_angle_l",
)
GRF_CHANNELS = ("left", "right")
TORQUE_UNIT = "Nm/kg"
GRF_UNIT = "BW"

_SUBJECT = re.compile(r"^S(?P<subject>\d{3})$")
_TRIAL = re.compile(r"^Trial(?P<trial>\d{2})$")
_NORMALIZED_TORQUE_UNITS = {
    "nm_per_kg",
    "n_m_per_kg",
    "nm/kg",
    "original_id_units_per_kg",
}
_ABSOLUTE_TORQUE_UNITS = {"nm", "n_m", "newton_meters", "newton_metres"}
_NORMALIZED_GRF_UNITS = {
    "body_weight_normalized",
    "body_weight_units",
    "body_weights",
    "bw",
    "force_columns_in_body_weight;_cop_and_torque_columns_in_corrected_physical_units",
}
_ABSOLUTE_GRF_UNITS = {"n", "newtons"}
_LEFT_GRF_COLUMNS = ("ground_force1_vy", "ground_force_l_vy", "ground_force_left_vy")
_RIGHT_GRF_COLUMNS = ("ground_force2_vy", "ground_force_r_vy", "ground_force_right_vy")


@dataclass(frozen=True, slots=True)
class Gait120KineticsTrial:
    """One validated reference trial and its phase-normalized kinetic cycles.

    Raw signal shapes are ``(n_frames, n_channels)``. Cycle shapes are
    ``(n_cycles, n_channels, 100)``. Moments are normalized to ``Nm/kg`` and
    vertical GRF to multiples of body weight before a gait window is selected.
    """

    path: Path
    subject: int
    trial: int
    time: FloatArray
    torque: FloatArray
    grf: FloatArray
    torque_valid: NDArray[np.bool_]
    grf_valid: NDArray[np.bool_]
    gait_windows: tuple[tuple[int, int], ...]
    torque_cycles: FloatArray
    grf_cycles: FloatArray
    torque_names: tuple[str, ...] = TORQUE_JOINTS
    grf_names: tuple[str, ...] = GRF_CHANNELS
    torque_unit: str = TORQUE_UNIT
    grf_unit: str = GRF_UNIT


def _scalar_text(data: Any, key: str, path: Path) -> str:
    if key not in data:
        raise ValueError(f"Gait120 kinetics reference is missing {key!r}: {path}")
    value = np.asarray(data[key])
    if value.shape != ():
        raise ValueError(f"Gait120 kinetics {key!r} must be a scalar in {path}, got shape {value.shape}.")
    return str(value.item())


def _columns(data: Any, key: str, path: Path) -> tuple[str, ...]:
    if key not in data:
        raise ValueError(f"Gait120 kinetics reference is missing {key!r}: {path}")
    values = np.asarray(data[key])
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"Gait120 kinetics {key!r} must be a non-empty vector in {path}.")
    columns = tuple(str(value) for value in values.tolist())
    if len(columns) != len(set(columns)):
        raise ValueError(f"Gait120 kinetics {key!r} contains duplicate names in {path}.")
    return columns


def _matrix(data: Any, key: str, path: Path) -> FloatArray:
    if key not in data:
        raise ValueError(f"Gait120 kinetics reference is missing {key!r}: {path}")
    values = np.asarray(data[key], dtype=float)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError(f"Gait120 kinetics {key!r} must be a 2D array with at least two rows in {path}.")
    return values


def _valid_rows(data: Any, key: str, n_frames: int, path: Path) -> NDArray[np.bool_]:
    if key not in data:
        return np.ones(n_frames, dtype=bool)
    valid = np.asarray(data[key])
    if valid.dtype.kind != "b" or valid.shape != (n_frames,):
        raise ValueError(f"Gait120 kinetics {key!r} must be a boolean vector of length {n_frames} in {path}.")
    return valid.astype(bool, copy=False)


def _subject_mass_kg(data: Any, path: Path) -> float:
    if "subject_weight_kg" in data:
        mass = float(np.asarray(data["subject_weight_kg"]).item())
    elif "subject_body_weight_n" in data:
        mass = float(np.asarray(data["subject_body_weight_n"]).item()) / 9.80665
    else:
        raise ValueError(f"Absolute kinetic units require subject_weight_kg or subject_body_weight_n in {path}.")
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError(f"Gait120 kinetics subject mass must be finite and positive in {path}, got {mass}.")
    return mass


def _normalize_torque(values: FloatArray, units: str, data: Any, path: Path) -> FloatArray:
    normalized = units.strip().casefold().replace(" ", "_")
    if normalized in _NORMALIZED_TORQUE_UNITS:
        return values
    if normalized in _ABSOLUTE_TORQUE_UNITS:
        return values / _subject_mass_kg(data, path)
    raise ValueError(f"Unsupported Gait120 joint_torque_units {units!r} in {path}; expected moments in Nm or Nm/kg.")


def _normalize_grf(values: FloatArray, units: str, data: Any, path: Path) -> FloatArray:
    normalized = units.strip().casefold().replace(" ", "_")
    if normalized in _NORMALIZED_GRF_UNITS:
        return values
    if normalized in _ABSOLUTE_GRF_UNITS:
        body_weight = _subject_mass_kg(data, path) * 9.80665
        return values / body_weight
    raise ValueError(
        f"Unsupported Gait120 computed_grf_units {units!r} in {path}; expected forces in N or body weights."
    )


def _required_indices(columns: tuple[str, ...], requested: tuple[str, ...], *, signal: str, path: Path) -> list[int]:
    missing = [name for name in requested if name not in columns]
    if missing:
        raise ValueError(f"Gait120 kinetics {signal} is missing required channel(s) in {path}: {', '.join(missing)}.")
    return [columns.index(name) for name in requested]


def _grf_index(columns: tuple[str, ...], candidates: tuple[str, ...], side: str, path: Path) -> int:
    present = [columns.index(name) for name in candidates if name in columns]
    if len(present) != 1:
        names = ", ".join(candidates)
        raise ValueError(
            f"Gait120 kinetics GRF must contain exactly one {side} vertical-force column ({names}) in {path}."
        )
    return present[0]


def left_heel_strike_windows(
    vertical_grf: FloatArray,
    *,
    window_frames: int = 100,
    contact_threshold: float = 0.05,
) -> tuple[tuple[int, int], ...]:
    """Return complete fixed windows beginning at each left heel strike.

    Fixed 100-frame windows are the declared alignment contract shared by the
    kinetic reference and policy-summary exporters. The duration and contact
    threshold are explicit and validated here.
    """
    if window_frames < 2:
        raise ValueError(f"window_frames must be at least 2, got {window_frames}.")
    if not np.isfinite(contact_threshold) or contact_threshold < 0:
        raise ValueError(f"contact_threshold must be finite and non-negative, got {contact_threshold}.")
    left = np.asarray(vertical_grf, dtype=float) > contact_threshold
    starts = [index for index in range(1, len(left)) if not left[index - 1] and left[index]]
    if left.size and left[0]:
        starts.insert(0, 0)
    return tuple((start, start + window_frames) for start in starts if start + window_frames <= len(left))


def phase_normalize_window(
    values: FloatArray,
    valid: NDArray[np.bool_],
    window: tuple[int, int],
    *,
    n_samples: int = 100,
) -> FloatArray:
    """Interpolate one time-preserving window to ``(channels, n_samples)``.

    Invalid rows retain their original phase coordinates; removing them does
    not compress the remaining samples in time. A channel with fewer than two
    finite valid points remains undefined.
    """
    start, end = window
    segment = np.asarray(values[start:end], dtype=float)
    valid_segment = np.asarray(valid[start:end], dtype=bool)
    source = np.linspace(0.0, 1.0, segment.shape[0])
    target = np.linspace(0.0, 1.0, n_samples)
    output = np.full((segment.shape[1], n_samples), np.nan, dtype=float)
    for channel in range(segment.shape[1]):
        finite = valid_segment & np.isfinite(segment[:, channel])
        if np.count_nonzero(finite) < 2:
            continue
        first = float(source[finite][0])
        last = float(source[finite][-1])
        covered = (target >= first) & (target <= last)
        output[channel, covered] = np.interp(target[covered], source[finite], segment[finite, channel])
    return output


def load_gait120_kinetics_trial(
    npz_path: str | Path,
    *,
    n_samples: int = 100,
    window_frames: int = 100,
    contact_threshold: float = 0.05,
) -> Gait120KineticsTrial:
    """Load one ``S###_Trial##_Step01_Step02.npz`` kinetics reference."""
    path = Path(npz_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Gait120 kinetics reference does not exist: {path}")

    with np.load(path, allow_pickle=False) as data:
        subject_text = _scalar_text(data, "subject", path)
        trial_text = _scalar_text(data, "trial", path)
        subject_match = _SUBJECT.fullmatch(subject_text)
        trial_match = _TRIAL.fullmatch(trial_text)
        if subject_match is None or trial_match is None:
            raise ValueError(
                f"Gait120 kinetics subject/trial must match S### and Trial## in {path}, "
                f"got {subject_text!r}/{trial_text!r}."
            )
        subject = int(subject_match.group("subject"))
        trial = int(trial_match.group("trial"))
        step_label = _scalar_text(data, "step_label", path)
        if step_label != "Step01_Step02":
            raise ValueError(f"Gait120 kinetics step_label must be 'Step01_Step02' in {path}, got {step_label!r}.")
        expected_stem = f"{subject_text}_{trial_text}_Step01_Step02"
        if path.stem != expected_stem:
            raise ValueError(f"Gait120 kinetics metadata expects filename {expected_stem}.npz, got {path.name}.")

        torque_raw = _matrix(data, "joint_torque", path)
        grf_raw = _matrix(data, "computed_grf", path)
        if torque_raw.shape[0] != grf_raw.shape[0]:
            raise ValueError(
                f"Gait120 kinetics torque and GRF frame counts differ in {path}: "
                f"{torque_raw.shape[0]} vs {grf_raw.shape[0]}."
            )
        n_frames = torque_raw.shape[0]
        torque_columns = _columns(data, "joint_torque_columns", path)
        grf_columns = _columns(data, "computed_grf_columns", path)
        if torque_raw.shape[1] != len(torque_columns) or grf_raw.shape[1] != len(grf_columns):
            raise ValueError(f"Gait120 kinetics data widths do not match their declared columns in {path}.")

        torque_indices = _required_indices(
            torque_columns,
            tuple(f"{name}_moment" for name in TORQUE_JOINTS),
            signal="joint torque",
            path=path,
        )
        left_index = _grf_index(grf_columns, _LEFT_GRF_COLUMNS, "left", path)
        right_index = _grf_index(grf_columns, _RIGHT_GRF_COLUMNS, "right", path)
        torque = _normalize_torque(
            torque_raw[:, torque_indices],
            _scalar_text(data, "joint_torque_units", path),
            data,
            path,
        )
        grf = _normalize_grf(
            grf_raw[:, [left_index, right_index]],
            _scalar_text(data, "computed_grf_units", path),
            data,
            path,
        )
        torque_valid = _valid_rows(data, "joint_torque_valid", n_frames, path)
        grf_valid = _valid_rows(data, "computed_grf_valid", n_frames, path)

        if "time_sec" in data:
            time = np.asarray(data["time_sec"], dtype=float)
            if time.shape != (n_frames,) or not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0):
                raise ValueError(f"Gait120 kinetics time_sec must be finite and strictly increasing in {path}.")
        else:
            time = np.arange(n_frames, dtype=float)

    windows = left_heel_strike_windows(
        grf[:, 0],
        window_frames=window_frames,
        contact_threshold=contact_threshold,
    )
    if not windows:
        raise ValueError(f"Gait120 kinetics reference contains no complete left-heel-strike window: {path}")
    torque_cycles = np.stack(
        [phase_normalize_window(torque, torque_valid, window, n_samples=n_samples) for window in windows]
    )
    grf_cycles = np.stack([phase_normalize_window(grf, grf_valid, window, n_samples=n_samples) for window in windows])
    missing_torque = [
        name for index, name in enumerate(TORQUE_JOINTS) if not np.isfinite(torque_cycles[:, index]).any()
    ]
    missing_grf = [name for index, name in enumerate(GRF_CHANNELS) if not np.isfinite(grf_cycles[:, index]).any()]
    if missing_torque or missing_grf:
        raise ValueError(
            f"Gait120 kinetics reference has no finite samples for required channels in {path}: "
            f"moments={missing_torque}, GRF={missing_grf}."
        )

    return Gait120KineticsTrial(
        path=path,
        subject=subject,
        trial=trial,
        time=time,
        torque=torque,
        grf=grf,
        torque_valid=torque_valid,
        grf_valid=grf_valid,
        gait_windows=windows,
        torque_cycles=torque_cycles,
        grf_cycles=grf_cycles,
    )
