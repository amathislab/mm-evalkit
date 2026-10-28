"""Load policy kinetic summaries and select matched gait profiles.

Kinetic values are collected during MuJoCo evaluation because actuator moments
and sensor forces are not part of the generic rollout archive. The summary NPZ
records those values on the OpenSim reference timeline. This module validates
that contract, resolves its references, and constructs the typed state consumed
by metrics and figures.
"""

from __future__ import annotations

import csv
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mm_evalkit.arrays import FloatArray, IntArray
from mm_evalkit.datasets.gait120_kinetics import (
    GRF_CHANNELS,
    GRF_UNIT,
    TORQUE_JOINTS,
    TORQUE_UNIT,
    Gait120KineticsTrial,
    KineticsWindow,
    load_gait120_kinetics_trial,
)
from mm_evalkit.policies import PolicySpec

from .motion_identity import subject_from_trajectory, trial_from_trajectory
from .numerics import finite_mean
from .rollout_validation import has_embedded_kinetics
from .state import KineticsCohort, KineticsCycles, KineticsOmission, KineticsProfiles, PolicyRun
from .world_grf import load_world_grf

_REFERENCE_NAME = re.compile(r"^(?P<subject>S\d{3})_Trial\d{2}_Step01_Step02$")
_SUMMARY_TORQUE_UNITS = {"nm_per_kg", "nm/kg"}
_SUMMARY_GRF_UNITS = {"body_weight_normalized", "body_weight_units", "bw"}
_RAW_TORQUE_UNITS = {"nm", "n_m"}
_RAW_FORCE_UNITS = {"n", "newtons"}
_GRAVITY_MPS2 = 9.80665
# Rollouts end a few frames before their reference trial does, so a stride
# that runs to the trial end is kept when this much of it, from heel strike
# on, is covered. The uncovered tail stays NaN.
MIN_STRIDE_COVERAGE = 0.9
# The references mirror left hip adduction and rotation, as Gait120 IK does for
# the left hip angles, so the matching policy moments are mirrored to compare.
_MIRRORED_LEFT_MOMENTS = frozenset({"hip_adduction_l", "hip_rotation_l"})


@dataclass(frozen=True, slots=True)
class LoadedPolicyKinetics:
    """One policy summary after schema validation and gait extraction."""

    policy_id: str
    cycles: KineticsCycles
    omissions: tuple[KineticsOmission, ...]
    references: dict[tuple[int, int], Gait120KineticsTrial]


def _scalar(data: Any, key: str, path: Path) -> Any:
    if key not in data:
        raise ValueError(f"Policy kinetics summary is missing {key!r}: {path}")
    value = np.asarray(data[key])
    if value.shape != ():
        raise ValueError(f"Policy kinetics {key!r} must be a scalar in {path}, got shape {value.shape}.")
    return value.item()


def _scalar_bool(data: Any, key: str, path: Path) -> bool:
    value = _scalar(data, key, path)
    if not isinstance(value, bool | np.bool_):
        raise ValueError(f"Policy kinetics {key!r} must be boolean in {path}.")
    return bool(value)


def _strings(data: Any, key: str, path: Path) -> tuple[str, ...]:
    if key not in data:
        raise ValueError(f"Policy kinetics summary is missing {key!r}: {path}")
    values = np.asarray(data[key])
    if values.ndim != 1:
        raise ValueError(f"Policy kinetics {key!r} must be a vector in {path}.")
    strings = tuple(str(value) for value in values.tolist())
    if len(strings) != len(set(strings)):
        raise ValueError(f"Policy kinetics {key!r} contains duplicate entries in {path}.")
    return strings


def _string_values(data: Any, key: str, path: Path) -> tuple[str, ...]:
    if key not in data:
        raise ValueError(f"Policy kinetics summary is missing {key!r}: {path}")
    values = np.asarray(data[key])
    if values.ndim != 1:
        raise ValueError(f"Policy kinetics {key!r} must be a vector in {path}.")
    return tuple(str(value) for value in values.tolist())


def _episode_matrix(data: Any, key: str, rows: int, columns: int, path: Path) -> FloatArray:
    if key not in data:
        raise ValueError(f"Policy kinetics summary is missing {key!r}: {path}")
    values = np.asarray(data[key], dtype=float)
    if values.shape != (rows, columns):
        raise ValueError(f"Policy kinetics {key!r} must have shape {(rows, columns)} in {path}, got {values.shape}.")
    return values


def _reference_path(
    data: Any,
    episode: int,
    summary_path: Path,
    reference_dir: Path | None,
) -> tuple[str, Path]:
    prefix = f"episode_{episode}"
    name = str(_scalar(data, f"{prefix}_reference_name", summary_path))
    match = _REFERENCE_NAME.fullmatch(name)
    if match is None:
        raise ValueError(
            f"Policy kinetics reference name must match S###_Trial##_Step01_Step02 in {summary_path}, got {name!r}."
        )
    if reference_dir is not None:
        path = reference_dir / match.group("subject") / f"{name}.npz"
    else:
        raw = Path(str(_scalar(data, f"{prefix}_reference_path", summary_path))).expanduser()
        path = raw if raw.is_absolute() else summary_path.parent / raw
    if not path.is_file():
        source = "kinetics_reference_dir" if reference_dir is not None else "embedded reference_path"
        raise FileNotFoundError(f"Policy kinetics {source} does not exist for {name}: {path}")
    return name, path.resolve()


def _closest_reference_indices(timesteps: FloatArray, reference_time: FloatArray, path: Path, episode: int) -> IntArray:
    if timesteps.ndim != 1 or not np.all(np.isfinite(timesteps)) or np.any(np.diff(timesteps) <= 0):
        raise ValueError(
            f"Policy kinetics episode {episode} timesteps must be finite and strictly increasing in {path}."
        )
    right = np.searchsorted(reference_time, timesteps, side="left")
    right = np.clip(right, 0, len(reference_time) - 1)
    left = np.maximum(right - 1, 0)
    use_left = np.abs(timesteps - reference_time[left]) <= np.abs(timesteps - reference_time[right])
    indices = np.where(use_left, left, right)
    reference_dt = np.diff(reference_time)
    tolerance = max(float(np.median(reference_dt)) * 0.51, 1e-9)
    error = np.abs(timesteps - reference_time[indices])
    if np.any(error > tolerance):
        worst = float(np.max(error))
        raise ValueError(
            f"Policy kinetics episode {episode} is not sampled on its reference timeline in {path}; "
            f"maximum timestamp mismatch is {worst:g}."
        )
    if len(indices) != len(np.unique(indices)):
        raise ValueError(f"Policy kinetics episode {episode} maps multiple samples to one reference frame in {path}.")
    return indices.astype(int)


def _window_samples(
    frame_indices: IntArray, window: tuple[int, int], kinetics_window: KineticsWindow
) -> np.ndarray | None:
    """Rows inside ``window``, or None when the policy does not cover enough of it.

    A fixed left-heel-strike window must be covered frame for frame. A right
    stride must be covered contiguously from its heel strike for at least
    ``MIN_STRIDE_COVERAGE`` of its length.
    """
    start, end = window
    in_window = (frame_indices >= start) & (frame_indices < end)
    covered = frame_indices[in_window]
    contiguous = np.array_equal(covered, np.arange(start, start + len(covered)))
    required = end - start if kinetics_window == "left_heel_strike" else int(np.ceil(MIN_STRIDE_COVERAGE * (end - start)))
    return in_window if contiguous and len(covered) >= required else None


def _phase_normalize_samples(values: FloatArray, frame_indices: IntArray, window: tuple[int, int]) -> FloatArray:
    """Interpolate onto the 100-point grid without extrapolating past coverage."""
    start, end = window
    source = (np.asarray(frame_indices, dtype=float) - start) / (end - start - 1)
    target = np.linspace(0.0, 1.0, 100)
    output = np.full((values.shape[1], 100), np.nan, dtype=float)
    for channel in range(values.shape[1]):
        finite = np.isfinite(values[:, channel])
        if np.count_nonzero(finite) < 2:
            continue
        inside = (target >= source[finite][0] - 1e-9) & (target <= source[finite][-1] + 1e-9)
        output[channel, inside] = np.interp(target[inside], source[finite], values[finite, channel])
    return output


def _swap_torque_sides(values: FloatArray, torque_joints: tuple[str, ...]) -> FloatArray:
    indexes = {name: index for index, name in enumerate(torque_joints)}
    source = []
    for name in torque_joints:
        opposite = f"{name[:-2]}_{'l' if name.endswith('_r') else 'r'}"
        source.append(indexes[opposite])
    return values[:, source]


def _empty_cycles(torque_joints: tuple[str, ...]) -> KineticsCycles:
    return KineticsCycles(
        torque=np.empty((0, len(torque_joints), 100)),
        torque_ref=np.empty((0, len(torque_joints), 100)),
        grf=np.empty((0, len(GRF_CHANNELS), 100)),
        grf_ref=np.empty((0, len(GRF_CHANNELS), 100)),
        subject_ids=np.empty(0, dtype=int),
        trial_ids=np.empty(0, dtype=int),
        gait_positions=np.empty(0, dtype=int),
    )


def _load_summary_policy_kinetics(
    policy: PolicySpec,
    *,
    reference_dir: Path | None,
    kinetics_window: KineticsWindow = "left_heel_strike",
    torque_joints: tuple[str, ...] = TORQUE_JOINTS,
) -> LoadedPolicyKinetics:
    """Validate and extract one policy's MuJoCo torque/GRF summary."""
    if policy.kinetics_summary_path is None:
        raise ValueError(f"Policy {policy.id!r} has no kinetics_summary_path.")
    path = policy.kinetics_summary_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Policy kinetics summary does not exist: {path}")

    torque_cycles: list[FloatArray] = []
    torque_references: list[FloatArray] = []
    grf_cycles: list[FloatArray] = []
    grf_references: list[FloatArray] = []
    subject_ids: list[int] = []
    trial_ids: list[int] = []
    gait_positions: list[int] = []
    omissions: list[KineticsOmission] = []
    references: dict[tuple[int, int], Gait120KineticsTrial] = {}

    with np.load(path, allow_pickle=False) as data:
        n_episodes = int(_scalar(data, "n_episodes", path))
        if n_episodes < 1:
            raise ValueError(f"Policy kinetics n_episodes must be positive in {path}.")
        joint_names = _strings(data, "joint_names", path)
        missing_joints = [name for name in torque_joints if name not in joint_names]
        if missing_joints:
            raise ValueError(
                f"Policy kinetics summary is missing required joint(s) in {path}: {', '.join(missing_joints)}."
            )
        joint_indexes = [joint_names.index(name) for name in torque_joints]
        grf_columns = _strings(data, "grf_columns", path)
        required_grf = ("group_2_3_y", "group_0_1_y")
        missing_grf = [name for name in required_grf if name not in grf_columns]
        if missing_grf:
            raise ValueError(
                f"Policy kinetics summary is missing required GRF channel(s) in {path}: {', '.join(missing_grf)}."
            )
        grf_indexes = [grf_columns.index(name) for name in required_grf]

        torque_units = str(_scalar(data, "torque_units", path)).strip().casefold()
        grf_units = str(_scalar(data, "grf_units", path)).strip().casefold()
        if torque_units not in _SUMMARY_TORQUE_UNITS:
            raise ValueError(f"Policy kinetics torque_units must be Nm_per_kg in {path}, got {torque_units!r}.")
        if grf_units not in _SUMMARY_GRF_UNITS:
            raise ValueError(f"Policy kinetics grf_units must be body_weight_normalized in {path}, got {grf_units!r}.")
        knee_flipped = _scalar_bool(data, "policy_knee_torque_sign_flipped", path)

        for episode in range(n_episodes):
            prefix = f"episode_{episode}"
            reference_name, resolved_reference = _reference_path(data, episode, path, reference_dir)
            reference = load_gait120_kinetics_trial(
                resolved_reference, window=kinetics_window, torque_joints=torque_joints
            )
            key = (reference.subject, reference.trial)
            existing = references.get(key)
            if existing is not None and existing.path != reference.path:
                raise ValueError(
                    f"Policy kinetics summary maps subject/trial {key} to multiple references: "
                    f"{existing.path} and {reference.path}."
                )
            references[key] = reference

            timesteps = np.asarray(data[f"{prefix}_timesteps"], dtype=float)
            n_rows = len(timesteps)
            policy_torque = _episode_matrix(data, f"{prefix}_joint_moment", n_rows, len(joint_names), path)[
                :, joint_indexes
            ]
            policy_grf = _episode_matrix(data, f"{prefix}_grf", n_rows, len(grf_columns), path)[:, grf_indexes]
            if not knee_flipped:
                for index, name in enumerate(torque_joints):
                    if name.startswith("knee_angle_"):
                        policy_torque[:, index] *= -1.0
            if policy.kinetics_swap_left_right:
                policy_torque = _swap_torque_sides(policy_torque, torque_joints)
                policy_grf = policy_grf[:, ::-1]
            # After any swap, so a moment is mirrored by the side it now stands for.
            for index, name in enumerate(torque_joints):
                if name in _MIRRORED_LEFT_MOMENTS:
                    policy_torque[:, index] *= -1.0
            indices = _closest_reference_indices(timesteps, reference.time, path, episode)

            for gait_position, window in enumerate(reference.gait_windows):
                in_window = _window_samples(indices, window, kinetics_window)
                if in_window is None:
                    omissions.append(
                        KineticsOmission(
                            policy_id=policy.id,
                            episode=episode,
                            reference_name=reference_name,
                            gait_position=gait_position,
                            reason="incomplete_reference_window_coverage",
                        )
                    )
                    continue
                covered = indices[in_window]
                normalized_torque = _phase_normalize_samples(policy_torque[in_window], covered, window)
                normalized_grf = _phase_normalize_samples(policy_grf[in_window], covered, window)
                finite_torque_channels = np.isfinite(normalized_torque).any(axis=1)
                finite_grf_channels = np.isfinite(normalized_grf).any(axis=1)
                if not np.all(finite_torque_channels) or not np.all(finite_grf_channels):
                    raise ValueError(
                        f"Policy kinetics {policy.id!r} episode {episode}, gait {gait_position + 1} "
                        "has no finite samples for one or more required torque/GRF channels."
                    )
                torque_cycles.append(normalized_torque)
                torque_references.append(reference.torque_cycles[gait_position])
                grf_cycles.append(normalized_grf)
                grf_references.append(reference.grf_cycles[gait_position])
                subject_ids.append(reference.subject)
                trial_ids.append(reference.trial)
                gait_positions.append(gait_position)

    cycles = (
        KineticsCycles(
            torque=np.stack(torque_cycles),
            torque_ref=np.stack(torque_references),
            grf=np.stack(grf_cycles),
            grf_ref=np.stack(grf_references),
            subject_ids=np.asarray(subject_ids, dtype=int),
            trial_ids=np.asarray(trial_ids, dtype=int),
            gait_positions=np.asarray(gait_positions, dtype=int),
        )
        if torque_cycles
        else _empty_cycles(torque_joints)
    )
    return LoadedPolicyKinetics(policy.id, cycles, tuple(omissions), references)


def policy_has_embedded_kinetics(policy: PolicySpec) -> bool:
    """Return whether a rollout carries the complete generic kinetics export."""
    path = policy.trajectory_path.expanduser().resolve()
    with np.load(path, allow_pickle=False) as data:
        return has_embedded_kinetics(data, path)


def _rollout_reference(
    trajectory_id: int,
    motion_list: tuple[str, ...],
    reference_dir: Path,
) -> tuple[str, Path]:
    subject = subject_from_trajectory(trajectory_id, motion_list)
    trial = trial_from_trajectory(trajectory_id, motion_list)
    name = f"S{subject:03d}_Trial{trial:02d}_Step01_Step02"
    path = reference_dir / f"S{subject:03d}" / f"{name}.npz"
    return name, path.resolve()


def _integer_vector(data: Any, key: str, rows: int, path: Path) -> IntArray:
    if key not in data:
        raise ValueError(f"Policy kinetics rollout is missing {key!r}: {path}")
    values = np.asarray(data[key])
    if values.shape != (rows,) or values.dtype.kind not in "iu":
        raise ValueError(f"Policy kinetics {key!r} must be an integer vector of length {rows} in {path}.")
    if np.any(np.diff(values) <= 0):
        raise ValueError(f"Policy kinetics {key!r} must be strictly increasing in {path}.")
    return values.astype(int, copy=False)


def _load_rollout_policy_kinetics(
    policy: PolicySpec,
    *,
    reference_dir: Path,
    motion_list: tuple[str, ...],
    kinetics_window: KineticsWindow = "left_heel_strike",
    torque_joints: tuple[str, ...] = TORQUE_JOINTS,
) -> LoadedPolicyKinetics:
    path = policy.trajectory_path.expanduser().resolve()
    torque_cycles: list[FloatArray] = []
    torque_references: list[FloatArray] = []
    grf_cycles: list[FloatArray] = []
    grf_references: list[FloatArray] = []
    subject_ids: list[int] = []
    trial_ids: list[int] = []
    gait_positions: list[int] = []
    omissions: list[KineticsOmission] = []
    references: dict[tuple[int, int], Gait120KineticsTrial] = {}

    with np.load(path, allow_pickle=False) as data:
        if not has_embedded_kinetics(data, path):
            raise ValueError(
                f"Policy {policy.id!r} has no kinetics_summary_path and its rollout has no embedded kinetics: {path}"
            )
        n_episodes = int(_scalar(data, "n_episodes", path))
        joint_names = _strings(data, "generalized_actuator_force_joint_names", path)
        joint_units = _string_values(data, "generalized_actuator_force_units", path)
        if len(joint_names) != len(joint_units):
            raise ValueError(f"Policy kinetics joint names and units differ in length in {path}.")
        missing_joints = [name for name in torque_joints if name not in joint_names]
        if missing_joints:
            raise ValueError(f"Policy kinetics rollout is missing required joint(s) in {path}: {missing_joints}.")
        joint_indexes = [joint_names.index(name) for name in torque_joints]
        invalid_joint_units = [
            joint_units[index] for index in joint_indexes if joint_units[index].casefold() not in _RAW_TORQUE_UNITS
        ]
        if invalid_joint_units:
            raise ValueError(f"Policy kinetics required hinge-joint forces must use Nm in {path}.")

        sensor_names = _strings(data, "force_sensor_names", path)
        sensor_units = _string_values(data, "force_sensor_units", path)
        if len(sensor_names) != len(sensor_units):
            raise ValueError(f"Policy kinetics force sensor names and units differ in length in {path}.")
        required_sensors = ("r_foot", "r_toes", "l_foot", "l_toes")
        missing_sensors = [name for name in required_sensors if name not in sensor_names]
        if missing_sensors:
            raise ValueError(
                f"Policy kinetics rollout is missing required force sensor(s) in {path}: {missing_sensors}."
            )
        sensor_indexes = [sensor_names.index(name) for name in required_sensors]
        if any(sensor_units[index].casefold() not in _RAW_FORCE_UNITS for index in sensor_indexes):
            raise ValueError(f"Policy kinetics foot-force sensors must use N in {path}.")

        model_mass_kg = float(_scalar(data, "model_mass_kg", path))
        if not np.isfinite(model_mass_kg) or model_mass_kg <= 0:
            raise ValueError(f"Policy kinetics model_mass_kg must be finite and positive in {path}.")
        model_body_weight_n = model_mass_kg * _GRAVITY_MPS2

        for episode in range(n_episodes):
            prefix = f"episode_{episode}"
            trajectory_id = int(_scalar(data, f"{prefix}_traj_id", path))
            reference_name, reference_path = _rollout_reference(trajectory_id, motion_list, reference_dir)
            if not reference_path.is_file():
                omissions.append(
                    KineticsOmission(
                        policy_id=policy.id,
                        episode=episode,
                        reference_name=reference_name,
                        gait_position=-1,
                        reason="missing_reference_file",
                    )
                )
                continue
            reference = load_gait120_kinetics_trial(
                reference_path, window=kinetics_window, torque_joints=torque_joints
            )
            key = (reference.subject, reference.trial)
            existing = references.get(key)
            if existing is not None and existing.path != reference.path:
                raise ValueError(
                    f"Policy kinetics rollout maps subject/trial {key} to multiple references: "
                    f"{existing.path} and {reference.path}."
                )
            references[key] = reference

            raw_torque = np.asarray(data[f"{prefix}_generalized_actuator_forces"], dtype=float)
            raw_sensors = np.asarray(data[f"{prefix}_force_sensor_values"], dtype=float)
            if raw_torque.ndim != 2 or raw_torque.shape[1] != len(joint_names):
                raise ValueError(f"Policy kinetics {prefix}_generalized_actuator_forces has invalid shape in {path}.")
            if raw_sensors.shape != (len(raw_torque), len(sensor_names)):
                raise ValueError(f"Policy kinetics {prefix}_force_sensor_values has invalid shape in {path}.")
            frame_indices = _integer_vector(
                data,
                f"{prefix}_kinetics_reference_steps",
                len(raw_torque),
                path,
            )
            if np.any(frame_indices < 0) or np.any(frame_indices >= len(reference.time)):
                raise ValueError(
                    f"Policy kinetics {prefix}_kinetics_reference_steps falls outside its reference in {path}."
                )

            policy_torque = raw_torque[:, joint_indexes] / model_mass_kg
            for index, name in enumerate(torque_joints):
                if name.startswith("knee_angle_"):
                    policy_torque[:, index] *= -1.0
            selected_sensors = raw_sensors[:, sensor_indexes]
            policy_grf = load_world_grf(data, prefix, len(raw_torque), path)
            if policy_grf is None:
                policy_grf = (
                    np.column_stack(
                        [
                            selected_sensors[:, 2] + selected_sensors[:, 3],
                            selected_sensors[:, 0] + selected_sensors[:, 1],
                        ]
                    )
                    / model_body_weight_n
                )
            if policy.kinetics_swap_left_right:
                policy_torque = _swap_torque_sides(policy_torque, torque_joints)
                policy_grf = policy_grf[:, ::-1]
            # After any swap, so a moment is mirrored by the side it now stands for.
            for index, name in enumerate(torque_joints):
                if name in _MIRRORED_LEFT_MOMENTS:
                    policy_torque[:, index] *= -1.0

            for gait_position, window in enumerate(reference.gait_windows):
                in_window = _window_samples(frame_indices, window, kinetics_window)
                if in_window is None:
                    omissions.append(
                        KineticsOmission(
                            policy_id=policy.id,
                            episode=episode,
                            reference_name=reference_name,
                            gait_position=gait_position,
                            reason="incomplete_reference_window_coverage",
                        )
                    )
                    continue
                covered = frame_indices[in_window]
                normalized_torque = _phase_normalize_samples(policy_torque[in_window], covered, window)
                normalized_grf = _phase_normalize_samples(policy_grf[in_window], covered, window)
                if not np.all(np.isfinite(normalized_torque).any(axis=1)) or not np.all(
                    np.isfinite(normalized_grf).any(axis=1)
                ):
                    raise ValueError(
                        f"Policy kinetics {policy.id!r} episode {episode}, gait {gait_position + 1} "
                        "has no finite samples for one or more required torque/GRF channels."
                    )
                torque_cycles.append(normalized_torque)
                torque_references.append(reference.torque_cycles[gait_position])
                grf_cycles.append(normalized_grf)
                grf_references.append(reference.grf_cycles[gait_position])
                subject_ids.append(reference.subject)
                trial_ids.append(reference.trial)
                gait_positions.append(gait_position)

    cycles = (
        KineticsCycles(
            torque=np.stack(torque_cycles),
            torque_ref=np.stack(torque_references),
            grf=np.stack(grf_cycles),
            grf_ref=np.stack(grf_references),
            subject_ids=np.asarray(subject_ids, dtype=int),
            trial_ids=np.asarray(trial_ids, dtype=int),
            gait_positions=np.asarray(gait_positions, dtype=int),
        )
        if torque_cycles
        else _empty_cycles(torque_joints)
    )
    return LoadedPolicyKinetics(policy.id, cycles, tuple(omissions), references)


def load_policy_kinetics(
    policy: PolicySpec,
    *,
    reference_dir: Path | None,
    motion_list: tuple[str, ...] | None = None,
    kinetics_window: KineticsWindow = "left_heel_strike",
    torque_joints: tuple[str, ...] = TORQUE_JOINTS,
) -> LoadedPolicyKinetics:
    """Load legacy summary kinetics or the generic rollout's embedded signals."""
    if policy.kinetics_summary_path is not None:
        return _load_summary_policy_kinetics(
            policy, reference_dir=reference_dir, kinetics_window=kinetics_window, torque_joints=torque_joints
        )
    if reference_dir is None or motion_list is None:
        raise ValueError(
            f"Policy {policy.id!r} embedded kinetics requires kinetics_reference_dir and an explicit motion group."
        )
    return _load_rollout_policy_kinetics(
        policy,
        reference_dir=reference_dir,
        motion_list=motion_list,
        kinetics_window=kinetics_window,
        torque_joints=torque_joints,
    )


def _reference_cycles(references: dict[tuple[int, int], Gait120KineticsTrial]) -> KineticsCycles:
    torque: list[FloatArray] = []
    grf: list[FloatArray] = []
    subjects: list[int] = []
    trials: list[int] = []
    positions: list[int] = []
    for key in sorted(references):
        reference = references[key]
        for position in range(len(reference.gait_windows)):
            torque.append(reference.torque_cycles[position])
            grf.append(reference.grf_cycles[position])
            subjects.append(reference.subject)
            trials.append(reference.trial)
            positions.append(position)
    torque_array = np.stack(torque)
    grf_array = np.stack(grf)
    return KineticsCycles(
        torque=torque_array,
        torque_ref=torque_array.copy(),
        grf=grf_array,
        grf_ref=grf_array.copy(),
        subject_ids=np.asarray(subjects, dtype=int),
        trial_ids=np.asarray(trials, dtype=int),
        gait_positions=np.asarray(positions, dtype=int),
    )


def load_kinetics_cohort(
    runs: list[PolicyRun],
    *,
    reference_dir: Path | None,
    motion_list: tuple[str, ...] | None = None,
    kinetics_window: KineticsWindow = "left_heel_strike",
    torque_joints: tuple[str, ...] = TORQUE_JOINTS,
) -> KineticsCohort | None:
    """Load the optional all-policy kinetics layer and require one reference sample."""
    summary_configured = [run.policy.kinetics_summary_path is not None for run in runs]
    if reference_dir is None:
        configured = summary_configured
        if not any(configured):
            return None
    else:
        configured = [
            summary or policy_has_embedded_kinetics(run.policy)
            for run, summary in zip(runs, summary_configured, strict=True)
        ]
    if not all(configured):
        raise ValueError("Kinetics must be configured for every compared policy.")

    loaded = [
        load_policy_kinetics(
            run.policy,
            reference_dir=reference_dir,
            motion_list=motion_list,
            kinetics_window=kinetics_window,
            torque_joints=torque_joints,
        )
        for run in runs
    ]
    non_trial_five = sorted(identity for result in loaded for identity in result.references if identity[1] != 5)
    if non_trial_five:
        raise ValueError(
            "Primary Gait120 kinetics must use Trial05 references to match the configured rollout and human cohort; "
            f"got {non_trial_five}."
        )
    expected_identities = set(loaded[0].references)
    if not expected_identities:
        raise ValueError("Kinetics was requested, but no rollout episode has a matching Gait120 kinetic reference.")
    for result in loaded[1:]:
        identities = set(result.references)
        if identities != expected_identities:
            missing = sorted(expected_identities - identities)
            extra = sorted(identities - expected_identities)
            raise ValueError(
                f"Policy {result.policy_id!r} kinetics references do not match the first policy; "
                f"missing={missing}, extra={extra}."
            )
        for identity in expected_identities:
            expected = loaded[0].references[identity]
            actual = result.references[identity]
            if (
                expected.gait_windows != actual.gait_windows
                or not np.allclose(expected.torque_cycles, actual.torque_cycles, equal_nan=True)
                or not np.allclose(expected.grf_cycles, actual.grf_cycles, equal_nan=True)
            ):
                raise ValueError(
                    f"Policy {result.policy_id!r} uses a different kinetics reference for subject/trial {identity}."
                )

    for run, result in zip(runs, loaded, strict=True):
        run.kinetics_cycles = result.cycles
        run.kinetics_omissions = result.omissions
        if result.omissions:
            print(f"  Policy {run.label}: excluded {len(result.omissions)} unavailable/incomplete kinetics input(s).")

    references = loaded[0].references
    paths = tuple(references[key].path for key in sorted(references))
    return KineticsCohort(
        torque_names=tuple(torque_joints),
        grf_names=GRF_CHANNELS,
        torque_unit=TORQUE_UNIT,
        grf_unit=GRF_UNIT,
        cycles=_reference_cycles(references),
        reference_paths=paths,
    )


def write_kinetics_omission_audit(runs: list[PolicyRun], output_dir: Path) -> Path | None:
    """Write one row per unavailable input or incomplete policy window."""
    if not any(run.kinetics_cycles is not None for run in runs):
        return None
    path = output_dir / "kinetics_input_omissions.csv"
    fieldnames = ("policy_id", "episode", "reference_name", "gait_position", "reason")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            writer.writerows(asdict(omission) for omission in run.kinetics_omissions)
    return path


def _selection_mask(cycles: KineticsCycles, subjects: list[int], gait_position: int | None) -> np.ndarray:
    mask = np.isin(cycles.subject_ids, subjects)
    if gait_position is not None:
        mask &= cycles.gait_positions == gait_position
    return mask


def available_subjects(cycles: KineticsCycles, subjects: list[int], gait_position: int | None) -> set[int]:
    """Return candidate subjects with at least one kinetic cycle in this selection."""
    mask = _selection_mask(cycles, subjects, gait_position)
    return {int(subject) for subject in np.unique(cycles.subject_ids[mask])}


def build_kinetics_profiles(cycles: KineticsCycles, mask: np.ndarray) -> KineticsProfiles:
    """Average cycles within subject, then subjects with equal weight."""
    selected_subjects = np.unique(cycles.subject_ids[mask]).astype(int)
    per_torque = {
        int(subject): finite_mean(cycles.torque[mask & (cycles.subject_ids == subject)], axis=0)
        for subject in selected_subjects
    }
    per_torque_ref = {
        int(subject): finite_mean(cycles.torque_ref[mask & (cycles.subject_ids == subject)], axis=0)
        for subject in selected_subjects
    }
    per_grf = {
        int(subject): finite_mean(cycles.grf[mask & (cycles.subject_ids == subject)], axis=0)
        for subject in selected_subjects
    }
    per_grf_ref = {
        int(subject): finite_mean(cycles.grf_ref[mask & (cycles.subject_ids == subject)], axis=0)
        for subject in selected_subjects
    }
    torque_subjects = np.stack(list(per_torque.values()))
    torque_ref_subjects = np.stack(list(per_torque_ref.values()))
    grf_subjects = np.stack(list(per_grf.values()))
    grf_ref_subjects = np.stack(list(per_grf_ref.values()))
    return KineticsProfiles(
        avg_torque=finite_mean(torque_subjects, axis=0),
        std_torque=np.nanstd(torque_subjects, axis=0),
        avg_torque_ref=finite_mean(torque_ref_subjects, axis=0),
        std_torque_ref=np.nanstd(torque_ref_subjects, axis=0),
        avg_grf=finite_mean(grf_subjects, axis=0),
        std_grf=np.nanstd(grf_subjects, axis=0),
        avg_grf_ref=finite_mean(grf_ref_subjects, axis=0),
        std_grf_ref=np.nanstd(grf_ref_subjects, axis=0),
        subject_ids=selected_subjects,
        per_subject_torque=per_torque,
        per_subject_torque_ref=per_torque_ref,
        per_subject_grf=per_grf,
        per_subject_grf_ref=per_grf_ref,
    )


def select_kinetics_subset(runs: list[PolicyRun], cohort: KineticsCohort, subjects: list[int], gait_position) -> None:
    """Rebuild human and policy kinetic profiles for the active shared cohort."""
    human_mask = _selection_mask(cohort.cycles, subjects, gait_position)
    human_profiles = build_kinetics_profiles(cohort.cycles, human_mask)
    cohort.subjects = [int(subject) for subject in human_profiles.subject_ids]
    cohort.torque_data = np.stack([human_profiles.per_subject_torque[subject] for subject in cohort.subjects])
    cohort.grf_data = np.stack([human_profiles.per_subject_grf[subject] for subject in cohort.subjects])

    for run in runs:
        if run.kinetics_cycles is None:
            raise RuntimeError(f"Policy {run.policy.id!r} has no loaded kinetics cycles.")
        mask = _selection_mask(run.kinetics_cycles, subjects, gait_position)
        run.kinetics_profiles = build_kinetics_profiles(run.kinetics_cycles, mask)
