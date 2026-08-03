"""Structural validation for exported full-body rollout archives."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from mm_kit.activation import activation_sample_phase
from mm_kit.policies import PolicySpec

EPISODE_FIELDS = (
    "traj_id",
    "traj_length",
    "joint_positions",
    "traj_qpos",
    "muscle_activations",
    "touch_observations",
)
KINETICS_TOP_LEVEL_FIELDS = (
    "kinetics_sample_phase",
    "generalized_actuator_force_joint_names",
    "generalized_actuator_force_units",
    "force_sensor_names",
    "force_sensor_units",
    "model_mass_kg",
)
KINETICS_EPISODE_FIELDS = (
    "generalized_actuator_forces",
    "force_sensor_values",
    "kinetics_reference_steps",
)


def _integer_scalar(archive: Any, key: str, source: Path) -> int:
    """Read an integer scalar without silently truncating a float."""
    array = np.asarray(archive[key])
    if array.size != 1:
        raise ValueError(f"Rollout archive {source} field {key!r} must be scalar.")
    value = array.item()
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError(f"Rollout archive {source} field {key!r} must be an integer.")
    if not np.isfinite(value) or int(value) != value:
        raise ValueError(f"Rollout archive {source} field {key!r} must be an integer.")
    return int(value)


def _optional_string_scalar(archive: Any, key: str, source: Path) -> str | None:
    """Read optional export provenance without guessing a missing value."""
    if key not in archive:
        return None
    array = np.asarray(archive[key])
    if array.size != 1:
        raise ValueError(f"Rollout archive {source} field {key!r} must be scalar.")
    value = array.item()
    if value is None:
        return None
    if not isinstance(value, (str, np.str_)) or not str(value):
        raise ValueError(f"Rollout archive {source} field {key!r} must be a non-empty string.")
    return str(value)


def has_embedded_kinetics(archive: Any, source: str | Path) -> bool:
    """Validate the optional top-level marker as all-present or all-absent."""
    path = Path(source)
    present = [key for key in KINETICS_TOP_LEVEL_FIELDS if key in archive]
    if not present:
        return False
    missing = [key for key in KINETICS_TOP_LEVEL_FIELDS if key not in archive]
    if missing:
        raise ValueError(
            f"Rollout archive {path} has a partial embedded kinetics schema; missing: {', '.join(missing)}."
        )
    phase = _optional_string_scalar(archive, "kinetics_sample_phase", path)
    if phase != "post_step":
        raise ValueError(f"Rollout archive {path} kinetics_sample_phase must be 'post_step', got {phase!r}.")
    return True


def _string_vector(archive: Any, key: str, source: Path) -> tuple[str, ...]:
    values = np.asarray(archive[key])
    if values.ndim != 1:
        raise ValueError(f"Rollout archive {source} field {key!r} must be a vector.")
    strings = tuple(str(value) for value in values.tolist())
    if any(not value for value in strings) or len(strings) != len(set(strings)):
        raise ValueError(f"Rollout archive {source} field {key!r} must contain unique non-empty names.")
    return strings


def _string_values(archive: Any, key: str, source: Path) -> tuple[str, ...]:
    values = np.asarray(archive[key])
    if values.ndim != 1:
        raise ValueError(f"Rollout archive {source} field {key!r} must be a vector.")
    strings = tuple(str(value) for value in values.tolist())
    if any(not value for value in strings):
        raise ValueError(f"Rollout archive {source} field {key!r} must contain non-empty values.")
    return strings


def _validate_embedded_kinetics_metadata(archive: Any, source: Path) -> tuple[int, int]:
    joint_names = _string_vector(archive, "generalized_actuator_force_joint_names", source)
    joint_units = _string_values(archive, "generalized_actuator_force_units", source)
    sensor_names = _string_vector(archive, "force_sensor_names", source)
    sensor_units = _string_values(archive, "force_sensor_units", source)
    if len(joint_names) != len(joint_units):
        raise ValueError(f"Rollout archive {source} generalized actuator-force names and units differ in length.")
    if len(sensor_names) != len(sensor_units):
        raise ValueError(f"Rollout archive {source} force-sensor names and units differ in length.")
    mass = np.asarray(archive["model_mass_kg"])
    if mass.shape != () or not np.isfinite(float(mass.item())) or float(mass.item()) <= 0:
        raise ValueError(f"Rollout archive {source} model_mass_kg must be a finite positive scalar.")
    return len(joint_names), len(sensor_names)


def validate_rollout_archive(
    archive: Any,
    path: str | Path,
    *,
    subject_based: bool,
    motion_count: int | None,
    expected_joint_names: tuple[str, ...] | None = None,
    expected_joint_width: int | None = None,
    expected_activation_width: int | None = None,
) -> int:
    """Validate the exported coordinate system and return ``n_episodes``.

    Early termination and absence of a usable gait cycle are scientific
    outcomes handled by the extractor.  Missing keys, invalid trajectory IDs,
    and inconsistent time axes instead indicate a malformed input archive and
    stop the run.
    """
    source = Path(path)
    top_level = ("n_episodes", "dt", "joint_names")
    missing_top_level = [key for key in top_level if key not in archive]
    if missing_top_level:
        raise ValueError(f"Rollout archive {source} is missing required key(s): {', '.join(missing_top_level)}.")
    n_episodes = _integer_scalar(archive, "n_episodes", source)
    if n_episodes <= 0:
        raise ValueError(f"Rollout archive {source} must contain at least one episode.")
    activation_sample_phase(archive)
    embedded_kinetics = has_embedded_kinetics(archive, source)
    kinetics_widths = _validate_embedded_kinetics_metadata(archive, source) if embedded_kinetics else None
    try:
        dt = float(np.asarray(archive["dt"]).item())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Rollout archive {source} has a non-scalar dt.") from exc
    if not np.isfinite(dt) or not np.isclose(dt, 0.01):
        raise ValueError(f"Rollout archive {source} must use dt=0.01 s, got {dt!r}.")
    raw_names = np.asarray(archive["joint_names"])
    if raw_names.ndim != 1 or not len(raw_names):
        raise ValueError(f"Rollout archive {source} joint_names must be a non-empty vector.")
    joint_names = tuple(str(name) for name in raw_names)
    if any(not name for name in joint_names) or len(set(joint_names)) != len(joint_names):
        raise ValueError(f"Rollout archive {source} joint_names must be non-empty and unique.")
    if expected_joint_names is not None and joint_names != expected_joint_names:
        raise ValueError(f"Rollout archive {source} joint_names do not match the MyoFullBody model.")

    required = [f"episode_{episode}_{field}" for episode in range(n_episodes) for field in EPISODE_FIELDS]
    missing = [key for key in required if key not in archive]
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "" if len(missing) <= 5 else f", ... ({len(missing)} missing)"
        raise ValueError(f"Rollout archive {source} is missing required key(s): {preview}{suffix}.")

    if subject_based and motion_count is None:
        raise ValueError("A motion list is required to decode subject-based Gait120 trajectory IDs.")

    for episode in range(n_episodes):
        prefix = f"episode_{episode}"
        traj_id = _integer_scalar(archive, f"{prefix}_traj_id", source)
        reference_length = _integer_scalar(
            archive,
            f"{prefix}_traj_length",
            source,
        )
        if reference_length <= 0:
            raise ValueError(f"{source} {prefix}_traj_length must be positive.")
        if subject_based and not 0 <= traj_id < motion_count:
            raise ValueError(
                f"{source} {prefix} has traj_id={traj_id}, outside the supplied motion list of length {motion_count}."
            )

        arrays = {
            field: np.asarray(archive[f"{prefix}_{field}"])
            for field in (
                "joint_positions",
                "traj_qpos",
                "muscle_activations",
                "touch_observations",
            )
        }
        for field, values in arrays.items():
            if values.ndim != 2:
                raise ValueError(f"{source} {prefix}_{field} must be two-dimensional, got {values.shape}.")
            if not np.isfinite(values).all():
                raise ValueError(f"{source} {prefix}_{field} contains non-finite values.")

        joint_length = len(arrays["joint_positions"])
        for field in ("muscle_activations", "touch_observations"):
            length = len(arrays[field])
            if length != joint_length:
                raise ValueError(f"{source} {prefix} has {joint_length} joint frames but {length} {field} frames.")
        if reference_length != len(arrays["traj_qpos"]):
            raise ValueError(
                f"{source} {prefix}_traj_length is {reference_length}, but "
                f"traj_qpos has {len(arrays['traj_qpos'])} frames."
            )
        if not 0 < joint_length <= reference_length:
            raise ValueError(f"{source} {prefix} rollout length {joint_length} must be in [1, {reference_length}].")

        if expected_joint_width is not None:
            for field in ("joint_positions", "traj_qpos"):
                width = arrays[field].shape[1]
                if width != expected_joint_width:
                    raise ValueError(
                        f"{source} {prefix}_{field} has {width} coordinates; "
                        f"MyoFullBody requires {expected_joint_width}."
                    )
        if expected_activation_width is not None and arrays["muscle_activations"].shape[1] != expected_activation_width:
            raise ValueError(
                f"{source} {prefix}_muscle_activations has "
                f"{arrays['muscle_activations'].shape[1]} columns; "
                f"MyoFullBody requires {expected_activation_width}."
            )
        if arrays["touch_observations"].shape[1] != 4:
            raise ValueError(
                f"{source} {prefix}_touch_observations has "
                f"{arrays['touch_observations'].shape[1]} columns; "
                "gait event extraction requires exactly 4 foot-contact sensors."
            )

        if embedded_kinetics:
            kinetics_keys = [f"{prefix}_{field}" for field in KINETICS_EPISODE_FIELDS]
            missing_kinetics = [key for key in kinetics_keys if key not in archive]
            if missing_kinetics:
                raise ValueError(
                    f"Rollout archive {source} is missing embedded kinetics key(s): {', '.join(missing_kinetics)}."
                )
            generalized = np.asarray(archive[f"{prefix}_generalized_actuator_forces"], dtype=float)
            sensors = np.asarray(archive[f"{prefix}_force_sensor_values"], dtype=float)
            steps = np.asarray(archive[f"{prefix}_kinetics_reference_steps"])
            assert kinetics_widths is not None
            if generalized.shape != (joint_length, kinetics_widths[0]):
                raise ValueError(
                    f"{source} {prefix}_generalized_actuator_forces must have shape "
                    f"{(joint_length, kinetics_widths[0])}, got {generalized.shape}."
                )
            if sensors.shape != (joint_length, kinetics_widths[1]):
                raise ValueError(
                    f"{source} {prefix}_force_sensor_values must have shape "
                    f"{(joint_length, kinetics_widths[1])}, got {sensors.shape}."
                )
            if not np.all(np.isfinite(generalized)) or not np.all(np.isfinite(sensors)):
                raise ValueError(f"{source} {prefix} embedded kinetics contains non-finite values.")
            if steps.shape != (joint_length,) or steps.dtype.kind not in "iu":
                raise ValueError(
                    f"{source} {prefix}_kinetics_reference_steps must be an integer vector of length {joint_length}."
                )
            if np.any(steps < 0) or np.any(steps >= reference_length) or np.any(np.diff(steps) <= 0):
                raise ValueError(
                    f"{source} {prefix}_kinetics_reference_steps must be strictly increasing within "
                    f"[0, {reference_length - 1}]."
                )

    return n_episodes


def validate_policy_rollouts(
    policies: list[PolicySpec],
    *,
    subject_based: bool,
    motion_count: int | None,
    expected_joint_names: tuple[str, ...],
    expected_joint_width: int,
    expected_activation_width: int,
    expected_motion_group: str | None,
) -> None:
    """Require every policy to use the same ordered reference cohort.

    A multi-policy comparison is only interpretable when episode ``i`` denotes
    the same trajectory in every archive.  This preflight compares trajectory
    IDs and the complete reference waveform, in addition to validating each
    archive's schema.
    """
    baseline_policy: str | None = None
    baseline_joint_names: tuple[str, ...] | None = None
    baseline_motion_group: str | None = None
    baseline_references: tuple[tuple[int, np.ndarray], ...] | None = None

    for policy in policies:
        source = Path(policy.trajectory_path)
        with np.load(source, allow_pickle=True) as archive:
            n_episodes = validate_rollout_archive(
                archive,
                source,
                subject_based=subject_based,
                motion_count=motion_count,
                expected_joint_names=expected_joint_names,
                expected_joint_width=expected_joint_width,
                expected_activation_width=expected_activation_width,
            )
            motion_group = _optional_string_scalar(
                archive,
                "motion_group",
                source,
            )
            if expected_motion_group is not None and motion_group is not None and motion_group != expected_motion_group:
                raise ValueError(
                    f"Policy {policy.id!r} was exported with motion_group "
                    f"{motion_group!r}, expected {expected_motion_group!r}."
                )

            joint_names = tuple(str(name) for name in np.asarray(archive["joint_names"]))
            references = tuple(
                (
                    _integer_scalar(archive, f"episode_{episode}_traj_id", source),
                    np.asarray(
                        archive[f"episode_{episode}_traj_qpos"],
                        dtype=float,
                    ).copy(),
                )
                for episode in range(n_episodes)
            )

        if baseline_references is None:
            baseline_policy = policy.id
            baseline_joint_names = joint_names
            baseline_motion_group = motion_group
            baseline_references = references
            continue

        if joint_names != baseline_joint_names:
            raise ValueError(f"Policy {policy.id!r} uses different joint coordinates from policy {baseline_policy!r}.")
        if motion_group is not None and baseline_motion_group is not None and motion_group != baseline_motion_group:
            raise ValueError(
                f"Policy {policy.id!r} uses motion_group {motion_group!r}, but "
                f"policy {baseline_policy!r} uses {baseline_motion_group!r}."
            )
        if len(references) != len(baseline_references):
            raise ValueError(
                f"Policy {policy.id!r} has {len(references)} episodes, but "
                f"policy {baseline_policy!r} has {len(baseline_references)}."
            )

        mismatches = [
            episode
            for episode, ((traj_id, reference), (base_id, base_reference)) in enumerate(
                zip(references, baseline_references, strict=True)
            )
            if traj_id != base_id or not np.array_equal(reference, base_reference)
        ]
        if mismatches:
            raise ValueError(
                f"Policy {policy.id!r} differs from policy {baseline_policy!r} "
                "in trajectory ID or reference waveform for episode(s) "
                f"{mismatches}."
            )
