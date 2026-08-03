"""Validated rollout loading and explicit ULTra-MoCap trial provenance."""

from __future__ import annotations

from collections.abc import Collection, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from mm_evalkit.activation import activation_sample_phase
from mm_evalkit.policies import PolicySpec

from .constants import CONTROL_FREQUENCY_HZ, CONTROL_TIMESTEP
from .models import RolloutMetadata

_EPISODE_FIELDS = (
    "traj_id",
    "traj_length",
    "joint_positions",
    "muscle_activations",
    "traj_qpos",
)


def _scalar(data: Any, key: str, expected_type: type) -> int | float:
    """Read one scalar while rejecting shape and integer-type mistakes."""
    array = np.asarray(data[key])
    if array.size != 1:
        raise ValueError(f"Rollout field {key!r} must contain one scalar value.")
    value = array.item()
    if expected_type is int:
        if isinstance(value, (bool, np.bool_)) or int(value) != value:
            raise ValueError(f"Rollout field {key!r} must be an integer.")
        return int(value)
    return float(value)


def _require_keys(data: Any, keys: Collection[str], context: str) -> None:
    missing = sorted(set(keys) - set(data.files))
    if missing:
        raise ValueError(f"{context} is missing required field(s): {', '.join(missing)}.")


def _embedded_motion_group(data: Any, path: Path) -> str | None:
    """Return optional provenance written by current trajectory exporters."""
    if "motion_group" not in data:
        return None
    value = np.asarray(data["motion_group"])
    if value.size != 1:
        raise ValueError(f"{path}: motion_group must contain one scalar value.")
    group = value.item()
    if group is None or str(group).strip() == "":
        return None
    return str(group)


def validate_rollout(
    data: np.lib.npyio.NpzFile,
    path: str | Path,
    *,
    expected_activation_count: int | None = None,
    expected_motion_group: str | None = None,
) -> RolloutMetadata:
    """Validate every rollout field consumed by the analysis."""
    rollout_path = Path(path)
    _require_keys(data, ("n_episodes", "dt", "joint_names"), str(rollout_path))
    embedded_motion_group = _embedded_motion_group(data, rollout_path)
    if (
        embedded_motion_group is not None
        and expected_motion_group is not None
        and embedded_motion_group != expected_motion_group
    ):
        raise ValueError(
            f"{rollout_path}: archive motion_group is "
            f"{embedded_motion_group!r}, but the analysis requested "
            f"{expected_motion_group!r}."
        )

    n_episodes = int(_scalar(data, "n_episodes", int))
    if n_episodes <= 0:
        raise ValueError(f"{rollout_path}: n_episodes must be positive.")
    activation_sample_phase(data)
    dt = float(_scalar(data, "dt", float))
    if not np.isfinite(dt) or not np.isclose(dt, CONTROL_TIMESTEP):
        raise ValueError(
            f"{rollout_path}: dt must be {CONTROL_TIMESTEP:g} s ({CONTROL_FREQUENCY_HZ:g} Hz), got {dt!r}."
        )

    raw_names = np.asarray(data["joint_names"])
    if raw_names.ndim != 1 or not len(raw_names):
        raise ValueError(f"{rollout_path}: joint_names must be a non-empty one-dimensional array.")
    joint_names = tuple(str(name) for name in raw_names)
    if any(not name for name in joint_names) or len(set(joint_names)) != len(joint_names):
        raise ValueError(f"{rollout_path}: joint_names must be non-empty and unique.")

    episode_keys = {f"episode_{episode}_{field}" for episode in range(n_episodes) for field in _EPISODE_FIELDS}
    _require_keys(data, episode_keys, str(rollout_path))

    activation_count: int | None = None
    for episode in range(n_episodes):
        prefix = f"episode_{episode}"
        traj_id = int(_scalar(data, f"{prefix}_traj_id", int))
        if traj_id < 0:
            raise ValueError(f"{rollout_path}: {prefix}_traj_id must be non-negative.")
        reference_length = int(_scalar(data, f"{prefix}_traj_length", int))
        policy_qpos = np.asarray(data[f"{prefix}_joint_positions"])
        activations = np.asarray(data[f"{prefix}_muscle_activations"])
        reference_qpos = np.asarray(data[f"{prefix}_traj_qpos"])

        for field, array in (
            ("joint_positions", policy_qpos),
            ("muscle_activations", activations),
            ("traj_qpos", reference_qpos),
        ):
            if array.ndim != 2:
                raise ValueError(f"{rollout_path}: {prefix}_{field} must be two-dimensional, got shape {array.shape}.")
            if not np.isfinite(array).all():
                raise ValueError(f"{rollout_path}: {prefix}_{field} contains non-finite values.")

        if policy_qpos.shape[1] != len(joint_names):
            raise ValueError(
                f"{rollout_path}: {prefix}_joint_positions has "
                f"{policy_qpos.shape[1]} columns for {len(joint_names)} joint names."
            )
        if reference_qpos.shape[1] != len(joint_names):
            raise ValueError(
                f"{rollout_path}: {prefix}_traj_qpos has "
                f"{reference_qpos.shape[1]} columns for {len(joint_names)} joint names."
            )
        if reference_length != len(reference_qpos):
            raise ValueError(
                f"{rollout_path}: {prefix}_traj_length is {reference_length}, "
                f"but traj_qpos has {len(reference_qpos)} rows."
            )
        if not 0 < len(policy_qpos) <= reference_length:
            raise ValueError(
                f"{rollout_path}: {prefix}_joint_positions length "
                f"{len(policy_qpos)} must be in [1, {reference_length}]."
            )
        if len(activations) != len(policy_qpos):
            raise ValueError(
                f"{rollout_path}: {prefix}_muscle_activations has "
                f"{len(activations)} rows but joint_positions has {len(policy_qpos)}."
            )

        if activation_count is None:
            activation_count = int(activations.shape[1])
        elif activations.shape[1] != activation_count:
            raise ValueError(
                f"{rollout_path}: activation width changes at episode {episode} "
                f"({activation_count} to {activations.shape[1]})."
            )

    assert activation_count is not None
    if expected_activation_count is not None and activation_count != expected_activation_count:
        raise ValueError(
            f"{rollout_path}: rollout has {activation_count} activation columns, "
            f"but MyoBimanualArm has {expected_activation_count} actuators."
        )
    return RolloutMetadata(
        path=rollout_path,
        n_episodes=n_episodes,
        dt=dt,
        joint_names=joint_names,
        activation_count=activation_count,
    )


@contextmanager
def open_rollout(
    path: str | Path,
    *,
    expected_activation_count: int | None = None,
    expected_motion_group: str | None = None,
) -> Iterator[tuple[np.lib.npyio.NpzFile, RolloutMetadata]]:
    """Open and validate a rollout, closing it when the caller leaves the block."""
    rollout_path = Path(path)
    if not rollout_path.is_file():
        raise FileNotFoundError(f"Policy rollout does not exist: {rollout_path}")
    data = np.load(rollout_path, allow_pickle=True)
    try:
        metadata = validate_rollout(
            data,
            rollout_path,
            expected_activation_count=expected_activation_count,
            expected_motion_group=expected_motion_group,
        )
        yield data, metadata
    finally:
        data.close()


def _validate_untagged_motion_group(
    trajectory: Any,
    metadata: RolloutMetadata,
    motion_paths: Sequence[str],
) -> None:
    """Accept legacy provenance only when it identifies the whole group.

    Current exports store ``motion_group`` explicitly.  For older untagged
    archives, complete one-to-one coverage is the only evidence that their
    integer ``traj_id`` values use the declared group's indexing.
    """
    if _embedded_motion_group(trajectory, metadata.path) is not None:
        return

    group_size = len(motion_paths)
    if metadata.n_episodes != group_size:
        raise ValueError(
            f"{metadata.path}: archive has no motion_group provenance and "
            f"contains {metadata.n_episodes} episodes, but the declared motion "
            f"group has {group_size}. An untagged archive must cover the "
            "complete declared group exactly once."
        )

    trajectory_ids = [
        int(_scalar(trajectory, f"episode_{episode}_traj_id", int)) for episode in range(metadata.n_episodes)
    ]
    if sorted(trajectory_ids) != list(range(group_size)):
        raise ValueError(
            f"{metadata.path}: archive has no motion_group provenance and its "
            f"traj_id values do not cover 0..{group_size - 1} exactly once."
        )


def resolve_trial_stems(
    trajectory: Any,
    metadata: RolloutMetadata,
    motion_paths: Sequence[str],
    episodes: Collection[int] | None = None,
) -> dict[int, str]:
    """Map each requested episode's exported ``traj_id`` to one declared motion.

    The evaluation motion group is an explicit analysis input.  This avoids
    reconstructing provenance from waveform similarity, which can silently pick
    a different recording when two motions have similar duration and shape.
    """
    if not motion_paths:
        raise ValueError("The selected C3D motion group is empty.")
    _validate_untagged_motion_group(trajectory, metadata, motion_paths)
    requested = tuple(range(metadata.n_episodes) if episodes is None else episodes)
    invalid = tuple(episode for episode in requested if not 0 <= episode < metadata.n_episodes)
    if invalid:
        raise ValueError(f"Episode indices {invalid} are outside rollout range [0, {metadata.n_episodes - 1}].")

    matches = {}
    for episode in requested:
        traj_id = int(_scalar(trajectory, f"episode_{episode}_traj_id", int))
        if not 0 <= traj_id < len(motion_paths):
            raise ValueError(
                f"Episode {episode} has traj_id={traj_id}, outside the declared "
                f"motion group of length {len(motion_paths)}."
            )
        stem = Path(motion_paths[traj_id]).name
        if not stem:
            raise ValueError(f"Motion group entry {traj_id} has no trial stem: {motion_paths[traj_id]!r}.")
        matches[episode] = stem
    return matches


def validate_policy_rollouts(
    policies: Collection[PolicySpec],
    expected_activation_count: int,
    expected_motion_group: str,
    episodes: Collection[int] | None = None,
) -> tuple[str, ...]:
    """Require every policy to use the same requested references and IDs."""
    expected_joint_names: tuple[str, ...] | None = None
    expected_references: dict[int, np.ndarray] | None = None
    expected_traj_ids: dict[int, int] | None = None
    expected_episode_count: int | None = None
    reference_policy_id: str | None = None
    for policy in policies:
        with open_rollout(
            policy.trajectory_path,
            expected_activation_count=expected_activation_count,
            expected_motion_group=expected_motion_group,
        ) as (trajectory, metadata):
            requested = tuple(range(metadata.n_episodes) if episodes is None else episodes)
            invalid = tuple(episode for episode in requested if not 0 <= episode < metadata.n_episodes)
            if invalid:
                raise ValueError(
                    f"Policy {policy.id!r} does not contain requested episode(s) {invalid}; "
                    f"valid range is [0, {metadata.n_episodes - 1}]."
                )
            if expected_joint_names is None:
                expected_joint_names = metadata.joint_names
                expected_references = {
                    episode: np.asarray(trajectory[f"episode_{episode}_traj_qpos"], dtype=float).copy()
                    for episode in range(metadata.n_episodes)
                    if episode in requested
                }
                expected_traj_ids = {
                    episode: int(_scalar(trajectory, f"episode_{episode}_traj_id", int)) for episode in requested
                }
                expected_episode_count = metadata.n_episodes
                reference_policy_id = policy.id
                continue

            if metadata.joint_names != expected_joint_names:
                raise ValueError(
                    f"Policy {policy.id!r} uses different joint_names from the "
                    "first policy; comparison requires identical coordinates."
                )
            assert expected_references is not None and expected_traj_ids is not None
            if episodes is None and metadata.n_episodes != expected_episode_count:
                raise ValueError(
                    f"Policy {policy.id!r} has {metadata.n_episodes} episodes, "
                    f"but policy {reference_policy_id!r} has "
                    f"{expected_episode_count}."
                )

            mismatched = [
                episode
                for episode, expected in expected_references.items()
                if int(_scalar(trajectory, f"episode_{episode}_traj_id", int)) != expected_traj_ids[episode]
                or not np.array_equal(
                    np.asarray(
                        trajectory[f"episode_{episode}_traj_qpos"],
                        dtype=float,
                    ),
                    expected,
                )
            ]
            if mismatched:
                preview = ", ".join(str(episode) for episode in mismatched[:8])
                suffix = "..." if len(mismatched) > 8 else ""
                raise ValueError(
                    f"Policy {policy.id!r} has different trajectory IDs or reference "
                    f"trajectories at requested episode(s) {preview}{suffix}; all "
                    "policies in one comparison must use the same motion group and "
                    "episode order."
                )
    assert expected_joint_names is not None
    return expected_joint_names
