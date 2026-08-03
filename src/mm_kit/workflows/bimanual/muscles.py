"""MyoBimanualArm actuators and the fixed electrode mapping."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .constants import EMG_MUSCLE_GROUPS
from .models import MuscleGroup


def load_muscle_names() -> list[str]:
    """Read actuator names in activation-column order from MyoBimanualArm."""
    import mujoco
    from musclemimic_models import get_xml_path

    model = mujoco.MjModel.from_xml_path(str(get_xml_path("bimanual")))
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index) for index in range(model.nu)]
    if not names or any(name is None or not name for name in names):
        raise ValueError("Every MyoBimanualArm actuator must have a non-empty name.")
    if len(set(names)) != len(names):
        raise ValueError("MyoBimanualArm actuator names must be unique.")
    return names


def build_muscle_groups(
    muscle_names: Sequence[str],
    suffix: str,
) -> tuple[tuple[MuscleGroup, ...], tuple[int, ...]]:
    """Build the fixed electrode mapping and stacked activation selection."""
    required = tuple(f"{muscle}{suffix}" for _, members in EMG_MUSCLE_GROUPS.values() for muscle in members)
    missing = tuple(name for name in required if name not in muscle_names)
    if missing:
        raise ValueError("MyoBimanualArm is missing actuator(s) required by the sEMG mapping: " + ", ".join(missing))

    groups: list[MuscleGroup] = []
    activation_columns: list[int] = []
    for channel, (label, members) in EMG_MUSCLE_GROUPS.items():
        columns = tuple(muscle_names.index(f"{muscle}{suffix}") for muscle in members)
        cycle_columns = tuple(
            range(
                len(activation_columns),
                len(activation_columns) + len(columns),
            )
        )
        activation_columns.extend(columns)
        groups.append(
            MuscleGroup(
                channel=channel,
                label=label,
                muscles=members,
                activation_columns=columns,
                cycle_columns=cycle_columns,
            )
        )
    return tuple(groups), tuple(activation_columns)


def combine_groups(
    activation_cycles: np.ndarray,
    groups: Sequence[MuscleGroup],
) -> np.ndarray:
    """Average only actuator heads mapped to the same surface electrode."""
    return np.stack(
        [activation_cycles[:, group.cycle_columns, :].mean(axis=1) for group in groups],
        axis=1,
    )
