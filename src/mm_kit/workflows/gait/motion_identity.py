"""Decode subject and trial identity from an explicit Gait120 motion list."""

from __future__ import annotations

import re
from collections.abc import Sequence


def _motion_path(trajectory_id: int, motion_list: Sequence[str]) -> str:
    try:
        return str(motion_list[int(trajectory_id)])
    except (IndexError, TypeError) as exc:
        raise ValueError(
            f"Trajectory id {trajectory_id!r} is outside the supplied motion list of length {len(motion_list)}."
        ) from exc


def subject_from_trajectory(
    trajectory_id: int,
    motion_list: Sequence[str],
) -> int:
    """Return the ``S<NNN>`` component of a configured motion path."""
    motion = _motion_path(trajectory_id, motion_list)
    for component in motion.split("/"):
        if re.fullmatch(r"S\d+", component):
            return int(component[1:])
    raise ValueError(f"No subject component S<NNN> appears in motion {motion!r}.")


def trial_from_trajectory(
    trajectory_id: int,
    motion_list: Sequence[str],
) -> int:
    """Return the numeric ``Trial<TT>`` component of a configured motion path."""
    motion = _motion_path(trajectory_id, motion_list)
    match = re.search(r"(?:^|/)Trial(\d+)(?:/|$)", motion)
    if match is None:
        raise ValueError(f"No trial component Trial<TT> appears in motion {motion!r}.")
    return int(match.group(1))
