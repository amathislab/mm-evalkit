"""Subject-cohort rules for comparable full-body policy analyses."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence


def shared_policy_subjects(
    human_subjects: Sequence[int],
    policy_subjects: Mapping[str, Iterable[int]],
    *,
    selection: str,
) -> tuple[int, ...]:
    """Return human subjects represented by every compared policy."""
    ordered_human = tuple(int(subject) for subject in human_subjects)
    if not ordered_human:
        raise ValueError(f"No human subjects are available for {selection}.")
    if not policy_subjects:
        raise ValueError("At least one policy is required for a policy comparison.")

    human_set = set(ordered_human)
    shared = human_set.copy()
    for policy_id, subjects in policy_subjects.items():
        available = {int(subject) for subject in subjects} & human_set
        if not available:
            raise ValueError(
                f"Policy {policy_id!r} has no retained rollout cycles for {selection} and the available human cohort."
            )
        shared &= available

    if not shared:
        policy_ids = ", ".join(repr(policy_id) for policy_id in policy_subjects)
        raise ValueError(f"Policies {policy_ids} have no shared subjects with retained cycles for {selection}.")
    return tuple(subject for subject in ordered_human if subject in shared)
