"""Strict policy configuration shared by both analysis workflows.

Policy identity is explicit and separate from its display label.  This matters
for multi-policy analyses: labels are presentation text, whereas ``id`` is the
stable key used in result tables and output directories.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_POLICY_COLORS = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#9467bd",
    "#8c564b",
)

_POLICY_ID = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_ROOT_FIELDS = {"policies"}
_POLICY_FIELDS = {
    "id",
    "label",
    "trajectory_data_path",
    "trajectory_data_path_trials_1_4",
    "motion_list_trials_1_4",
    "kinetics_summary_path",
    "kinetics_swap_left_right",
    "color",
    "linestyle",
}


@dataclass(frozen=True)
class PolicySpec:
    """One policy rollout and its presentation metadata.

    The two ``trials_1_4`` fields are used only by the full-body Gait120
    sensitivity analysis.  They must either both be present or both be absent.
    """

    id: str
    label: str
    trajectory_path: Path
    color: str
    linestyle: str = "-"
    earlier_trials_path: Path | None = None
    earlier_trials_motion_set: str | None = None
    kinetics_summary_path: Path | None = None
    kinetics_swap_left_right: bool = False

    @property
    def slug(self) -> str:
        """Filesystem-safe, stable output key."""
        return self.id


def _required_text(entry: dict[str, Any], field: str, location: str) -> str:
    value = entry.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}.{field} must be a non-empty string.")
    return value.strip()


def _optional_text(entry: dict[str, Any], field: str, location: str) -> str | None:
    value = entry.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}.{field} must be a non-empty string when provided.")
    return value.strip()


def _optional_bool(entry: dict[str, Any], field: str, location: str, *, default: bool = False) -> bool:
    value = entry.get(field, default)
    if not isinstance(value, bool):
        raise ValueError(f"{location}.{field} must be true or false when provided.")
    return value


def load_policy_specs(config_path: str | Path) -> list[PolicySpec]:
    """Parse a policy YAML file and reject ambiguous or misspelled fields."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Policy configuration does not exist: {path}")

    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{path} must contain a YAML mapping with a 'policies' list.")

    unknown_root = set(document) - _ROOT_FIELDS
    if unknown_root:
        fields = ", ".join(sorted(unknown_root))
        raise ValueError(f"{path} contains unknown top-level field(s): {fields}.")

    entries = document.get("policies")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path}.policies must be a non-empty list.")

    policies: list[PolicySpec] = []
    seen_ids: set[str] = set()
    for index, entry in enumerate(entries):
        location = f"{path}.policies[{index}]"
        if not isinstance(entry, dict):
            raise ValueError(f"{location} must be a mapping.")

        unknown_fields = set(entry) - _POLICY_FIELDS
        if unknown_fields:
            fields = ", ".join(sorted(unknown_fields))
            raise ValueError(f"{location} contains unknown field(s): {fields}.")

        policy_id = _required_text(entry, "id", location)
        if not _POLICY_ID.fullmatch(policy_id):
            raise ValueError(
                f"{location}.id must contain only lowercase letters, digits, '-' or '_', "
                "and must start with a letter or digit."
            )
        if policy_id in seen_ids:
            raise ValueError(f"{location}.id duplicates policy id {policy_id!r}.")
        seen_ids.add(policy_id)

        earlier_path = _optional_text(entry, "trajectory_data_path_trials_1_4", location)
        earlier_motion_set = _optional_text(entry, "motion_list_trials_1_4", location)
        kinetics_summary = _optional_text(entry, "kinetics_summary_path", location)
        kinetics_swap = _optional_bool(entry, "kinetics_swap_left_right", location)
        if (earlier_path is None) != (earlier_motion_set is None):
            raise ValueError(
                f"{location} must provide trajectory_data_path_trials_1_4 and motion_list_trials_1_4 together."
            )

        policies.append(
            PolicySpec(
                id=policy_id,
                label=_required_text(entry, "label", location),
                trajectory_path=Path(_required_text(entry, "trajectory_data_path", location)),
                color=_optional_text(entry, "color", location)
                or DEFAULT_POLICY_COLORS[index % len(DEFAULT_POLICY_COLORS)],
                linestyle=_optional_text(entry, "linestyle", location) or "-",
                earlier_trials_path=Path(earlier_path) if earlier_path is not None else None,
                earlier_trials_motion_set=earlier_motion_set,
                kinetics_summary_path=Path(kinetics_summary) if kinetics_summary is not None else None,
                kinetics_swap_left_right=kinetics_swap,
            )
        )

    return policies


def single_policy_spec(trajectory_path: str | Path) -> PolicySpec:
    """Build the explicit one-policy configuration used by the bimanual CLI."""
    path = Path(trajectory_path)
    label = path.parent.name or path.stem
    policy_id = re.sub(r"[^a-z0-9]+", "-", label.casefold()).strip("-") or "policy"
    return PolicySpec(
        id=policy_id,
        label=label,
        trajectory_path=path,
        color=DEFAULT_POLICY_COLORS[0],
    )


def validate_policy_files(policies: list[PolicySpec]) -> None:
    """Fail before analysis if any configured rollout file is unavailable."""
    missing: list[Path] = []
    for policy in policies:
        if not policy.trajectory_path.is_file():
            missing.append(policy.trajectory_path)
        if policy.earlier_trials_path is not None and not policy.earlier_trials_path.is_file():
            missing.append(policy.earlier_trials_path)
        if policy.kinetics_summary_path is not None and not policy.kinetics_summary_path.is_file():
            missing.append(policy.kinetics_summary_path)
    if missing:
        listed = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Configured policy rollout file(s) do not exist:\n{listed}")
