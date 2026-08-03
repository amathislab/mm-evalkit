"""Policy selection for the ULTra-MoCap command-line adapter."""

from __future__ import annotations

from pathlib import Path

from mm_kit.policies import (
    PolicySpec,
    load_policy_specs,
    single_policy_spec,
    validate_policy_files,
)


def load_policies(
    config_path: str | Path | None,
    trajectory_path: str | Path | None,
) -> list[PolicySpec]:
    """Load exactly one explicit policy source and validate its files."""
    if (config_path is None) == (trajectory_path is None):
        raise ValueError("Provide exactly one of policy_config_file or trajectory_path.")
    policies = load_policy_specs(config_path) if config_path is not None else [single_policy_spec(trajectory_path)]
    for policy in policies:
        if policy.earlier_trials_path is not None:
            raise ValueError(
                f"Policy {policy.id!r} configures Gait120 trials 1-4 fields, which are not valid for ULTra-MoCap."
            )
        if policy.kinetics_summary_path is not None or policy.kinetics_swap_left_right:
            raise ValueError(
                f"Policy {policy.id!r} configures Gait120 kinetics fields, which are not valid for ULTra-MoCap."
            )
    validate_policy_files(policies)
    return policies
