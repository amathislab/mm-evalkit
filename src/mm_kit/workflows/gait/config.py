"""Validated configuration for the Gait120 and Wang workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from mm_kit.paths import resolve_dataset_path
from mm_kit.policies import (
    PolicySpec,
    load_policy_specs,
    validate_policy_files,
)

from .constants import MOTION_SETS, PRIMARY_MOTION_SETS

DatasetName = Literal["Gait120", "Wang"]
KinematicsSource = Literal["opensim-ik", "mfb-retargeted"]
HumanSubjectPolicy = Literal["drop-and-report", "error"]


@dataclass(frozen=True)
class FullBodyConfig:
    """Inputs that define one reproducible full-body analysis.

    ``human_subject_policy`` controls the only dataset-level omission allowed
    by the loader.  ``drop-and-report`` preserves the historical Gait120
    behavior while making the choice explicit; ``error`` requires every
    requested subject to be loadable.
    """

    dataset_name: DatasetName
    dataset_dir: Path
    policy_config_file: Path
    plots_root: Path
    subject_based: bool
    motion_set_name: str | None
    kinematics_source: KinematicsSource
    human_subject_policy: HumanSubjectPolicy = "drop-and-report"
    kinetics_reference_dir: Path | None = None

    @classmethod
    def from_arguments(
        cls,
        *,
        dataset_name: str,
        dataset_dir: str | Path | None,
        policy_config_file: str | Path | None,
        subject_based: bool | None,
        motion_set_name: str | None,
        plots_root: str | Path,
        kinematics_source: str,
        human_subject_policy: str,
        kinetics_reference_dir: str | Path | None = None,
    ) -> FullBodyConfig:
        """Validate public API arguments before any rollout is opened."""
        if dataset_name not in {"Gait120", "Wang"}:
            raise ValueError(f"dataset_name must be 'Gait120' or 'Wang', got {dataset_name!r}.")
        dataset_path = resolve_dataset_path(dataset_name.casefold(), dataset_dir)
        if policy_config_file is None:
            raise ValueError("policy_config_file is required.")

        if kinematics_source not in {"opensim-ik", "mfb-retargeted"}:
            raise ValueError(f"kinematics_source must be 'opensim-ik' or 'mfb-retargeted', got {kinematics_source!r}.")
        if human_subject_policy not in {"drop-and-report", "error"}:
            raise ValueError(
                f"human_subject_policy must be 'drop-and-report' or 'error', got {human_subject_policy!r}."
            )

        resolved_subject_based = dataset_name == "Gait120" if subject_based is None else subject_based
        if dataset_name == "Wang" and resolved_subject_based:
            raise ValueError("Wang has no subject identity mapping; set subject_based=False explicitly.")
        if kinematics_source == "mfb-retargeted" and (dataset_name != "Gait120" or not resolved_subject_based):
            raise ValueError(
                "mfb-retargeted kinematic ground truth is only defined for subject-based Gait120 analyses."
            )

        selected_motion_set: str | None = None
        if dataset_name == "Gait120":
            if motion_set_name is None:
                raise ValueError(
                    "motion_list_Gait120 is required for Gait120 so rollout "
                    "traj_id values have explicit subject/trial provenance."
                )
            if motion_set_name not in PRIMARY_MOTION_SETS:
                available = ", ".join(sorted(PRIMARY_MOTION_SETS))
                raise ValueError(
                    "The primary Gait120 comparison requires a trial-5 motion "
                    f"set; got {motion_set_name!r}. Choose one of: {available}."
                )
            selected_motion_set = motion_set_name

        plots_path = Path(plots_root)
        if plots_path.exists() and not plots_path.is_dir():
            raise NotADirectoryError(f"plots_root is not a directory: {plots_path}")
        kinetics_path = (
            Path(kinetics_reference_dir).expanduser().resolve() if kinetics_reference_dir is not None else None
        )
        if kinetics_path is not None and not kinetics_path.is_dir():
            raise FileNotFoundError(f"Gait120 kinetics reference directory does not exist: {kinetics_path}")
        if kinetics_path is not None and dataset_name != "Gait120":
            raise ValueError("kinetics_reference_dir is only valid for Gait120 analyses.")

        return cls(
            dataset_name=dataset_name,
            dataset_dir=dataset_path,
            policy_config_file=Path(policy_config_file),
            plots_root=plots_path,
            subject_based=resolved_subject_based,
            motion_set_name=selected_motion_set,
            kinematics_source=kinematics_source,
            human_subject_policy=human_subject_policy,
            kinetics_reference_dir=kinetics_path,
        )

    def load_policies(self) -> list[PolicySpec]:
        """Load and validate all policy files and dataset-specific options."""
        policies = load_policy_specs(self.policy_config_file)
        validate_policy_files(policies)

        configured_kinetics = [policy.kinetics_summary_path is not None for policy in policies]
        if any(configured_kinetics) and not all(configured_kinetics) and self.kinetics_reference_dir is None:
            missing = [policy.id for policy in policies if policy.kinetics_summary_path is None]
            raise ValueError(
                "Kinetics must be configured for every compared policy; missing kinetics_summary_path for: "
                f"{', '.join(missing)}."
            )
        kinetics_requested = any(configured_kinetics) or self.kinetics_reference_dir is not None
        if kinetics_requested and (self.dataset_name != "Gait120" or not self.subject_based):
            raise ValueError("Kinetics is only valid for subject-based Gait120 analyses.")

        for policy in policies:
            if self.dataset_name == "Wang" and policy.earlier_trials_path is not None:
                raise ValueError(f"Policy {policy.id!r} configures Gait120 trials 1-4 data for Wang.")
            if policy.earlier_trials_motion_set is not None and policy.earlier_trials_motion_set not in MOTION_SETS:
                available = ", ".join(sorted(MOTION_SETS))
                raise ValueError(
                    f"Policy {policy.id!r} uses unknown trials 1-4 motion set "
                    f"{policy.earlier_trials_motion_set!r}; choose one of: {available}."
                )
            if policy.kinetics_swap_left_right and policy.kinetics_summary_path is None:
                raise ValueError(
                    f"Policy {policy.id!r} sets kinetics_swap_left_right without a compatibility kinetics summary."
                )
        return policies
