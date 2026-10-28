"""Public entrypoint for the full-body gait workflow.

The state lives in :class:`~mm_evalkit.workflows.gait.state.GaitAnalysis`, loading in
:mod:`~mm_evalkit.workflows.gait.loading`, and the scientific stages in the sibling
modules. This class is the thin object those pieces are reached through, kept
because it is the constructor-style API the launchers and notebooks already use.
"""

from __future__ import annotations

from pathlib import Path

from mm_evalkit.activation import DEFAULT_MUSCLE_SIGNAL

from . import pipeline
from .config import FullBodyConfig
from .loading import load_gait_analysis
from .state import GaitAnalysis


class FullBodyAnalysis:
    """Compare policy gait kinematics, optional kinetics, and muscle activity."""

    def __init__(
        self,
        dataset_name: str = "Gait120",
        dataset_dir: str | Path | None = None,
        policy_config_file: str | Path | None = None,
        subject_based: bool | None = None,
        motion_group: str | None = None,
        plots_dir: str | Path = "mm_evalkit_outputs/neuromuscular_analysis",
        kinematics_source: str = "opensim-ik",
        human_subject_policy: str = "drop-and-report",
        kinetics_reference_dir: str | Path | None = None,
        *,
        muscle_signal: str = DEFAULT_MUSCLE_SIGNAL,
        kinetics_window: str = "left_heel_strike",
        extended_channels: bool = False,
        motion_list_Gait120: str | None = None,
    ) -> None:
        """Validate every configured input, then load rollout and human data.

        ``motion_list_Gait120`` is the historical keyword for ``motion_group``.
        It remains accepted so notebooks using the constructor-style API do not
        break while commands and new Python callers converge on one vocabulary.
        """
        if motion_group is not None and motion_list_Gait120 is not None and motion_group != motion_list_Gait120:
            raise ValueError("motion_group and its legacy alias motion_list_Gait120 must match when both are provided.")
        resolved_motion_group = motion_group if motion_group is not None else motion_list_Gait120
        self.config = FullBodyConfig.from_arguments(
            dataset_name=dataset_name,
            dataset_dir=dataset_dir,
            policy_config_file=policy_config_file,
            subject_based=subject_based,
            motion_set_name=resolved_motion_group,
            plots_root=plots_dir,
            kinematics_source=kinematics_source,
            human_subject_policy=human_subject_policy,
            kinetics_reference_dir=kinetics_reference_dir,
            muscle_signal=muscle_signal,
            kinetics_window=kinetics_window,
            extended_channels=extended_channels,
        )
        self.state: GaitAnalysis = load_gait_analysis(self.config)

    def run_gait_position_analyses(self, positions=(0, 1, None)):
        """Run the complete analysis for the requested gait positions."""
        return pipeline.run_gait_position_analyses(self.state, positions)


# Historical name retained at the package boundary. New code should use the
# workflow-specific ``FullBodyAnalysis`` name.
NeuroMuscularAnalysis = FullBodyAnalysis
