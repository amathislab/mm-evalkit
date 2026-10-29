"""Command-line entrypoints for Gait120 and Wang analyses."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from mm_evalkit.activation import DEFAULT_MUSCLE_SIGNAL

from ...cli_options import (
    add_dataset_dir,
    add_motion_group,
    add_plots_dir,
    add_policy_source,
)
from .analysis import FullBodyAnalysis
from .constants import PRIMARY_MOTION_SETS


def build_parser(*, fixed_dataset: str | None = None) -> argparse.ArgumentParser:
    """Build the legacy or dataset-specific command-line parser.

    Dataset-specific entrypoints omit ``--dataset_name`` entirely.  A forwarded
    dataset flag is therefore rejected by argparse instead of silently
    overriding the registry selection.
    """
    parser = argparse.ArgumentParser(
        description="Compare full-body policies against human gait kinematics, optional kinetics, and EMG."
    )
    if fixed_dataset is None:
        parser.add_argument(
            "--dataset_name",
            choices=["Gait120", "Wang"],
            default="Gait120",
            help="Human dataset analyzed by this adapter (default: Gait120).",
        )
    add_dataset_dir(parser, dataset=fixed_dataset.casefold() if fixed_dataset else None)
    if fixed_dataset != "Wang":
        # Only trial-5 sets may decode the primary Gait120 rollout; Wang has no
        # motion list because its release carries no per-trial identity.
        add_motion_group(
            parser,
            choices=PRIMARY_MOTION_SETS,
            required=fixed_dataset == "Gait120",
            aliases=("motion_list_Gait120",),
        )
    else:
        parser.set_defaults(motion_group=None)
    add_policy_source(parser)
    add_plots_dir(parser, default="mm_evalkit_outputs/neuromuscular_analysis")
    if fixed_dataset != "Wang":
        parser.add_argument(
            "--kinematics_source",
            choices=["opensim-ik", "mfb-retargeted"],
            default="opensim-ik",
            help="Kinematic ground truth; EMG always comes from the human dataset.",
        )
        parser.add_argument(
            "--human_subject_policy",
            choices=["drop-and-report", "error"],
            default="drop-and-report",
            help="How Gait120 handles requested subjects that human preprocessing cannot load.",
        )
        parser.add_argument(
            "--kinetics_reference_dir",
            "--kinetics-reference-dir",
            help=(
                "Optional root containing S###/S###_Trial##_Step01_Step02.npz Gait120 force-plate and "
                "inverse-dynamics references; enables kinetics from the normal rollout export."
            ),
        )
    else:
        parser.set_defaults(
            human_subject_policy="drop-and-report",
            kinematics_source="opensim-ik",
            kinetics_reference_dir=None,
        )
    parser.add_argument(
        "--muscle_signal",
        "--muscle-signal",
        choices=["activation", "excitation"],
        default=DEFAULT_MUSCLE_SIGNAL,
        help="Policy muscle signal compared with EMG (default: excitation from muscle commands).",
    )
    parser.add_argument(
        "--kinetics_window",
        choices=["left_heel_strike", "right_stride"],
        default="left_heel_strike",
        help="Kinetics cycle: fixed 1 s from left heel strike, or the force-plate right stride.",
    )
    parser.add_argument(
        "--extended_channels",
        action="store_true",
        help="Gait120 only: also compare subtalar and MTP angles, and hip adduction/rotation, subtalar and MTP "
        "moments.",
    )
    return parser


def _run(argv: Sequence[str] | None, *, dataset_name: str | None) -> int:
    """Construct and execute one validated analysis from parsed CLI inputs."""
    args = build_parser(fixed_dataset=dataset_name).parse_args(argv)
    selected_dataset = dataset_name or args.dataset_name
    analysis = FullBodyAnalysis(
        dataset_name=selected_dataset,
        dataset_dir=args.dataset_dir,
        policy_config_file=args.policy_config_file,
        subject_based=selected_dataset == "Gait120",
        plots_dir=args.plots_dir,
        motion_group=args.motion_group,
        kinematics_source=args.kinematics_source,
        human_subject_policy=args.human_subject_policy,
        kinetics_reference_dir=args.kinetics_reference_dir,
        muscle_signal=args.muscle_signal,
        kinetics_window=args.kinetics_window,
        extended_channels=args.extended_channels,
    )
    analysis.run_gait_position_analyses()
    return 0


def gait120_main(argv: Sequence[str] | None = None) -> int:
    """Run the Gait120 adapter selected by the shared registry."""
    return _run(argv, dataset_name="Gait120")


def wang_main(argv: Sequence[str] | None = None) -> int:
    """Run the Wang adapter selected by the shared registry."""
    return _run(argv, dataset_name="Wang")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the legacy full-body CLI, which accepts ``--dataset_name``."""
    return _run(argv, dataset_name=None)
