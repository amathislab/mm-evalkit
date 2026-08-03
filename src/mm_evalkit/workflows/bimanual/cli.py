"""Command-line interface for the ULTra-MoCap analysis."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from ...cli_options import (
    add_dataset_dir,
    add_motion_group,
    add_plots_dir,
    add_policy_source,
)
from ...datasets.motion_sets import ULTRAMOCAP_MOTION_SETS
from ...datasets.ultra_mocap import validate_emg_roots
from ...paths import resolve_dataset_path
from .config import load_policies
from .models import AnalysisConfig
from .pipeline import run_analysis


def build_parser() -> argparse.ArgumentParser:
    """Describe only user-facing inputs; scientific work lives in the pipeline."""
    parser = argparse.ArgumentParser(
        description=("Compare MyoBimanualArm policy activation with timestamp-aligned ULTra-MoCap kinematics and EMG.")
    )
    add_policy_source(parser, allow_single_rollout=True)
    add_dataset_dir(parser, dataset="ultra-mocap", aliases=("ultramocap_roots",), multiple=True)
    add_motion_group(parser, choices=ULTRAMOCAP_MOTION_SETS)
    add_plots_dir(parser, default="mm_evalkit_outputs/ultramocap_bimanual")
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        help="Optional explicit rollout episode indices.",
    )
    parser.add_argument(
        "--arm",
        choices=("right", "left"),
        default="right",
        help="Arm corresponding to the recorded electrodes.",
    )
    parser.add_argument(
        "--all_joints",
        action="store_true",
        help="Report every rollout joint instead of the seven primary arm joints.",
    )
    parser.add_argument(
        "--raw_trials",
        action="store_true",
        help="Also save unaveraged per-trial waveforms.",
    )
    return parser


def parse_options(argv: Sequence[str] | None = None) -> AnalysisConfig:
    """Parse paths and fail before the scientific pipeline sees invalid inputs."""
    args = build_parser().parse_args(argv)
    policies = load_policies(args.policy_config_file, args.trajectory_path)
    configured_roots = args.dataset_dir
    if configured_roots is None:
        configured_roots = [resolve_dataset_path("ultra-mocap")]
    emg_roots = validate_emg_roots(configured_roots)
    motion_paths = ULTRAMOCAP_MOTION_SETS[args.motion_group]

    episodes = tuple(args.episodes) if args.episodes is not None else None
    if episodes is not None:
        if any(episode < 0 for episode in episodes):
            raise ValueError("Episode indices must be non-negative.")
        if len(set(episodes)) != len(episodes):
            raise ValueError("Episode indices must not contain duplicates.")

    plots_dir = Path(args.plots_dir)
    if plots_dir.exists() and not plots_dir.is_dir():
        raise NotADirectoryError(f"plots_dir is not a directory: {plots_dir}")
    if plots_dir.exists() and any(plots_dir.iterdir()):
        raise FileExistsError(
            f"plots_dir is not empty: {plots_dir}. Choose a new or empty "
            "directory so stale artifacts cannot be mistaken for this run."
        )

    return AnalysisConfig(
        policies=tuple(policies),
        emg_roots=emg_roots,
        motion_group=args.motion_group,
        motion_paths=motion_paths,
        plots_dir=plots_dir,
        episodes=episodes,
        arm=args.arm,
        all_joints=args.all_joints,
        raw_trials=args.raw_trials,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the complete ULTra-MoCap analysis."""
    return run_analysis(parse_options(argv))
