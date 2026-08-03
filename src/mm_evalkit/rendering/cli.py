"""Command-line adapters for reusable synchronized video rendering."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from mm_evalkit.cli_options import add_dataset_dir, add_motion_group, add_video_output
from mm_evalkit.datasets.motion_sets import GAIT120_MOTION_SETS

from .api import render_video


def _add_camera(parser: argparse.ArgumentParser, *, azimuth: float, elevation: float, distance: float) -> None:
    """Declare the viewpoint options, whose useful defaults differ per body model."""
    parser.add_argument("--camera_azimuth", "--camera-azimuth", type=float, default=azimuth)
    parser.add_argument("--camera_elevation", "--camera-elevation", type=float, default=elevation)
    parser.add_argument("--camera_distance", "--camera-distance", type=float, default=distance)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mm-evalkit render",
        description="Render policy motion with synchronized human and artificial muscle signals.",
    )
    datasets = parser.add_subparsers(dest="dataset", required=True)

    ultra = datasets.add_parser("ultra-mocap", aliases=["ultramocap"], help="Clock-synchronized ULTRA-MoCap video.")
    ultra.add_argument("--trajectory_path", "--trajectory-path", dest="trajectory_path", required=True)
    ultra.add_argument("--episode", required=True, type=int)
    ultra.add_argument("--trial_stem", "--trial-stem", dest="trial_stem", required=True)
    add_dataset_dir(ultra, dataset="ultra-mocap", aliases=("emg_root",))
    ultra.add_argument("--cycles", type=int, default=4)
    ultra.add_argument("--cycle_offset", "--cycle-offset", dest="cycle_offset", type=int, default=0)
    _add_camera(ultra, azimuth=90.0, elevation=-8.0, distance=0.88)
    add_video_output(ultra)

    gait = datasets.add_parser("gait120", help="Gait-phase-synchronized Gait120 video.")
    gait.add_argument("--trajectory_path", "--trajectory-path", dest="trajectory_path", required=True)
    gait.add_argument("--episode", required=True, type=int)
    add_motion_group(gait, choices=GAIT120_MOTION_SETS, required=False, aliases=("motion_set",))
    gait.set_defaults(motion_group="GAIT120_TESTING")
    gait.add_argument("--gait_index", "--gait-index", dest="gait_index", type=int, choices=(0, 1), default=0)
    add_dataset_dir(gait, dataset="gait120")
    gait.add_argument("--channels", nargs="+", default=None)
    _add_camera(gait, azimuth=105.0, elevation=-12.0, distance=2.4)
    add_video_output(gait)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    options = vars(args)
    dataset = options.pop("dataset")
    output = options.pop("output")
    fps = options.pop("fps")
    if dataset == "ultramocap":
        dataset = "ultra-mocap"
    if dataset == "gait120" and options.get("channels") is None:
        options.pop("channels")
    render_video(dataset=dataset, output=output, fps=fps, **options)
    return 0
