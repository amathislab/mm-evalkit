"""Inspect and update dataset paths used by mm-kit."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from .paths import DATASET_PATH_KEYS, config_path, load_path_config, set_dataset_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mm-kit paths", description=__doc__)
    commands = parser.add_subparsers(dest="paths_command", required=True)
    commands.add_parser("show", help="Show dataset keys from the active YAML file.")
    setter = commands.add_parser("set", help="Validate and persist one dataset directory.")
    setter.add_argument("dataset", choices=("gait120", "wang", "ultra-mocap"))
    setter.add_argument("path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.paths_command == "set":
        resolved = set_dataset_path(args.dataset, args.path)
        print(f"Set {DATASET_PATH_KEYS[args.dataset]}={resolved} in {config_path()}")
        return 0
    document = load_path_config()
    print(f"Config: {config_path()}")
    for dataset in ("gait120", "wang", "ultra-mocap"):
        key = DATASET_PATH_KEYS[dataset]
        print(f"{key}: {document.get(key, '<not set>')}")
    return 0
