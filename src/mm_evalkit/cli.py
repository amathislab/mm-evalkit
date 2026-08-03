"""Top-level command dispatcher for evaluation and synchronized rendering."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .registry import get_dataset, registered_datasets


def _evaluate_parser() -> argparse.ArgumentParser:
    adapters = registered_datasets()
    lines = [f"  {adapter.name:<12} {adapter.workflow:<9} {adapter.description}" for adapter in adapters]
    parser = argparse.ArgumentParser(
        prog="mm-evalkit evaluate",
        description="Run one kinematics, kinetics, or neuromuscular analysis through its dataset adapter.",
        epilog="Available datasets:\n" + "\n".join(lines),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("dataset", nargs="?", metavar="DATASET", help="Dataset key or alias.")
    parser.add_argument("dataset_args", nargs=argparse.REMAINDER, help="Arguments forwarded to the dataset adapter.")
    parser.add_argument("--list-datasets", action="store_true", help="List dataset keys and aliases, then exit.")
    return parser


def evaluate_main(argv: Sequence[str] | None = None) -> int:
    """Resolve one dataset and forward its remaining arguments unchanged."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = _evaluate_parser()
    if not arguments or arguments[0] in {"-h", "--help"}:
        parser.print_help()
        return 0
    if arguments[0] == "--list-datasets":
        for adapter in registered_datasets():
            aliases = f" (aliases: {', '.join(adapter.aliases)})" if adapter.aliases else ""
            print(f"{adapter.name}: {adapter.workflow}{aliases}")
        return 0

    dataset_name = arguments.pop(0)
    try:
        adapter = get_dataset(dataset_name)
    except KeyError as exc:
        parser.error(exc.args[0])
    return adapter.run(arguments)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mm-evalkit",
        description="Evaluate and render kinematics, kinetics, and neuromuscular policy rollouts.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("evaluate", "render", "paths"),
        help="Run an analysis, render a synchronized video, or configure dataset paths.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch the stable public commands without importing heavy backends."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = _parser()
    if not arguments or arguments[0] in {"-h", "--help"}:
        parser.print_help()
        return 0

    command = arguments.pop(0)
    if command == "evaluate":
        return evaluate_main(arguments)
    if command == "render":
        from .rendering.cli import main as render_main

        return render_main(arguments)
    if command in {"paths", "config"}:  # "config" is the name this command shipped under
        from .paths_cli import main as paths_main

        return paths_main(arguments)
    parser.error(f"unknown command: {command}")
