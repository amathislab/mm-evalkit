"""Command-line options shared by every workflow and render adapter.

Workflows differ in what they analyse, not in how a run is pointed at its
inputs. Defining those inputs once means a new adapter inherits the flag names,
help text, and defaults the existing ones use, and a user who has learned one
adapter can predict the next.

Canonical spellings are snake_case. The kebab-case spelling of each flag is
registered as an alias, so ``mm-kit evaluate`` and ``mm-kit render`` accept the
same command line.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from .paths import DATASET_PATH_KEYS


def _names(canonical: str, *aliases: str) -> tuple[str, ...]:
    """Return every declared spelling plus its kebab-case counterpart."""
    ordered: list[str] = []
    for name in (canonical, *aliases):
        ordered.extend((name, name.replace("_", "-")))
    return tuple(f"--{name}" for name in dict.fromkeys(ordered))


def add_policy_source(
    parser: argparse.ArgumentParser,
    *,
    allow_single_rollout: bool = False,
) -> None:
    """Declare where the compared policy rollouts come from.

    A workflow that can analyse one rollout on its own takes either a single
    ``--trajectory_path`` or a policy YAML; a workflow that always compares
    policies requires the YAML.
    """
    config_help = "YAML file listing the policy rollout exports to compare."
    if not allow_single_rollout:
        parser.add_argument(*_names("policy_config_file"), required=True, help=config_help)
        return

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(*_names("trajectory_path"), help="One exported rollout NPZ.")
    source.add_argument(*_names("policy_config_file"), help=config_help)


def add_dataset_dir(
    parser: argparse.ArgumentParser,
    *,
    dataset: str | None = None,
    aliases: Sequence[str] = (),
    multiple: bool = False,
) -> None:
    """Declare the human dataset location, defaulting to the configured path.

    ``dataset`` is ``None`` for an adapter that selects its dataset at parse
    time and so cannot name one configuration key up front. ``multiple`` is for
    datasets released as several sibling roots; the parsed value is a list in
    that case and a single path otherwise.
    """
    default_source = DATASET_PATH_KEYS[dataset] if dataset else "the dataset's MUSCLEMIMIC_*_DATA_PATH key"
    described = f"human {dataset}" if dataset else "human dataset"
    parser.add_argument(
        *_names("dataset_dir", *aliases),
        nargs="+" if multiple else None,
        help=f"Directory containing the {described} data. Defaults to {default_source}.",
    )


def add_motion_group(
    parser: argparse.ArgumentParser,
    *,
    choices: Sequence[str],
    required: bool = True,
    aliases: Sequence[str] = (),
) -> None:
    """Declare the ordered motion set that decodes exported ``traj_id`` values."""
    parser.add_argument(
        *_names("motion_group", *aliases),
        required=required,
        choices=sorted(choices),
        help="Ordered motion set whose entries define the exported traj_id values.",
    )


def add_plots_dir(parser: argparse.ArgumentParser, *, default: str) -> None:
    """Declare the output root for this adapter's tables and figures."""
    parser.add_argument(
        *_names("plots_dir"),
        default=default,
        help=f"Directory for tables and figures (default: {default}).",
    )


def add_video_output(parser: argparse.ArgumentParser) -> None:
    """Declare the output options every render adapter shares."""
    parser.add_argument(*_names("output"), required=True, help="Path of the rendered video file.")
    parser.add_argument(*_names("fps"), type=int, default=30, help="Frames per second (default: 30).")
