"""Metadata-only registry for supported movement-evaluation datasets.

Dataset packages are imported only when an adapter runs. Listing datasets and
rendering top-level help therefore do not import scientific or plotting
dependencies.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import metadata


@dataclass(frozen=True)
class DatasetAdapter:
    """Describe one public dataset-package entrypoint."""

    name: str
    workflow: str
    body_model: str
    module: str
    entrypoint: str
    description: str
    aliases: tuple[str, ...] = ()
    runner: Callable[[Sequence[str] | None], int] | None = None

    def run(self, argv: Sequence[str] | None = None) -> int:
        """Load the selected workflow lazily and invoke its dataset entrypoint."""
        if self.runner is not None:
            return int(self.runner(argv))
        implementation = importlib.import_module(self.module)
        run = getattr(implementation, self.entrypoint)
        return int(run(list(argv or ())))


_DATASETS = (
    DatasetAdapter(
        name="gait120",
        workflow="gait",
        body_model="fullbody",
        module="mm_kit.workflows.gait",
        entrypoint="gait120_main",
        description="Full-body gait kinematics, kinetics, and EMG from Gait120.",
    ),
    DatasetAdapter(
        name="wang",
        workflow="gait",
        body_model="fullbody",
        module="mm_kit.workflows.gait",
        entrypoint="wang_main",
        description="Full-body gait kinematics and EMG from Wang.",
    ),
    DatasetAdapter(
        name="ultra-mocap",
        workflow="bimanual",
        body_model="bimanual",
        module="mm_kit.workflows.bimanual",
        entrypoint="main",
        description="Bimanual upper-limb kinematics and sEMG from ULTra-MoCap.",
        aliases=("ultramocap",),
    ),
)


def _normalize_name(name: str) -> str:
    """Dataset names are case-insensitive but otherwise explicit."""
    return name.strip().casefold()


def _plugin_datasets() -> tuple[DatasetAdapter, ...]:
    """Discover third-party adapters without importing them until requested."""
    builtins = {adapter.name for adapter in _DATASETS}
    discovered: list[DatasetAdapter] = []
    available = metadata.entry_points()
    selected = (
        available.select(group="mm_kit.datasets")
        if hasattr(available, "select")
        else available.get("mm_kit.datasets", ())
    )
    for entry_point in selected:
        if entry_point.name in builtins:
            continue

        def run(argv: Sequence[str] | None, *, selected=entry_point) -> int:
            implementation = selected.load()
            if not callable(implementation):
                raise TypeError(f"Dataset entry point {selected.name!r} must resolve to a callable.")
            return int(implementation(list(argv or ())))

        discovered.append(
            DatasetAdapter(
                name=entry_point.name,
                workflow="plugin",
                body_model="external",
                module=entry_point.module,
                entrypoint=entry_point.attr or "",
                description=f"External adapter from {entry_point.value}.",
                runner=run,
            )
        )
    return tuple(sorted(discovered, key=lambda adapter: adapter.name))


def registered_datasets() -> tuple[DatasetAdapter, ...]:
    """Return built-in and installed plugin adapters in display order."""
    return (*_DATASETS, *_plugin_datasets())


def get_dataset(name: str) -> DatasetAdapter:
    """Resolve a canonical dataset name or alias."""
    requested = _normalize_name(name)
    for adapter in registered_datasets():
        names = (adapter.name, *adapter.aliases)
        if requested in {_normalize_name(candidate) for candidate in names}:
            return adapter
    supported = ", ".join(adapter.name for adapter in registered_datasets())
    raise KeyError(f"Unknown neuromuscular dataset {name!r}; choose one of: {supported}")


def run_dataset(name: str, argv: Sequence[str] | None = None) -> int:
    """Run a registered dataset adapter from Python."""
    return get_dataset(name).run(argv)
