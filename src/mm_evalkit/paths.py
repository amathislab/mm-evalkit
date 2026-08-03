"""Resolve dataset paths without importing MuscleMimic or LocoMuJoCo."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import yaml

CONFIG_PATH_ENV = "MUSCLEMIMIC_CONFIG_PATH"
DEFAULT_CONFIG_PATH = Path("~/.musclemimic/MUSCLEMIMIC_VARIABLES.yaml")

DATASET_PATH_KEYS = {
    "gait120": "MUSCLEMIMIC_GAIT120_DATA_PATH",
    "wang": "MUSCLEMIMIC_WANG_DATA_PATH",
    "ultra-mocap": "MUSCLEMIMIC_ULTRA_MOCAP_DATA_PATH",
    "ultramocap": "MUSCLEMIMIC_ULTRA_MOCAP_DATA_PATH",
}


def config_path() -> Path:
    """Return the configured YAML path, even when it does not exist yet."""
    return Path(os.environ.get(CONFIG_PATH_ENV, DEFAULT_CONFIG_PATH)).expanduser()


def load_path_config(path: str | Path | None = None) -> dict[str, str]:
    """Load the shared path YAML as a flat string mapping."""
    selected = Path(path).expanduser() if path is not None else config_path()
    if not selected.exists():
        return {}
    if not selected.is_file():
        raise ValueError(f"Path configuration is not a file: {selected}")
    document = yaml.safe_load(selected.read_text(encoding="utf-8")) or {}
    if not isinstance(document, dict):
        raise ValueError(f"Path configuration must contain a YAML mapping: {selected}")
    invalid = [key for key, value in document.items() if not isinstance(key, str) or not isinstance(value, str)]
    if invalid:
        raise ValueError(f"Path configuration values must be strings: {selected}")
    return document


def resolve_dataset_path(
    dataset: str,
    explicit: str | Path | None = None,
    *,
    must_exist: bool = True,
) -> Path:
    """Resolve a dataset directory by explicit value, environment, then YAML.

    The exact resolved absolute path is returned so callers can store it in
    their run manifest. No dataset-specific directory guessing is performed.
    """
    normalized = dataset.strip().casefold()
    try:
        key = DATASET_PATH_KEYS[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(set(DATASET_PATH_KEYS)))
        raise KeyError(f"Unknown dataset {dataset!r}; choose one of: {supported}") from exc

    configured = explicit
    source = "explicit argument"
    if configured is None:
        configured = os.environ.get(key)
        source = f"environment variable {key}"
    if configured is None:
        configured = load_path_config().get(key)
        source = f"{config_path()} key {key}"
    if configured is None:
        raise FileNotFoundError(
            f"No data path is configured for {dataset}. Pass the dataset path, "
            f"set {key}, or add {key} to {config_path()}."
        )

    path = Path(configured).expanduser().resolve()
    if must_exist and not path.is_dir():
        raise FileNotFoundError(f"{dataset} data directory from {source} does not exist: {path}")
    return path


def set_dataset_path(dataset: str, value: str | Path) -> Path:
    """Persist one validated absolute dataset directory in the shared YAML."""
    normalized = dataset.strip().casefold()
    try:
        key = DATASET_PATH_KEYS[normalized]
    except KeyError as exc:
        raise KeyError(f"Unknown dataset {dataset!r}.") from exc
    resolved = Path(value).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {resolved}")

    destination = config_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = load_path_config(destination)
    document[key] = str(resolved)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        yaml.safe_dump(document, handle, sort_keys=True)
        temporary = Path(handle.name)
    os.replace(temporary, destination)
    return resolved
