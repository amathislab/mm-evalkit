"""Strict ULTra-MoCap preprocessed-EMG discovery and time alignment."""

from __future__ import annotations

from collections.abc import Collection
from pathlib import Path

import numpy as np
import pandas as pd


def validate_emg_roots(roots: Collection[str | Path]) -> tuple[Path, ...]:
    """Require explicitly supplied directories before processing any rollout."""
    paths = tuple(Path(root) for root in roots)
    if not paths:
        raise ValueError("At least one preprocessed ULTra-MoCap EMG root is required.")
    missing = [path for path in paths if not path.is_dir()]
    if missing:
        listed = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Preprocessed ULTra-MoCap EMG root(s) do not exist:\n{listed}")
    return paths


def find_preprocessed_emg(
    subject: str,
    trial_stem: str,
    roots: Collection[str | Path],
) -> Path:
    """Require the unique preprocessed sEMG CSV for one trial."""
    target = f"{trial_stem}_preprocessed".casefold()
    matches: list[Path] = []
    for root in roots:
        pattern = Path(root) / subject
        matches.extend(
            path for path in pattern.glob("*/EMG_preprocessed/*_preprocessed.csv") if path.stem.casefold() == target
        )
    unique = sorted(set(matches))
    if len(unique) > 1:
        listed = "\n".join(f"  - {path}" for path in unique)
        raise ValueError(f"Multiple preprocessed EMG files match {trial_stem!r}:\n{listed}")
    if not unique:
        roots_text = "\n".join(f"  - {Path(root)}" for root in roots)
        raise FileNotFoundError(f"No preprocessed EMG file matches {trial_stem!r} under:\n{roots_text}")
    return unique[0]


def load_aligned_emg(
    emg_path: str | Path,
    n_frames: int,
    fs: float,
    *,
    required_channels: Collection[str] | None = None,
) -> pd.DataFrame:
    """Place explicit EMG timestamps on the ``fs`` [Hz] rollout frame-zero grid.

    Samples outside the recording interval remain NaN.  The function never
    shifts rows to frame zero and never extrapolates uncovered edges.
    """
    path = Path(emg_path)
    if not path.is_file():
        raise FileNotFoundError(f"Preprocessed EMG file does not exist: {path}")
    if n_frames <= 0:
        raise ValueError(f"n_frames must be positive, got {n_frames}.")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"fs must be finite and positive, got {fs!r}.")

    frame = pd.read_csv(path)
    if "time" not in frame:
        raise ValueError(
            f"{path} has no time column. Regenerate it with the timestamp-aware EMG preprocessing pipeline."
        )

    channels = tuple(required_channels or ())
    missing_channels = [channel for channel in channels if channel not in frame]
    if missing_channels:
        listed = ", ".join(missing_channels)
        raise ValueError(f"{path} is missing required EMG channel(s): {listed}.")

    times = pd.to_numeric(frame.pop("time"), errors="raise").to_numpy(dtype=float)
    if not len(times) or not np.isfinite(times).all() or np.any(times < 0):
        raise ValueError(f"{path} must contain finite, non-negative time samples.")
    if np.any(np.diff(times) <= 0):
        raise ValueError(f"{path} timestamps must be strictly increasing.")

    sample_index = np.rint(times * fs).astype(int)
    if not np.allclose(times, sample_index / fs, atol=1e-8, rtol=0):
        raise ValueError(f"{path} is not on the expected {fs:g} Hz grid.")
    if len(np.unique(sample_index)) != len(sample_index):
        raise ValueError(f"{path} contains duplicate timestamps.")

    for channel in channels:
        values = pd.to_numeric(frame[channel], errors="raise").to_numpy(dtype=float)
        invalid = ~np.isfinite(values) & ~np.isnan(values)
        if invalid.any():
            raise ValueError(f"{path} contains non-finite values for EMG channel {channel!r}.")
        if not np.isfinite(values).any():
            raise ValueError(f"{path} has no finite values for EMG channel {channel!r}.")
        frame[channel] = values

    frame.index = sample_index
    return frame.reindex(range(n_frames))
