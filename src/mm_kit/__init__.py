"""Extensible neuromuscular evaluation and synchronized video rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .registry import DatasetAdapter, get_dataset, registered_datasets, run_dataset


def render_video(*, dataset: str, output: str | Path, **options: Any) -> Path:
    """Prepare and render one dataset-specific synchronized video.

    Rendering dependencies are imported lazily, so gait/statistical evaluation
    does not require MuJoCo or FFmpeg.
    """
    from .rendering.api import render_video as _render_video

    return _render_video(dataset=dataset, output=output, **options)


__all__ = [
    "DatasetAdapter",
    "get_dataset",
    "registered_datasets",
    "render_video",
    "run_dataset",
]
