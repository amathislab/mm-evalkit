"""High-level rendering API and renderer-adapter discovery."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from typing import Any

from .models import RenderClip

_BUILTINS = {
    "gait120": "mm_evalkit.workflows.gait.render:prepare_clip",
    "ultra-mocap": "mm_evalkit.workflows.bimanual.render:prepare_clip",
    "ultramocap": "mm_evalkit.workflows.bimanual.render:prepare_clip",
}


def _load_target(value: str) -> Callable[..., RenderClip]:
    module_name, separator, attribute = value.partition(":")
    if not separator:
        raise ValueError(f"Render adapter target must use 'module:callable', got {value!r}.")
    implementation = getattr(importlib.import_module(module_name), attribute)
    if not callable(implementation):
        raise TypeError(f"Render adapter {value!r} is not callable.")
    return implementation


def _plugin_target(dataset: str) -> Callable[..., RenderClip] | None:
    available = metadata.entry_points()
    selected = (
        available.select(group="mm_evalkit.renderers")
        if hasattr(available, "select")
        else available.get("mm_evalkit.renderers", ())
    )
    matches = [entry_point for entry_point in selected if entry_point.name.casefold() == dataset]
    if len(matches) > 1:
        raise RuntimeError(f"Multiple mm_evalkit.renderers entry points claim {dataset!r}.")
    return matches[0].load() if matches else None


def prepare_video(*, dataset: str, **options: Any) -> RenderClip:
    """Prepare a validated clip while leaving frame rendering to the caller."""
    normalized = dataset.strip().casefold()
    target = _BUILTINS.get(normalized)
    implementation = _load_target(target) if target is not None else _plugin_target(normalized)
    if implementation is None:
        supported = ", ".join(sorted({"gait120", "ultra-mocap"}))
        raise KeyError(f"No video renderer is registered for {dataset!r}; built-ins: {supported}.")
    clip = implementation(**options)
    if not isinstance(clip, RenderClip):
        raise TypeError(f"Renderer adapter {dataset!r} must return RenderClip, got {type(clip).__name__}.")
    return clip


def render_video(
    *,
    dataset: str,
    output: str | Path,
    fps: int = 30,
    **options: Any,
) -> Path:
    """Prepare and encode one synchronized policy/signal video."""
    from .renderer import VideoRenderer

    return VideoRenderer(fps=fps).render(prepare_video(dataset=dataset, **options), output)
