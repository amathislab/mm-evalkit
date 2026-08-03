"""Composable synchronized-video rendering."""

from .api import prepare_video, render_video
from .layouts import compose_side_by_side
from .models import CameraConfig, RenderClip, SignalComparison, TendonSignal
from .renderer import VideoRenderer

__all__ = [
    "CameraConfig",
    "RenderClip",
    "SignalComparison",
    "TendonSignal",
    "VideoRenderer",
    "compose_side_by_side",
    "prepare_video",
    "render_video",
]
