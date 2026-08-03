"""Dataset-neutral synchronized video frame loop."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from .encoder import FFmpegEncoder
from .layouts import compose_side_by_side
from .models import RenderClip
from .panels import SignalPanel
from .scenes import MujocoScene
from .timeline import interpolate_row


class VideoRenderer:
    """Render any validated :class:`RenderClip` with the standard components."""

    def __init__(
        self,
        *,
        fps: int = 30,
        scene_factory: Callable[[RenderClip], MujocoScene] = MujocoScene,
        panel_factory: Callable[[RenderClip], SignalPanel] = SignalPanel,
        layout: Callable[..., np.ndarray] = compose_side_by_side,
    ) -> None:
        if fps < 1:
            raise ValueError("fps must be at least one.")
        self.fps = fps
        self.scene_factory = scene_factory
        self.panel_factory = panel_factory
        self.layout = layout

    def render(self, clip: RenderClip, output: str | Path) -> Path:
        output_path = Path(output).expanduser().resolve()
        scene = self.scene_factory(clip)
        panel = self.panel_factory(clip)
        encoder = FFmpegEncoder(output_path, width=1920, height=1080, fps=self.fps)
        n_frames = round(clip.duration_seconds * self.fps) + 1
        try:
            for frame_number in range(n_frames):
                elapsed = min(frame_number / self.fps, clip.duration_seconds)
                sample_position = min(elapsed * clip.sample_hz, len(clip.qpos) - 1)
                motion = scene.render(interpolate_row(clip.qpos, sample_position), sample_position)
                signals = panel.render(sample_position)
                frame = self.layout(
                    clip,
                    motion,
                    signals,
                    sample_position=sample_position,
                )
                encoder.write(frame)
                if frame_number % max(self.fps * 2, 1) == 0 or frame_number == n_frames - 1:
                    print(f"Rendered {frame_number + 1}/{n_frames} frames")
        finally:
            scene.close()
            panel.close()
            encoder.close()
        print(f"Saved {output_path}")
        return output_path
