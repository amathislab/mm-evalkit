"""Streaming FFmpeg encoder for RGB frames."""

from __future__ import annotations

import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np


class FFmpegEncoder:
    """Encode fixed-size RGB arrays without accumulating video in memory."""

    def __init__(self, output: str | Path, *, width: int, height: int, fps: int) -> None:
        self.output = Path(output)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.width = width
        self.height = height
        self.fps = fps
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self.output),
        ]
        self._process = subprocess.Popen(command, stdin=subprocess.PIPE)
        if self._process.stdin is None:
            raise RuntimeError("Could not open the FFmpeg input pipe.")

    def write(self, frame: np.ndarray) -> None:
        image = np.asarray(frame, dtype=np.uint8)
        if image.shape != (self.height, self.width, 3):
            raise ValueError(f"Encoder expected {(self.height, self.width, 3)}, got {image.shape}.")
        assert self._process.stdin is not None
        self._process.stdin.write(image.tobytes())

    def close(self) -> None:
        if self._process.stdin is not None and not self._process.stdin.closed:
            self._process.stdin.close()
        return_code = self._process.wait()
        if return_code:
            raise RuntimeError(f"FFmpeg exited with status {return_code}.")
