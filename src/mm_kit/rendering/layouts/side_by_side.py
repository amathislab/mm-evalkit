"""Standard 1920x1080 policy-motion and signal layout."""

from __future__ import annotations

import cv2
import numpy as np

from ..models import RenderClip


def _text(
    image: np.ndarray,
    value: str,
    origin: tuple[int, int],
    size: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    cv2.putText(image, value, origin, cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness, cv2.LINE_AA)


def compose_side_by_side(
    clip: RenderClip,
    motion_view: np.ndarray,
    signal_panel: np.ndarray,
    *,
    sample_position: float,
) -> np.ndarray:
    """Compose the canonical full-HD frame from reusable panel outputs."""
    policy_view = cv2.resize(motion_view, (900, 675), interpolation=cv2.INTER_AREA)
    left = np.empty((1080, 960, 3), dtype=np.uint8)
    left[:] = (10, 16, 32)
    left[140:815, 30:930] = policy_view
    _text(left, "Policy motion (MuJoCo)", (36, 55), 1.05, (255, 255, 255), 2)
    _text(left, clip.subtitle, (38, 100), 0.55, (172, 186, 213), 1)

    sample = min(round(sample_position), len(clip.x) - 1)
    coordinate = clip.x[sample]
    if "%" in clip.x_label:
        progress = f"gait phase {coordinate:.1f}%"
    else:
        progress = f"clip {coordinate:05.2f} / {clip.x[-1]:05.2f} s"
    _text(left, progress, (38, 872), 0.68, (255, 255, 255), 1)

    y = 916
    spacing = min(42, max(30, 132 // len(clip.comparisons)))
    for comparison in clip.comparisons:
        rgb = tuple(int(value * 255) for value in _hex_rgb(comparison.color))
        cv2.line(left, (42, y - 6), (95, y - 6), rgb, 7, cv2.LINE_AA)
        _text(left, comparison.label, (112, y), 0.50, rgb, 1)
        y += spacing
        if y > 1045:
            break
    return np.hstack((left, signal_panel))


def _hex_rgb(color: str) -> tuple[float, float, float]:
    value = color.lstrip("#")
    if len(value) != 6:
        return (1.0, 1.0, 1.0)
    return tuple(int(value[index : index + 2], 16) / 255.0 for index in (0, 2, 4))
