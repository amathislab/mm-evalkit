"""The panel and layout seams a custom renderer composes against.

A new presentation is built by swapping one of these while keeping the rest, so
their frame contracts -- size, dtype, and independence from the scene backend --
are covered directly.
"""

import numpy as np
import pytest

from mm_evalkit.rendering.models import CameraConfig, RenderClip, SignalComparison

cv2 = pytest.importorskip("cv2", reason="frame composition needs opencv")

from mm_evalkit.rendering.layouts import compose_side_by_side  # noqa: E402
from mm_evalkit.rendering.panels import SignalPanel  # noqa: E402


def _clip(n_comparisons: int = 2) -> RenderClip:
    signal = np.linspace(0.0, 1.0, 10)
    colors = ("#4c9aff", "#ff9f43", "#45d483")
    return RenderClip(
        qpos=np.zeros((10, 2)),
        activations=np.zeros((10, 3)),
        sample_hz=100.0,
        x=np.linspace(0.0, 100.0, 10),
        x_label="gait cycle (%)",
        comparisons=tuple(
            SignalComparison(f"channel {index}", signal, signal[::-1], colors[index % len(colors)])
            for index in range(n_comparisons)
        ),
        model_name="test",
        camera=CameraConfig(("body",), 1.0, 90.0, 0.0),
        title="title",
        subtitle="subtitle",
    )


@pytest.mark.parametrize("n_comparisons", [1, 2, 5])
def test_signal_panel_renders_one_row_per_comparison(n_comparisons):
    panel = SignalPanel(_clip(n_comparisons))

    assert len(panel.axes) == n_comparisons
    frame = panel.render(sample_position=0.0)
    assert frame.dtype == np.uint8
    assert frame.shape == (1080, 960, 3)


def test_signal_panel_advances_with_the_sample_position():
    panel = SignalPanel(_clip())

    first = panel.render(sample_position=0.0)
    last = panel.render(sample_position=9.0)

    assert not np.array_equal(first, last)


def test_side_by_side_produces_a_full_hd_frame_from_any_scene_view():
    """The layout resizes whatever the scene backend rendered; it owns no camera."""
    clip = _clip()
    panel = SignalPanel(clip).render(sample_position=0.0)
    motion_view = np.full((480, 640, 3), 128, dtype=np.uint8)

    frame = compose_side_by_side(clip, motion_view, panel, sample_position=0.0)

    assert frame.shape == (1080, 1920, 3)
    assert frame.dtype == np.uint8
    # The right half is the panel unchanged; only the left half is composed.
    assert np.array_equal(frame[:, 960:], panel)


def test_side_by_side_clamps_a_sample_position_past_the_last_frame():
    clip = _clip()
    panel = SignalPanel(clip).render(sample_position=0.0)
    motion_view = np.zeros((480, 640, 3), dtype=np.uint8)

    frame = compose_side_by_side(clip, motion_view, panel, sample_position=len(clip.x) + 5)

    assert frame.shape == (1080, 1920, 3)
