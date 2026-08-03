import numpy as np
import pytest

from mm_kit.rendering.models import CameraConfig, RenderClip, SignalComparison
from mm_kit.rendering.timeline import normalize_shape, resample_profile


def _clip() -> RenderClip:
    signal = np.linspace(0.0, 1.0, 10)
    return RenderClip(
        qpos=np.zeros((10, 2)),
        activations=np.zeros((10, 3)),
        sample_hz=100.0,
        x=np.linspace(0.0, 100.0, 10),
        x_label="gait cycle (%)",
        comparisons=(SignalComparison("test", signal, signal, "#ffffff"),),
        model_name="test",
        camera=CameraConfig(("body",), 1.0, 90.0, 0.0),
        title="title",
        subtitle="subtitle",
    )


def test_render_clip_keeps_playback_time_separate_from_display_coordinate():
    clip = _clip()
    assert clip.duration_seconds == pytest.approx(0.09)
    assert clip.x[-1] == 100.0


def test_render_clip_rejects_unaligned_signal():
    clip = _clip()
    with pytest.raises(ValueError, match="not aligned"):
        RenderClip(
            **{
                **clip.__dict__,
                "comparisons": (SignalComparison("bad", np.zeros(9), np.zeros(10), "#ffffff"),),
            }
        )


def test_shape_normalization_and_phase_resampling():
    assert np.allclose(normalize_shape(np.array([2.0, 4.0, 6.0])), [0.0, 0.5, 1.0])
    assert np.allclose(normalize_shape(np.ones(3)), 0.0)
    assert np.allclose(resample_profile(np.array([0.0, 1.0]), 3), [0.0, 0.5, 1.0])
