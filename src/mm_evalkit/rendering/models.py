"""Backend-neutral rendering data models."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class SignalComparison:
    """One human/policy waveform pair shown on a signal axis."""

    label: str
    human: FloatArray
    policy: FloatArray
    color: str
    human_label: str = "Human signal"
    policy_label: str = "Policy activation"


@dataclass(frozen=True)
class TendonSignal:
    """Map one rendered tendon to a policy activation column."""

    tendon_name: str
    activation_index: int
    color: tuple[float, float, float]


@dataclass(frozen=True)
class CameraConfig:
    """Free-camera framing shared by all frames of one clip."""

    body_names: tuple[str, ...]
    distance: float
    azimuth: float
    elevation: float


@dataclass(frozen=True)
class RenderClip:
    """A synchronized policy motion and human/policy signal comparison.

    ``activations`` retains its historical name and carries the selected muscle
    signal [frames, actuators] used for tendon styling. Gait120 records that
    signal's identity in ``metadata["muscle_signal"]``.
    """

    qpos: FloatArray
    activations: FloatArray
    sample_hz: float
    x: FloatArray
    x_label: str
    comparisons: tuple[SignalComparison, ...]
    model_name: str
    camera: CameraConfig
    title: str
    subtitle: str
    joint_names: tuple[str, ...] | None = None
    activation_names: tuple[str, ...] | None = None
    tendon_signals: tuple[TendonSignal, ...] = ()
    boundaries: tuple[float, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        qpos = np.asarray(self.qpos, dtype=float)
        activation = np.asarray(self.activations, dtype=float)
        x = np.asarray(self.x, dtype=float)
        if qpos.ndim != 2 or len(qpos) < 2:
            raise ValueError(f"qpos must have shape (at least 2 frames, nq), got {qpos.shape}.")
        if activation.ndim != 2 or len(activation) != len(qpos):
            raise ValueError("activations must be two-dimensional and aligned one-to-one with qpos.")
        if x.shape != (len(qpos),) or not np.isfinite(x).all() or np.any(np.diff(x) <= 0):
            raise ValueError("x must be a finite, strictly increasing coordinate for every qpos frame.")
        if not np.isfinite(self.sample_hz) or self.sample_hz <= 0:
            raise ValueError("sample_hz must be finite and positive.")
        if not self.comparisons:
            raise ValueError("At least one signal comparison is required.")
        for comparison in self.comparisons:
            if np.asarray(comparison.human).shape != (len(qpos),):
                raise ValueError(f"Human trace {comparison.label!r} is not aligned with qpos.")
            if np.asarray(comparison.policy).shape != (len(qpos),):
                raise ValueError(f"Policy trace {comparison.label!r} is not aligned with qpos.")

    @property
    def duration_seconds(self) -> float:
        return (len(self.qpos) - 1) / self.sample_hz
