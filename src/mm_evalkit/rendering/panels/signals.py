"""Persistent Matplotlib signal panel."""

from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from ..models import RenderClip
from ..timeline import normalize_shape

BG = "#ffffff"
AX_BG = "#ffffff"
TEXT = "#202124"
MUTED = "#5f6368"


class SignalPanel:
    """Animate aligned human and policy traces without rebuilding axes."""

    def __init__(self, clip: RenderClip) -> None:
        self.clip = clip
        count = len(clip.comparisons)
        self.figure, raw_axes = plt.subplots(count, 1, figsize=(9.6, 10.8), dpi=100, sharex=True)
        self.axes = np.atleast_1d(raw_axes)
        self.figure.patch.set_facecolor(BG)
        self.figure.subplots_adjust(left=0.105, right=0.97, top=0.86, bottom=0.12, hspace=0.34)
        self.figure.text(0.5, 0.955, clip.title, color=TEXT, fontsize=20, fontweight="bold", ha="center", va="top")
        self.figure.text(0.5, 0.905, clip.subtitle, color=MUTED, fontsize=11, ha="center", va="top")
        self.figure.text(
            0.5,
            0.035,
            "Traces are independently min-max normalized: compare timing/shape, not amplitude.",
            color=MUTED,
            fontsize=9,
            ha="center",
        )
        self.canvas = FigureCanvasAgg(self.figure)
        self._artists: list[dict[str, object]] = []
        self._human: list[np.ndarray] = []
        self._policy: list[np.ndarray] = []

        for index, (axis, comparison) in enumerate(zip(self.axes, clip.comparisons, strict=True)):
            human = normalize_shape(comparison.human)
            policy = normalize_shape(comparison.policy)
            self._human.append(human)
            self._policy.append(policy)
            finite = np.isfinite(comparison.human) & np.isfinite(comparison.policy)
            correlation = (
                float(np.corrcoef(comparison.human[finite], comparison.policy[finite])[0, 1])
                if finite.sum() >= 2
                and np.std(comparison.human[finite]) > 1e-12
                and np.std(comparison.policy[finite]) > 1e-12
                else np.nan
            )
            correlation_label = f"{correlation:+.3f}" if np.isfinite(correlation) else "undefined"
            axis.set_facecolor(AX_BG)
            axis.set_xlim(clip.x[0], clip.x[-1])
            axis.set_ylim(-0.08, 1.08)
            axis.grid(color="#d9dce1", linewidth=0.7, alpha=0.8)
            axis.tick_params(colors=MUTED, labelsize=9)
            for spine in axis.spines.values():
                spine.set_color("#c5c9d0")
            axis.set_ylabel("normalized", color=MUTED, fontsize=9)
            axis.set_title(
                f"{comparison.label}     Pearson r = {correlation_label}",
                color=TEXT,
                fontsize=11,
                loc="left",
                pad=8,
            )
            for boundary in clip.boundaries:
                axis.axvline(boundary, color="#9aa0a6", linestyle=":", linewidth=0.8, alpha=0.75)
            axis.plot(clip.x, human, color=TEXT, linestyle="--", linewidth=1.0, alpha=0.14)
            axis.plot(clip.x, policy, color=comparison.color, linewidth=1.0, alpha=0.14)
            (human_line,) = axis.plot(
                [], [], color=TEXT, linestyle="--", linewidth=2.1, label=comparison.human_label
            )
            (policy_line,) = axis.plot([], [], color=comparison.color, linewidth=2.4, label=comparison.policy_label)
            cursor = axis.axvline(clip.x[0], color="#d97706", linewidth=1.5, alpha=0.95)
            (human_marker,) = axis.plot([], [], "o", color=TEXT, markersize=5)
            (policy_marker,) = axis.plot([], [], "o", color=comparison.color, markersize=6)
            self._artists.append(
                {
                    "human_line": human_line,
                    "policy_line": policy_line,
                    "cursor": cursor,
                    "human_marker": human_marker,
                    "policy_marker": policy_marker,
                }
            )
            if index == 0:
                axis.legend(
                    loc="upper right",
                    frameon=True,
                    facecolor=AX_BG,
                    edgecolor="#c5c9d0",
                    labelcolor=TEXT,
                    fontsize=9,
                )
        self.axes[-1].set_xlabel(clip.x_label, color=MUTED, fontsize=10)

    def render(self, sample_position: float) -> np.ndarray:
        low = min(int(np.floor(sample_position)), len(self.clip.x) - 1)
        high = min(low + 1, len(self.clip.x) - 1)
        blend = sample_position - low
        x_now = float((1.0 - blend) * self.clip.x[low] + blend * self.clip.x[high])
        stop = min(low + 2, len(self.clip.x))
        for index, artists in enumerate(self._artists):
            human = self._human[index]
            policy = self._policy[index]
            artists["human_line"].set_data(self.clip.x[:stop], human[:stop])
            artists["policy_line"].set_data(self.clip.x[:stop], policy[:stop])
            artists["cursor"].set_xdata([x_now, x_now])
            human_now = (1.0 - blend) * human[low] + blend * human[high]
            policy_now = (1.0 - blend) * policy[low] + blend * policy[high]
            artists["human_marker"].set_data([x_now], [human_now])
            artists["policy_marker"].set_data([x_now], [policy_now])
        self.canvas.draw()
        return np.asarray(self.canvas.buffer_rgba(), dtype=np.uint8)[:, :, :3].copy()

    def close(self) -> None:
        plt.close(self.figure)
