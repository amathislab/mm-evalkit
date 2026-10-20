"""Shared plotting primitives for full-body reports."""

from __future__ import annotations

import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory

from mm_evalkit.plotting import policy_face_color, policy_fill_alpha

from ..constants import (
    REF_GREY,
)

# Re-exported so the gait figures keep reaching their style vocabulary through this
# module; the mapping itself is shared with the other workflows.
__all__ = ["policy_fill_alpha"]


def policy_color_rgba(analysis, policy_idx):
    """The policy's plot color at the same fill alpha the bar plots use for it."""
    return policy_face_color(analysis.policies[policy_idx].policy)


def policy_marker_box(analysis, policy_idx, text="Policy"):
    """Build a "<text> [square]" box whose square is filled with the policy's plot color.

    The policy labels are long sentences, so the RSA figures identify a policy by its
    plot color instead of by name; the square is the same color (and fill alpha) used in
    the other plots, so same-colored policies stay distinguishable.
    """
    square = DrawingArea(9, 9, 0, 0)
    square.add_artist(
        Rectangle(
            (0, 0),
            9,
            9,
            facecolor=policy_color_rgba(analysis, policy_idx),
            edgecolor="black",
            linewidth=0.5,
        )
    )
    return HPacker(children=[TextArea(text), square], align="center", pad=0, sep=4)


def add_policy_title(analysis, ax, policy_idx, suffix=""):
    """Place a "Policy [square]" marker where the axes title would go."""
    ax.add_artist(
        AnchoredOffsetbox(
            loc="lower center",
            child=policy_marker_box(analysis, policy_idx, f"Policy{suffix}"),
            bbox_to_anchor=(0.5, 1.0),
            bbox_transform=ax.transAxes,
            pad=0,
            borderpad=0.4,
            frameon=False,
        )
    )


def add_policy_ytick_markers(analysis, ax, n_subjects):
    """Replace the policy rows' y tick labels with "Policy [square]" markers.

    The subject rows keep their text labels; only the policy rows below them are
    swapped, so the caller must have already set tick labels for every row.
    """
    # x in axes coords (pin to the left spine), y in data coords (the matrix row).
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    for policy_idx in range(len(analysis.policies)):
        ax.add_artist(
            AnchoredOffsetbox(
                loc="center right",
                child=policy_marker_box(analysis, policy_idx),
                bbox_to_anchor=(0, n_subjects + policy_idx),
                bbox_transform=trans,
                pad=0,
                borderpad=0.4,
                frameon=False,
            )
        )


def draw_rsm(analysis, ax, rsm, muscle_names, display_muscle_names, title, annotate=True):
    """Draw one lower-triangular muscle representational-similarity matrix."""

    def _mask_upper_triangle(ax, matrix_size, color="white"):
        """Mask redundant upper-triangle entries while retaining the diagonal."""
        for i in range(matrix_size):
            for j in range(i + 1, matrix_size):
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, color=color, zorder=2))

    matrix_size = rsm.shape[0]
    im = ax.imshow(rsm, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    _mask_upper_triangle(ax, matrix_size)
    ax.set_xticks(range(matrix_size))
    ax.set_yticks(range(matrix_size))
    ax.set_xticklabels(display_muscle_names, rotation=90, fontsize=7)
    ax.set_yticklabels(display_muscle_names, fontsize=7)
    ax.set_title(title, fontsize=9)
    if annotate:
        for i in range(matrix_size):
            for j in range(i):
                val = rsm[i, j]
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black" if abs(val) < 0.6 else "white",
                    zorder=3,
                )
    return im


def draw_grouped_hbar(analysis, ax, hh_samples, hh_label, policy_samples, show_group_labels, ref_hatch="//"):
    """Draw one horizontal grouped bar plot (reference, then one bar per policy) for a single metric.

    Groups run top-to-bottom so they read in the same order as the labels. Only the
    aggregate panel shows the group names on its y axis; it stands in for the legend,
    so the per-item panels leave their y ticks unlabelled. The reference (top) bar is
    drawn grey and hatched with ``ref_hatch`` so the different human references
    (within-subject vs between-subject) are told apart by texture, not color.
    """
    labels = [hh_label] + [run.policy.label for run in analysis.policies]
    colors = [REF_GREY] + [run.policy.color for run in analysis.policies]
    # The ground-truth reference bar is always solid; policy bars take their linestyle
    # from the config (":" or "-"), which also sets the edge style and fill alpha below.
    linestyles = ["-"] + [run.policy.linestyle for run in analysis.policies]
    samples = [hh_samples, *policy_samples]

    bar_height = 0.7
    y = np.arange(len(labels))[::-1]
    # Jitter separates overlapping observations but carries no information. Keep
    # it local and deterministic so repeated analyses produce the same figure and
    # do not depend on unrelated consumers of NumPy's process-wide RNG.
    jitter_rng = np.random.default_rng(0)

    for group_idx, (y_pos, color, linestyle, group_samples) in enumerate(
        zip(y, colors, linestyles, samples, strict=True)
    ):
        group_samples = np.asarray(group_samples, dtype=float)
        group_samples = group_samples[np.isfinite(group_samples)]

        # A policy with no retained gait cycles contributes no samples; leave its slot
        # empty rather than drawing a NaN bar.
        if group_samples.size == 0:
            continue

        # Dotted bars are drawn faint so they read as secondary; solid bars stay strong.
        # The alpha lives in the facecolor (not the patch-wide alpha) so the edge line
        # keeps its full opacity and its linestyle stays legible.
        fill_alpha = policy_fill_alpha(linestyle)

        ax.barh(
            y_pos,
            np.mean(group_samples),
            height=bar_height,
            color=to_rgba(color, fill_alpha),
            # Only the ground-truth reference bar carries a std error bar; the policy
            # bars are described by their scatter alone.
            xerr=np.std(group_samples) if group_idx == 0 else None,
            capsize=5 if group_idx == 0 else 0,
            # Policy bars edge in their own color; the ground-truth bar uses a darker grey
            # so its hatch texture stays visible against the light-grey fill.
            edgecolor="grey" if group_idx == 0 else color,
            linestyle=linestyle,
            linewidth=1.5,
            # The ground-truth reference bar is hatched so it reads as distinct from the policies.
            hatch=ref_hatch if group_idx == 0 else None,
            zorder=2,
        )

        jitter = jitter_rng.normal(0, bar_height * 0.08, len(group_samples))

        ax.scatter(
            group_samples,
            np.full(len(group_samples), y_pos) + jitter,
            color=color,
            alpha=0.7,
            s=30,
            zorder=3,
            edgecolors="black",
            linewidths=0.5,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels if show_group_labels else [])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.7, zorder=1)
