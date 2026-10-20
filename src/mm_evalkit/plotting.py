"""Presentation conventions shared by every workflow's figures.

Policies are identified in figures by their configured color, and two policies may
deliberately share one (e.g. finetuned vs from-scratch of the same family). What tells
them apart is the linestyle, so every figure that fills a shape for a policy -- bars,
table label cells, legend keys -- maps that linestyle to a fill alpha through
:func:`policy_fill_alpha`. Keeping the mapping here means a color means the same thing
in the gait and bimanual reports.

The text sizes live here for the same reason: the two reports draw the same figures at
the same sizes, so a panel title should not be 12pt in one and 10pt in the other.
"""

from __future__ import annotations

from matplotlib.colors import to_rgba

# Text sizes, in points, for the small-multiple figures -- the waveform panels, grids and
# summary bars that make up most of both reports. They are read at print size and in
# slides, so the whole vocabulary sits above matplotlib's defaults; naming them rather
# than writing a literal at each call site is what keeps the hierarchy (tick < label <
# panel title < figure title) the same everywhere.
SMALL_TEXT_SIZE = 9  # footnotes, matrix cell annotations, legends drawn inside a panel
TICK_SIZE = 10
LEGEND_SIZE = 11  # figure-level shared legends, which have a full figure width to use
AXIS_LABEL_SIZE = 11
PANEL_TITLE_SIZE = 12
FIGURE_TITLE_SIZE = 15
TABLE_CELL_SIZE = 11  # a metrics table's body cells

#: Text-size multiplier for the full-page figures. The aggregate metric bars give each
#: panel ~6 x 9 inches, and at that size the small-multiple sizes above read as tiny.
FULL_PAGE_SCALE = 1.3


def full_page(size: float) -> int:
    """One of the sizes above, scaled for a figure that fills a page.

    Deriving the large sizes from the small ones (rather than listing a second set of
    literals) keeps the two tiers in the same proportions, so a full-page figure reads
    as the same design at a different scale.
    """
    return round(size * FULL_PAGE_SCALE)


#: Fill alpha of a solid-linestyle policy and of any other (dotted, dashed) one.
SOLID_FILL_ALPHA = 0.8
DASHED_FILL_ALPHA = 0.2


def policy_fill_alpha(linestyle: str) -> float:
    """Fill alpha that distinguishes same-colored policies (finetuned vs from-scratch).

    Policies can share a plot color and differ only by linestyle (":" vs "-"); the bar
    plots encode that as a fill alpha, and the metrics tables and markers reuse the
    same mapping so a color is disambiguated consistently everywhere.
    """
    return SOLID_FILL_ALPHA if linestyle == "-" else DASHED_FILL_ALPHA


def policy_face_color(policy):
    """The policy's plot color at the fill alpha its linestyle maps to."""
    return to_rgba(policy.color, policy_fill_alpha(policy.linestyle))


def policy_bar_style(policy) -> dict:
    """Keyword arguments that draw one bar in the policy's color and texture.

    The alpha lives in the facecolor rather than in a patch-wide alpha so the edge
    keeps full opacity and its linestyle stays legible.
    """
    return {
        "color": policy_face_color(policy),
        "edgecolor": policy.color,
        "linestyle": policy.linestyle,
        "linewidth": 1.5,
    }
