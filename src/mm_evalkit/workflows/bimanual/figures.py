"""Figure output conventions for the ULTRA-MoCap report.

Every figure is written in both formats the full-body report uses: a PNG to look at
and an SVG to put in a paper. Callers name a figure once, without an extension, and
this module owns the pair.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from mm_evalkit.plotting import (
    AXIS_LABEL_SIZE,
    FIGURE_TITLE_SIZE,
    LEGEND_SIZE,
    PANEL_TITLE_SIZE,
    SMALL_TEXT_SIZE,
    TABLE_CELL_SIZE,
    TICK_SIZE,
    full_page,
)

FIGURE_DPI = 160

# Re-exported so the ULTRA-MoCap figures reach their whole presentation vocabulary --
# text sizes and file formats -- through this one module; the sizes themselves are
# shared with the gait report.
__all__ = [
    "AXIS_LABEL_SIZE",
    "FIGURE_TITLE_SIZE",
    "LEGEND_SIZE",
    "PANEL_TITLE_SIZE",
    "SMALL_TEXT_SIZE",
    "TABLE_CELL_SIZE",
    "TICK_SIZE",
    "full_page",
    "save_figure",
]


def save_figure(fig, out_path, *, dpi: int = FIGURE_DPI, close: bool = True, announce: str = "  saved") -> Path:
    """Save one figure as ``<base>.png`` and ``<base>.svg`` and report the PNG.

    ``out_path`` may carry a ``.png``/``.svg`` extension (the callers that grew up
    naming a PNG keep working) or none at all; either way both files are written next
    to each other. Returns the PNG path.
    """
    base = Path(out_path)
    if base.suffix.lower() in {".png", ".svg"}:
        base = base.with_suffix("")
    png = base.with_suffix(".png")
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    if close:
        plt.close(fig)
    if announce:
        print(f"{announce} {png}")
    return png
