"""Reproducibility contracts for gait plotting primitives."""

import matplotlib
import numpy as np
from matplotlib.collections import PathCollection

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from mm_evalkit.workflows.gait.plotting.style import draw_grouped_hbar

from .conftest import make_analysis, make_run


def _scatter_offsets(global_seed):
    """Draw after perturbing the process RNG and return only scatter points."""
    np.random.seed(global_seed)
    analysis = make_analysis(policies=[make_run("policy")])
    fig, ax = plt.subplots()
    draw_grouped_hbar(
        analysis,
        ax,
        [1.0, 2.0, 3.0],
        "reference",
        [[4.0, 5.0]],
        show_group_labels=True,
    )
    offsets = [
        collection.get_offsets().copy() for collection in ax.collections if isinstance(collection, PathCollection)
    ]
    plt.close(fig)
    return offsets


def test_grouped_bar_jitter_is_reproducible_and_global_rng_independent():
    first = _scatter_offsets(1)
    second = _scatter_offsets(999)

    assert len(first) == len(second) == 2
    for first_group, second_group in zip(first, second, strict=True):
        np.testing.assert_array_equal(first_group, second_group)
