"""Representational similarity of muscle activation patterns.

A correlation per muscle says how well a model reproduces *that muscle's* time course.
It says nothing about the relationships **between** muscles -- which pairs co-activate,
which oppose -- and two models with the same per-muscle correlation can have completely
different coordination. Representational similarity analysis asks that second question:

1. Reduce a set of activation patterns to their **RSM** (:func:`compute_rsm`): the
   muscle x muscle correlation matrix over the cycle. It is a description of the
   coordination pattern alone -- amplitudes and units have been divided out.
2. Compare two RSMs (:func:`rsa`) by correlating their lower off-diagonal triangles.
   The diagonal (self-similarities, always 1.0) and the redundant upper triangle are
   excluded, so the score is driven only by the between-muscle structure.

Spearman is the primary metric -- it is a rank correlation, so it is not distorted by
the distribution of RSM values -- and Pearson is returned alongside as a sensitivity
check.

Both full-body and bimanual analyses use this: the former over the 11 Gait120 muscles (55
pairs) and ``neuromuscular_bimanual`` over the 3 ULTra-MoCap electrodes (**3 pairs**).
Three pairs is a very short vector to correlate -- a Spearman on it can only take a
handful of values -- so the bimanual RSA is a coarse, qualitative statement about
coordination and its significance tests are underpowered by construction. That is a
property of the electrode montage, not of the code; the figures and tables that use it
say so.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import spearmanr


def compute_rsm(patterns):
    """Muscle x muscle correlation matrix of ``(muscles, time_steps)`` activations.

    Constant channels have no defined correlation. Those entries remain
    ``NaN``: replacing them with zero would manufacture evidence that the
    channels are unrelated.
    """
    patterns = np.asarray(patterns, dtype=float)
    if patterns.ndim != 2:
        raise ValueError(f"RSM patterns must have shape (channels, time_steps), got {patterns.shape}.")
    if patterns.shape[0] < 2 or patterns.shape[1] < 2:
        raise ValueError(f"RSM patterns require at least two channels and two time samples, got {patterns.shape}.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.corrcoef(patterns)


def rsa(rsm_a, rsm_b):
    """Representational similarity between two RSMs -> ``(spearman, pearson)``.

    Only finite pairs in the lower off-diagonal triangles are used, excluding
    the diagonal's trivial 1.0s and the redundant upper triangle. ``NaN`` is
    returned when fewer than two pairs remain or either vector has no variance.
    """
    rsm_a = np.asarray(rsm_a, dtype=float)
    rsm_b = np.asarray(rsm_b, dtype=float)
    if rsm_a.ndim != 2 or rsm_a.shape[0] != rsm_a.shape[1]:
        raise ValueError(f"rsm_a must be square, got {rsm_a.shape}.")
    if rsm_b.shape != rsm_a.shape:
        raise ValueError(f"RSMs must have the same square shape, got {rsm_a.shape} and {rsm_b.shape}.")
    index = np.tril_indices(rsm_a.shape[0], k=-1)
    a, b = rsm_a[index], rsm_b[index]
    finite = np.isfinite(a) & np.isfinite(b)
    a, b = a[finite], b[finite]
    if a.size < 2 or np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return np.nan, np.nan
    return float(spearmanr(a, b).statistic), float(np.corrcoef(a, b)[0, 1])
