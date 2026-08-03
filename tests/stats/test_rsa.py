"""Representational similarity: the RSM itself, and how two are compared."""

import numpy as np
import pytest

from mm_kit.stats.rsa import compute_rsm, rsa


def test_compute_rsm_is_the_channel_correlation_matrix():
    time = np.linspace(0, 2 * np.pi, 100)
    patterns = np.stack([np.sin(time), -np.sin(time), np.cos(time)])

    rsm = compute_rsm(patterns)

    assert rsm.shape == (3, 3)
    assert np.diag(rsm) == pytest.approx(1.0)
    assert rsm[0, 1] == pytest.approx(-1.0)  # exact anti-phase
    assert rsm[0, 2] == pytest.approx(0.0, abs=0.02)  # quadrature


def test_compute_rsm_preserves_undefined_silent_channel_correlations():
    """A silent channel is missing information, not zero correlation."""
    time = np.linspace(0, 2 * np.pi, 100)
    rsm = compute_rsm(np.stack([np.sin(time), np.zeros_like(time)]))

    assert rsm[0, 0] == pytest.approx(1.0)
    assert np.isnan(rsm[0, 1])
    assert np.isnan(rsm[1]).all()


def _example_rsm():
    time = np.linspace(0, 6, 60)
    return compute_rsm(np.stack([np.sin(time), np.cos(time), np.linspace(0, 1, 60)]))


def test_rsa_ranks_the_off_diagonal_structure():
    a = _example_rsm()
    index = np.tril_indices(3, k=-1)

    # A matrix against itself is a perfect match...
    assert rsa(a, a)[0] == pytest.approx(1.0)
    # ...and inverting every off-diagonal relation reverses their ranking exactly.
    inverted = a.copy()
    inverted[index] = -a[index]
    inverted.T[index] = inverted[index]
    assert rsa(a, inverted)[0] == pytest.approx(-1.0)


def test_rsa_ignores_the_diagonal_and_the_upper_triangle():
    """Only the lower off-diagonal triangle is scored -- the rest is trivial or redundant."""
    a, b = _example_rsm(), compute_rsm(np.random.default_rng(9).normal(0.0, 1.0, (3, 60)))
    baseline = rsa(a, b)

    tampered = a.copy()
    np.fill_diagonal(tampered, 0.3)  # the trivial self-similarities
    tampered[np.triu_indices(3, k=1)] = -0.9  # the redundant mirror half

    assert rsa(tampered, b) == baseline


def test_rsa_refuses_a_degenerate_matrix():
    """An all-zero RSM has no structure to correlate, so it must not return a number."""
    spearman, pearson = rsa(np.eye(3), np.eye(3))
    assert np.isnan(spearman) and np.isnan(pearson)


def test_rsa_uses_finite_pairs_from_a_partial_triangle():
    """Unmeasured relations are omitted without inventing correlation values."""
    first = np.eye(3)
    second = np.eye(3)
    first[1, 0], first[2, 0], first[2, 1] = 0.1, np.nan, 0.3
    second[1, 0], second[2, 0], second[2, 1] = 0.2, 0.4, 0.6

    spearman, pearson = rsa(first, second)

    assert spearman == pytest.approx(1.0)
    assert pearson == pytest.approx(1.0)
