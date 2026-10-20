"""Amplitude scaling is optional for policy and human muscle traces."""

import numpy as np
import pytest

from mm_evalkit.workflows.gait.plotting.muscles import _muscle_plot_trace


@pytest.mark.parametrize("values", [[0.2, 0.3, 0.6], [12.0, 18.0, 24.0], [0.4, 0.4, 0.4]])
def test_muscle_plot_preserves_loaded_amplitudes_without_mutation(values):
    trace = np.array(values)
    original = trace.copy()
    np.testing.assert_array_equal(_muscle_plot_trace(trace, normalize=False), original)
    expected = original if np.ptp(original) == 0 else (original - original.min()) / np.ptp(original)
    np.testing.assert_allclose(_muscle_plot_trace(trace, normalize=True), expected)
    np.testing.assert_array_equal(trace, original)
