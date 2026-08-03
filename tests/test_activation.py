"""Activation export-timing regression tests."""

import numpy as np
import pytest

from mm_kit.activation import (
    activation_sample_phase,
    align_activation_to_observation_timeline,
    complete_activation_gait_indexes,
)


def test_pre_step_activation_is_already_on_the_state_timeline():
    values = np.arange(8, dtype=float).reshape(4, 2)
    phase = activation_sample_phase({"activation_sample_phase": np.array("pre_step")})

    np.testing.assert_array_equal(
        align_activation_to_observation_timeline(values, phase),
        values,
    )


def test_historical_post_step_activation_moves_forward_one_state():
    values = np.arange(8, dtype=float).reshape(4, 2)
    aligned = align_activation_to_observation_timeline(
        values,
        activation_sample_phase({}),
    )

    assert np.isnan(aligned[0]).all()
    np.testing.assert_array_equal(aligned[1:], values[:-1])


def test_unknown_activation_sample_phase_is_rejected():
    with pytest.raises(
        ValueError,
        match=r"post_step.*pre_step|pre_step.*post_step",
    ):
        activation_sample_phase({"activation_sample_phase": np.array("mid_step")})


def test_cycle_with_unobserved_activation_is_excluded():
    activation = np.arange(12, dtype=float).reshape(6, 2)
    activation[0] = np.nan

    assert complete_activation_gait_indexes(
        activation,
        [(0, 3), (3, 6)],
    ) == [(3, 6)]
