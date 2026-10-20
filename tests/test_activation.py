"""Activation export-timing regression tests."""

import numpy as np
import pytest

from mm_evalkit.activation import (
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


def test_ctrl_uses_action_row_while_legacy_activation_uses_next_state():
    from mm_evalkit.activation import load_emg_signal

    activation = np.arange(8, dtype=float).reshape(4, 2)
    commands = activation / 10
    archive = {"episode_0_muscle_activations": activation, "episode_0_muscle_commands": commands}
    np.testing.assert_array_equal(load_emg_signal(archive, 0), commands)
    aligned = load_emg_signal(archive, 0, "activations")
    assert np.isnan(aligned[0]).all()
    np.testing.assert_array_equal(aligned[1:], activation[:-1])


def test_ctrl_clips_before_processing_without_mutating_archive():
    from mm_evalkit.activation import load_emg_signal

    commands = np.array([[-2, 0.2], [0, 1], [1.3, 0.8], [0.5, -0.1]])
    original = commands.copy()
    archive = {"episode_0_muscle_activations": np.zeros_like(commands),
               "episode_0_muscle_commands": commands}
    result = load_emg_signal(archive, 0, "ctrl")
    np.testing.assert_array_equal(result, [[0, 0.2], [0, 1], [1, 0.8], [0.5, 0]])
    np.testing.assert_array_equal(commands, original)


@pytest.mark.parametrize("commands", [None, np.zeros((3, 2)), np.full((4, 2), np.nan)])
def test_ctrl_rejects_missing_or_invalid_commands(commands):
    from mm_evalkit.activation import load_emg_signal

    archive = {"episode_0_muscle_activations": np.zeros((4, 2))}
    if commands is not None:
        archive["episode_0_muscle_commands"] = commands
    with pytest.raises(ValueError, match="muscle_commands"):
        load_emg_signal(archive, 0, "ctrl")
