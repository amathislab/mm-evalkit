"""New-format GRF validation must not silently fall back to legacy sensors."""

import numpy as np
import pytest

from mm_evalkit.workflows.gait.world_grf import load_world_grf


def archive():
    return {
        "grf_coordinate_frame": "world",
        "grf_units": "N",
        "grf_sides": np.array(["left", "right"]),
        "grf_sample_phase": "solver_interval",
        "grf_control_reduction": "last_physics_step",
        "gravity_m_s2": np.array([0, 0, -10]),
        "model_mass_kg": 10.0,
        "episode_0_ground_reaction_forces": np.array([[[0.0, 0.0, 100.0], [0.0, 0.0, 200.0]]]),
        "episode_0_kinetics_interval_start_times": np.array([0.008]),
        "episode_0_kinetics_interval_end_times": np.array([0.01]),
    }


def test_legacy_absence_and_new_side_order():
    assert load_world_grf({}, "episode_0", 1, "legacy.npz") is None
    np.testing.assert_array_equal(load_world_grf(archive(), "episode_0", 1, "new.npz"), [[1.0, 2.0]])


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("grf_sides", np.array(["right", "right"])),
        ("grf_coordinate_frame", "contact"),
        ("grf_sample_phase", "pre_step"),
        ("grf_units", "BW"),
        ("gravity_m_s2", np.zeros(3)),
        ("model_mass_kg", -1.0),
        ("episode_0_ground_reaction_forces", np.zeros((2, 3))),
        ("episode_0_ground_reaction_forces", np.full((1, 2, 3), np.nan)),
        ("episode_0_kinetics_interval_start_times", np.array([0.02])),
    ],
)
def test_invalid_world_grf_is_rejected(key, value):
    data = archive()
    data[key] = value
    with pytest.raises(ValueError):
        load_world_grf(data, "episode_0", 1, "invalid.npz")


def test_partial_world_grf_is_rejected():
    with pytest.raises(ValueError, match="missing fields"):
        load_world_grf({"grf_units": "N"}, "episode_0", 1, "partial.npz")
