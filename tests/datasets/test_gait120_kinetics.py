"""Contract tests for the Gait120/OpenSim kinetics reference reader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mm_kit.datasets.gait120_kinetics import (
    GRF_CHANNELS,
    TORQUE_JOINTS,
    left_heel_strike_windows,
    load_gait120_kinetics_trial,
)


def _write_reference(
    path: Path,
    *,
    torque_units: str = "Nm_per_kg",
    grf_units: str = "body_weight_normalized",
    include_mass: bool = False,
) -> None:
    n_frames = 220
    phase = np.arange(n_frames, dtype=float)
    # Two complete windows begin with left contact at rows 0 and 100. The last
    # contact at row 200 is incomplete and must not be retained.
    left = np.zeros(n_frames)
    left[0:50] = 1.0
    left[100:150] = 1.0
    left[200:] = 1.0
    right = 1.0 - left
    scale = 10.0 if torque_units == "Nm" else 1.0
    grf_scale = 10.0 * 9.80665 if grf_units == "N" else 1.0
    values = {
        "subject": "S012",
        "trial": "Trial05",
        "step_label": "Step01_Step02",
        "time_sec": phase * 0.01,
        "joint_torque": scale * np.column_stack([phase + index for index in range(len(TORQUE_JOINTS))]),
        "joint_torque_columns": np.asarray([f"{name}_moment" for name in TORQUE_JOINTS]),
        "joint_torque_valid": np.ones(n_frames, dtype=bool),
        "joint_torque_units": torque_units,
        "computed_grf": grf_scale * np.column_stack([left, right]),
        "computed_grf_columns": np.asarray(["ground_force1_vy", "ground_force2_vy"]),
        "computed_grf_valid": np.ones(n_frames, dtype=bool),
        "computed_grf_units": grf_units,
    }
    if include_mass:
        values["subject_weight_kg"] = 10.0
    path.parent.mkdir(parents=True)
    np.savez(path, **values)


def test_reference_reader_selects_channels_units_and_gait_windows(tmp_path: Path) -> None:
    path = tmp_path / "S012" / "S012_Trial05_Step01_Step02.npz"
    _write_reference(path)

    trial = load_gait120_kinetics_trial(path)

    assert trial.subject == 12
    assert trial.trial == 5
    assert trial.torque_names == TORQUE_JOINTS
    assert trial.grf_names == GRF_CHANNELS
    assert trial.gait_windows == ((0, 100), (100, 200))
    assert trial.torque_cycles.shape == (2, 6, 100)
    assert trial.grf_cycles.shape == (2, 2, 100)
    np.testing.assert_allclose(trial.torque_cycles[1, 0], np.arange(100.0, 200.0))
    np.testing.assert_allclose(trial.grf_cycles[0, 0, :50], 1.0)


def test_reference_reader_normalizes_absolute_moments_and_forces(tmp_path: Path) -> None:
    path = tmp_path / "S012" / "S012_Trial05_Step01_Step02.npz"
    _write_reference(path, torque_units="Nm", grf_units="N", include_mass=True)

    trial = load_gait120_kinetics_trial(path)

    np.testing.assert_allclose(trial.torque_cycles[0, 0], np.arange(100.0))
    np.testing.assert_allclose(trial.grf_cycles[0, 0, :50], 1.0)


def test_reference_reader_accepts_mixed_units_when_selected_force_columns_are_normalized(tmp_path: Path) -> None:
    path = tmp_path / "S012" / "S012_Trial05_Step01_Step02.npz"
    _write_reference(
        path,
        grf_units="force_columns_in_body_weight; cop_and_torque_columns_in_corrected_physical_units",
    )

    trial = load_gait120_kinetics_trial(path)

    np.testing.assert_allclose(trial.grf_cycles[0, 0, :50], 1.0)


def test_invalid_rows_keep_their_original_phase_coordinates(tmp_path: Path) -> None:
    path = tmp_path / "S012" / "S012_Trial05_Step01_Step02.npz"
    _write_reference(path)
    with np.load(path) as source:
        values = {key: source[key] for key in source.files}
    values["joint_torque_valid"][:20] = False
    np.savez(path, **values)

    trial = load_gait120_kinetics_trial(path)

    assert np.isnan(trial.torque_cycles[0, 0, :20]).all()
    np.testing.assert_allclose(trial.torque_cycles[0, 0, 20:], np.arange(20.0, 100.0))


def test_contact_windows_reject_invalid_parameters_and_incomplete_tail() -> None:
    grf = np.r_[np.zeros(5), np.ones(5), np.zeros(5), np.ones(5)]

    assert left_heel_strike_windows(grf, window_frames=10) == ((5, 15),)
    with pytest.raises(ValueError, match="window_frames"):
        left_heel_strike_windows(grf, window_frames=1)


def test_present_malformed_reference_is_not_silently_skipped(tmp_path: Path) -> None:
    path = tmp_path / "S012" / "S012_Trial05_Step01_Step02.npz"
    _write_reference(path)
    with np.load(path) as source:
        values = {key: source[key] for key in source.files if key != "joint_torque_units"}
    np.savez(path, **values)

    with pytest.raises(ValueError, match="joint_torque_units"):
        load_gait120_kinetics_trial(path)
