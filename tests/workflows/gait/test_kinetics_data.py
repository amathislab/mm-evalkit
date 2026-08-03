"""Policy-summary ingestion and shared-cohort tests for gait kinetics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mm_evalkit.datasets.gait120_kinetics import TORQUE_JOINTS
from mm_evalkit.policies import PolicySpec
from mm_evalkit.workflows.gait.kinetics_data import (
    load_kinetics_cohort,
    load_policy_kinetics,
    select_kinetics_subset,
    write_kinetics_omission_audit,
)

from .conftest import make_run


def _write_reference(path: Path) -> None:
    n_frames = 220
    time = np.arange(n_frames) * 0.01
    left = np.zeros(n_frames)
    left[:50] = 1.0
    left[100:150] = 1.0
    left[200:] = 1.0
    path.parent.mkdir(parents=True)
    np.savez(
        path,
        subject="S012",
        trial="Trial05",
        step_label="Step01_Step02",
        time_sec=time,
        joint_torque=np.column_stack([np.arange(n_frames) + 1000 * index for index in range(6)]),
        joint_torque_columns=np.asarray([f"{name}_moment" for name in TORQUE_JOINTS]),
        joint_torque_valid=np.ones(n_frames, dtype=bool),
        joint_torque_units="Nm_per_kg",
        computed_grf=np.column_stack([left, 2.0 * (1.0 - left)]),
        computed_grf_columns=np.asarray(["ground_force1_vy", "ground_force2_vy"]),
        computed_grf_valid=np.ones(n_frames, dtype=bool),
        computed_grf_units="body_weight_normalized",
    )


def _write_summary(
    path: Path,
    reference: Path,
    *,
    n_rows: int = 200,
    knee_flipped: bool = True,
) -> None:
    joint_moment = np.column_stack([np.arange(n_rows) + 10 * index for index in range(6)]).astype(float)
    # Export order is right sensor group followed by left sensor group.
    grf = np.column_stack([2.0 + np.arange(n_rows) / 1000, 1.0 + np.arange(n_rows) / 1000])
    np.savez(
        path,
        joint_names=np.asarray(TORQUE_JOINTS),
        grf_columns=np.asarray(["group_0_1_y", "group_2_3_y"]),
        grf_units="body_weight_normalized",
        torque_units="Nm_per_kg",
        policy_knee_torque_sign_flipped=knee_flipped,
        n_episodes=1,
        reference_paths=np.asarray([str(reference)]),
        reference_names=np.asarray([reference.stem]),
        episode_0_reference_path=str(reference),
        episode_0_reference_name=reference.stem,
        episode_0_timesteps=np.arange(n_rows) * 0.01,
        episode_0_joint_moment=joint_moment,
        episode_0_grf=grf,
    )


def _policy(summary: Path, *, swap: bool = False, policy_id: str = "policy") -> PolicySpec:
    return PolicySpec(
        id=policy_id,
        label=policy_id.title(),
        trajectory_path=Path("unused-rollout.npz"),
        color="blue",
        kinetics_summary_path=summary,
        kinetics_swap_left_right=swap,
    )


def _write_embedded_rollout(path: Path) -> None:
    rows = 200
    phase = np.arange(rows, dtype=float)
    model_mass_kg = 10.0
    body_weight_n = model_mass_kg * 9.80665
    raw_moments = model_mass_kg * np.column_stack([phase + 10 * index for index in range(6)])
    sensors = np.column_stack(
        [
            np.full(rows, 2.0 * body_weight_n),
            np.zeros(rows),
            np.full(rows, body_weight_n),
            np.zeros(rows),
        ]
    )
    np.savez(
        path,
        n_episodes=1,
        kinetics_sample_phase="post_step",
        generalized_actuator_force_joint_names=np.asarray(TORQUE_JOINTS),
        generalized_actuator_force_units=np.asarray(["Nm"] * 6),
        force_sensor_names=np.asarray(["r_foot", "r_toes", "l_foot", "l_toes"]),
        force_sensor_units=np.asarray(["N"] * 4),
        model_mass_kg=model_mass_kg,
        episode_0_traj_id=0,
        episode_0_generalized_actuator_forces=raw_moments,
        episode_0_force_sensor_values=sensors,
        episode_0_kinetics_reference_steps=np.arange(rows, dtype=int),
    )


def test_policy_summary_maps_exported_sensor_groups_to_anatomical_sides(tmp_path: Path) -> None:
    reference = tmp_path / "references" / "S012" / "S012_Trial05_Step01_Step02.npz"
    summary = tmp_path / "summary.npz"
    _write_reference(reference)
    _write_summary(summary, reference)

    result = load_policy_kinetics(_policy(summary), reference_dir=None)

    assert result.cycles.torque.shape == (2, 6, 100)
    assert result.cycles.grf.shape == (2, 2, 100)
    assert result.cycles.subject_ids.tolist() == [12, 12]
    assert result.cycles.gait_positions.tolist() == [0, 1]
    # Anatomical left comes from touch-sensor group 2+3, right from 0+1.
    np.testing.assert_allclose(result.cycles.grf[0, 0], 1.0 + np.arange(100) / 1000)
    np.testing.assert_allclose(result.cycles.grf[0, 1], 2.0 + np.arange(100) / 1000)
    assert result.omissions == ()


def test_generic_rollout_embeds_raw_kinetics_without_a_separate_policy_summary(tmp_path: Path) -> None:
    reference_root = tmp_path / "references"
    reference = reference_root / "S012" / "S012_Trial05_Step01_Step02.npz"
    rollout = tmp_path / "rollout.npz"
    _write_reference(reference)
    _write_embedded_rollout(rollout)
    policy = PolicySpec(
        id="policy",
        label="Policy",
        trajectory_path=rollout,
        color="blue",
    )

    result = load_policy_kinetics(
        policy,
        reference_dir=reference_root,
        motion_list=("dataset/S012/Trial05/motion",),
    )

    assert result.cycles.torque.shape == (2, 6, 100)
    assert result.cycles.grf.shape == (2, 2, 100)
    np.testing.assert_allclose(result.cycles.torque[0, 0], np.arange(100.0))
    np.testing.assert_allclose(result.cycles.torque[0, 2], -(20.0 + np.arange(100.0)))
    np.testing.assert_allclose(result.cycles.grf[0, 0], 1.0)
    np.testing.assert_allclose(result.cycles.grf[0, 1], 2.0)
    assert result.omissions == ()


def test_embedded_rollout_records_a_missing_reference_as_an_omission(tmp_path: Path) -> None:
    rollout = tmp_path / "rollout.npz"
    _write_embedded_rollout(rollout)
    policy = PolicySpec(
        id="policy",
        label="Policy",
        trajectory_path=rollout,
        color="blue",
    )

    result = load_policy_kinetics(
        policy,
        reference_dir=tmp_path / "references",
        motion_list=("dataset/S012/Trial05/motion",),
    )

    assert result.cycles.torque.shape == (0, 6, 100)
    assert result.references == {}
    assert len(result.omissions) == 1
    assert result.omissions[0].reference_name == "S012_Trial05_Step01_Step02"
    assert result.omissions[0].gait_position == -1
    assert result.omissions[0].reason == "missing_reference_file"


def test_explicit_reference_dir_rejects_a_rollout_without_embedded_kinetics(tmp_path: Path) -> None:
    rollout = tmp_path / "rollout.npz"
    np.savez(rollout, n_episodes=1)
    run = make_run("policy")
    run.policy = PolicySpec(
        id="policy",
        label="Policy",
        trajectory_path=rollout,
        color="blue",
    )

    with pytest.raises(ValueError, match="every compared policy"):
        load_kinetics_cohort(
            [run],
            reference_dir=tmp_path / "references",
            motion_list=("dataset/S012/Trial05/motion",),
        )


def test_embedded_kinetics_requires_at_least_one_matching_reference(tmp_path: Path) -> None:
    rollout = tmp_path / "rollout.npz"
    _write_embedded_rollout(rollout)
    run = make_run("policy")
    run.policy = PolicySpec(
        id="policy",
        label="Policy",
        trajectory_path=rollout,
        color="blue",
    )

    with pytest.raises(ValueError, match="no rollout episode has a matching"):
        load_kinetics_cohort(
            [run],
            reference_dir=tmp_path / "references",
            motion_list=("dataset/S012/Trial05/motion",),
        )


def test_explicit_side_swap_and_knee_sign_metadata_are_applied_once(tmp_path: Path) -> None:
    reference = tmp_path / "references" / "S012" / "S012_Trial05_Step01_Step02.npz"
    summary = tmp_path / "summary.npz"
    _write_reference(reference)
    _write_summary(summary, reference, knee_flipped=False)

    result = load_policy_kinetics(_policy(summary, swap=True), reference_dir=None)

    # Right hip receives the exported left hip after the declared correction.
    np.testing.assert_allclose(result.cycles.torque[0, 0], 10.0 + np.arange(100))
    # Right knee receives exported left knee and then the one required sign flip.
    np.testing.assert_allclose(result.cycles.torque[0, 2], -(30.0 + np.arange(100)))
    np.testing.assert_allclose(result.cycles.grf[0, 0], 2.0 + np.arange(100) / 1000)


def test_incomplete_summary_window_is_a_recorded_exclusion_not_stretched(tmp_path: Path) -> None:
    reference = tmp_path / "references" / "S012" / "S012_Trial05_Step01_Step02.npz"
    summary = tmp_path / "summary.npz"
    _write_reference(reference)
    _write_summary(summary, reference, n_rows=150)

    result = load_policy_kinetics(_policy(summary), reference_dir=None)

    assert result.cycles.gait_positions.tolist() == [0]
    assert len(result.omissions) == 1
    assert result.omissions[0].gait_position == 1
    assert result.omissions[0].reason == "incomplete_reference_window_coverage"

    run = make_run("policy")
    run.kinetics_cycles = result.cycles
    run.kinetics_omissions = result.omissions
    audit = write_kinetics_omission_audit([run], tmp_path)
    assert audit is not None
    assert "policy,0,S012_Trial05_Step01_Step02,1,incomplete_reference_window_coverage" in audit.read_text(
        encoding="utf-8"
    )


def test_reference_override_uses_portable_subject_subdirectory(tmp_path: Path) -> None:
    reference_root = tmp_path / "portable"
    reference = reference_root / "S012" / "S012_Trial05_Step01_Step02.npz"
    summary = tmp_path / "summary.npz"
    _write_reference(reference)
    _write_summary(summary, Path("/missing/machine/S012/S012_Trial05_Step01_Step02.npz"))

    result = load_policy_kinetics(_policy(summary), reference_dir=reference_root)

    assert set(result.references) == {(12, 5)}


def test_all_policies_share_references_and_build_equal_subject_profiles(tmp_path: Path) -> None:
    reference = tmp_path / "references" / "S012" / "S012_Trial05_Step01_Step02.npz"
    first_summary = tmp_path / "first.npz"
    second_summary = tmp_path / "second.npz"
    _write_reference(reference)
    _write_summary(first_summary, reference)
    _write_summary(second_summary, reference)
    runs = [make_run("first"), make_run("second")]
    runs[0].policy = _policy(first_summary, policy_id="first")
    runs[1].policy = _policy(second_summary, policy_id="second")

    cohort = load_kinetics_cohort(runs, reference_dir=None)
    assert cohort is not None
    select_kinetics_subset(runs, cohort, [12], None)

    assert cohort.subjects == [12]
    assert cohort.torque_data.shape == (1, 6, 100)
    assert all(run.kinetics_profiles.subject_ids.tolist() == [12] for run in runs)


def test_mismatched_policy_reference_samples_are_rejected(tmp_path: Path) -> None:
    first_reference = tmp_path / "first" / "S012" / "S012_Trial05_Step01_Step02.npz"
    second_reference = tmp_path / "second" / "S013" / "S013_Trial05_Step01_Step02.npz"
    first_summary = tmp_path / "first.npz"
    second_summary = tmp_path / "second.npz"
    _write_reference(first_reference)
    _write_reference(second_reference)
    with np.load(second_reference) as source:
        values = {key: source[key] for key in source.files}
    values["subject"] = "S013"
    np.savez(second_reference, **values)
    _write_summary(first_summary, first_reference)
    _write_summary(second_summary, second_reference)
    runs = [make_run("first"), make_run("second")]
    runs[0].policy = _policy(first_summary, policy_id="first")
    runs[1].policy = _policy(second_summary, policy_id="second")

    with pytest.raises(ValueError, match="do not match"):
        load_kinetics_cohort(runs, reference_dir=None)
