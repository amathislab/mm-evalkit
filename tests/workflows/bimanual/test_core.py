"""Strict input and cycle-pairing contracts for the ULTra-MoCap adapter."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mm_evalkit.datasets.ultra_mocap import (
    load_aligned_emg,
)
from mm_evalkit.policies import PolicySpec
from mm_evalkit.workflows.bimanual.cli import build_parser
from mm_evalkit.workflows.bimanual.metrics import (
    _cycle_mean_std,
    _row_correlations,
    _row_rmse,
    pairwise_corr,
    split_half,
)
from mm_evalkit.workflows.bimanual.models import (
    AnalysisConfig,
    RolloutMetadata,
)
from mm_evalkit.workflows.bimanual.muscles import (
    build_muscle_groups,
)
from mm_evalkit.workflows.bimanual.pipeline import (
    analyse_policy,
    validate_comparable_trials,
)
from mm_evalkit.workflows.bimanual.references import (
    select_reference_record,
)
from mm_evalkit.workflows.bimanual.rollout import (
    resolve_trial_stems,
    validate_policy_rollouts,
    validate_rollout,
)
from mm_evalkit.workflows.bimanual.segmentation import select_declared_critical_joint
from mm_evalkit.workflows.bimanual.trials import load_trial


def test_correlations_use_finite_pairs_and_reject_constant_profiles():
    first = np.array(
        [
            [0.0, 1.0, np.nan, 2.0],
            [1.0, 1.0, np.nan, 1.0],
            [0.0, np.nan, np.nan, np.nan],
        ]
    )
    second = np.array(
        [
            [0.0, 2.0, 99.0, 4.0],
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 1.0, 2.0, 3.0],
        ]
    )

    correlations = _row_correlations(first, second)

    assert correlations[0] == pytest.approx(1.0)
    assert np.isnan(correlations[1:]).all()


def test_rmse_uses_finite_pairs():
    first = np.array([[0.0, 1.0, np.nan], [0.0, 1.0, 2.0]])
    second = np.array([[0.0, 2.0, 4.0], [1.0, 2.0, 3.0]])

    rmse = _row_rmse(first, second)

    assert rmse[0] == pytest.approx(np.sqrt(0.5))
    assert rmse[1] == pytest.approx(1.0)


def test_pairwise_correlations_preserve_pair_and_channel_shape():
    cycles = np.array(
        [
            [[0.0, 1.0, np.nan], [1.0, 1.0, 1.0]],
            [[0.0, 2.0, 3.0], [0.0, 1.0, 2.0]],
            [[2.0, 1.0, 0.0], [3.0, 2.0, 1.0]],
        ]
    )

    correlations = pairwise_corr(cycles)

    assert correlations.shape == (3, 2)
    assert correlations[0, 0] == pytest.approx(1.0)
    assert np.isnan(correlations[0, 1])


def test_cycle_summary_uses_available_finite_repetitions():
    cycles = np.array(
        [
            [[1.0, np.nan, np.nan]],
            [[3.0, 4.0, np.nan]],
        ]
    )

    mean, std = _cycle_mean_std(cycles)

    np.testing.assert_allclose(mean[0, :2], [2.0, 4.0])
    np.testing.assert_allclose(std[0, :2], [1.0, 0.0])
    assert np.isnan(mean[0, 2])
    assert np.isnan(std[0, 2])


def test_split_half_drops_an_unmatched_final_repetition():
    cycles = np.array([0.0, 10.0, 2.0, 12.0, 1_000.0])[:, None, None]

    first, second = split_half(cycles)

    np.testing.assert_allclose(first, [[1.0]])
    np.testing.assert_allclose(second, [[11.0]])


def _rollout_file(tmp_path: Path, **overrides) -> Path:
    fields = {
        "n_episodes": np.array(1),
        "dt": np.array(0.01),
        "joint_names": np.array(["shoulder_elv_r", "elbow_flex_r"]),
        "motion_group": np.array("fixture"),
        "activation_sample_phase": np.array("pre_step"),
        "episode_0_traj_id": np.array(0),
        "episode_0_traj_length": np.array(8),
        "episode_0_joint_positions": np.zeros((7, 2)),
        "episode_0_muscle_activations": np.zeros((7, 4)),
        "episode_0_traj_qpos": np.zeros((8, 2)),
    }
    fields.update(overrides)
    path = tmp_path / "rollout.npz"
    np.savez(path, **fields)
    return path


def test_cli_requires_an_explicit_motion_group(capsys):
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(["--trajectory_path", "rollout.npz"])

    assert exc_info.value.code == 2
    error = capsys.readouterr().err
    assert "--dataset_dir" in error
    assert "--motion_group" in error


def test_cli_accepts_the_shared_and_historical_spellings_of_a_flag():
    """One canonical name per input, with kebab-case and the old name as aliases."""
    group = "ULTRA_MOCAP_BIMANUAL_TESTING_SUBJECTS"
    canonical = build_parser().parse_args(
        ["--trajectory_path", "rollout.npz", "--motion_group", group, "--dataset_dir", "/data"]
    )
    aliased = build_parser().parse_args(
        ["--trajectory-path", "rollout.npz", "--motion-group", group, "--ultramocap_roots", "/data"]
    )

    assert canonical.motion_group == aliased.motion_group == group
    assert canonical.dataset_dir == aliased.dataset_dir == ["/data"]


def test_rollout_validation_accepts_the_documented_schema(tmp_path):
    path = _rollout_file(tmp_path)
    with np.load(path, allow_pickle=True) as data:
        metadata = validate_rollout(data, path, expected_activation_count=4)

    assert metadata.n_episodes == 1
    assert metadata.dt == pytest.approx(0.01)
    assert metadata.joint_names == ("shoulder_elv_r", "elbow_flex_r")
    assert metadata.activation_count == 4


def test_rollout_validation_checks_embedded_motion_group(tmp_path):
    path = _rollout_file(tmp_path)
    with np.load(path, allow_pickle=True) as data:
        with pytest.raises(ValueError, match="archive motion_group"):
            validate_rollout(
                data,
                path,
                expected_activation_count=4,
                expected_motion_group="other-group",
            )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"dt": np.array(0.02)}, "dt must be 0.01"),
        (
            {"episode_0_muscle_activations": np.zeros((6, 4))},
            "muscle_activations has 6 rows",
        ),
        (
            {"episode_0_joint_positions": np.zeros((7, 3))},
            "3 columns for 2 joint names",
        ),
    ],
)
def test_rollout_validation_rejects_incompatible_arrays(
    tmp_path,
    override,
    message,
):
    path = _rollout_file(tmp_path, **override)
    with np.load(path, allow_pickle=True) as data:
        with pytest.raises(ValueError, match=message):
            validate_rollout(data, path, expected_activation_count=4)


def test_policy_comparison_requires_identical_reference_episodes(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first_path = _rollout_file(first_dir)
    second_path = _rollout_file(
        second_dir,
        episode_0_traj_qpos=np.ones((8, 2)),
    )
    policies = [
        PolicySpec("first", "First", first_path, "blue"),
        PolicySpec("second", "Second", second_path, "orange"),
    ]

    with pytest.raises(ValueError, match="different trajectory IDs or reference"):
        validate_policy_rollouts(
            policies,
            expected_activation_count=4,
            expected_motion_group="fixture",
        )


def test_trial_identity_is_the_declared_motion_at_exported_traj_id(tmp_path):
    path = _rollout_file(tmp_path, episode_0_traj_id=np.array(1))
    motion_paths = (
        "ULTRA_MOCAP/P11/P11_ArmSwing_Slow",
        "ULTRA_MOCAP/P12/P12_ElbowFlexion_Fast",
    )
    with np.load(path, allow_pickle=True) as trajectory:
        metadata = validate_rollout(
            trajectory,
            path,
            expected_activation_count=4,
        )
        assert resolve_trial_stems(
            trajectory,
            metadata,
            motion_paths,
        ) == {0: "P12_ElbowFlexion_Fast"}


def test_trial_identity_rejects_traj_id_outside_declared_motion_group(tmp_path):
    path = _rollout_file(tmp_path, episode_0_traj_id=np.array(2))
    with np.load(path, allow_pickle=True) as trajectory:
        metadata = validate_rollout(
            trajectory,
            path,
            expected_activation_count=4,
        )
        with pytest.raises(ValueError, match="outside the declared motion group"):
            resolve_trial_stems(
                trajectory,
                metadata,
                ("ULTRA_MOCAP/P11/P11_ArmSwing_Slow",),
            )


def test_untagged_partial_rollout_cannot_be_mapped_to_larger_group():
    trajectory = {f"episode_{episode}_traj_id": np.array(episode) for episode in range(39)}
    metadata = RolloutMetadata(
        path=Path("legacy-39.npz"),
        n_episodes=39,
        dt=0.01,
        joint_names=("shoulder_elv_r",),
        activation_count=4,
    )
    motion_paths = tuple(f"ULTRA_MOCAP/P11/P11_Trial_{index:02d}" for index in range(48))

    with pytest.raises(
        ValueError,
        match=r"no motion_group provenance.*39 episodes.*48",
    ):
        resolve_trial_stems(
            trajectory,
            metadata,
            motion_paths,
            episodes=(0,),
        )


def test_untagged_rollout_requires_complete_unique_traj_id_coverage():
    metadata = RolloutMetadata(
        path=Path("untagged.npz"),
        n_episodes=2,
        dt=0.01,
        joint_names=("shoulder_elv_r",),
        activation_count=4,
    )
    motion_paths = (
        "ULTRA_MOCAP/P11/P11_ArmSwing_Slow",
        "ULTRA_MOCAP/P12/P12_ElbowFlexion_Fast",
    )

    complete = {
        "episode_0_traj_id": np.array(0),
        "episode_1_traj_id": np.array(1),
    }
    assert resolve_trial_stems(
        complete,
        metadata,
        motion_paths,
        episodes=(1,),
    ) == {1: "P12_ElbowFlexion_Fast"}

    duplicate = {
        "episode_0_traj_id": np.array(0),
        "episode_1_traj_id": np.array(0),
    }
    with pytest.raises(ValueError, match=r"cover 0\.\.1 exactly once"):
        resolve_trial_stems(
            duplicate,
            metadata,
            motion_paths,
        )


def _two_episode_rollout(
    tmp_path: Path,
    *,
    episode_0_reference: np.ndarray,
    episode_1_reference: np.ndarray,
    episode_1_traj_id: int = 1,
) -> Path:
    return _rollout_file(
        tmp_path,
        n_episodes=np.array(2),
        episode_0_traj_qpos=episode_0_reference,
        episode_1_traj_id=np.array(episode_1_traj_id),
        episode_1_traj_length=np.array(len(episode_1_reference)),
        episode_1_joint_positions=np.zeros((len(episode_1_reference) - 1, episode_1_reference.shape[1])),
        episode_1_muscle_activations=np.zeros((len(episode_1_reference) - 1, 4)),
        episode_1_traj_qpos=episode_1_reference,
    )


def test_policy_comparison_checks_only_requested_episode_subset(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    third_dir = tmp_path / "third"
    first_dir.mkdir()
    second_dir.mkdir()
    third_dir.mkdir()
    selected_reference = np.ones((8, 2))
    first = _two_episode_rollout(
        first_dir,
        episode_0_reference=np.zeros((8, 2)),
        episode_1_reference=selected_reference,
    )
    differs_only_outside_subset = _two_episode_rollout(
        second_dir,
        episode_0_reference=np.full((8, 2), 7.0),
        episode_1_reference=selected_reference,
    )
    differs_inside_subset = _two_episode_rollout(
        third_dir,
        episode_0_reference=np.zeros((8, 2)),
        episode_1_reference=selected_reference,
        episode_1_traj_id=0,
    )

    policies = [
        PolicySpec("first", "First", first, "blue"),
        PolicySpec(
            "outside",
            "Outside-only difference",
            differs_only_outside_subset,
            "orange",
        ),
    ]
    assert validate_policy_rollouts(
        policies,
        expected_activation_count=4,
        expected_motion_group="fixture",
        episodes=[1],
    ) == ("shoulder_elv_r", "elbow_flex_r")

    policies[1] = PolicySpec(
        "inside",
        "Requested difference",
        differs_inside_subset,
        "red",
    )
    with pytest.raises(ValueError, match="requested episode"):
        validate_policy_rollouts(
            policies,
            expected_activation_count=4,
            expected_motion_group="fixture",
            episodes=[1],
        )


def test_analyse_policy_runs_alignment_segmentation_and_exact_metrics(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "emg"
    emg_dir = root / "P11" / "Arm Swing" / "EMG_preprocessed"
    emg_dir.mkdir(parents=True)
    biceps = np.tile([0.0, 1.0, 0.0, -1.0], 2)
    triceps = np.tile([1.0, 0.0, -1.0, 0.0], 2)
    deltoid = np.tile([0.0, 1.0, 2.0, 0.0], 2)
    pd.DataFrame(
        {
            "time": np.arange(8) / 100.0,
            "IM EMG4": biceps,
            "IM EMG5": triceps,
            "IM EMG6": deltoid,
        }
    ).to_csv(
        emg_dir / "P11_ArmSwing_Slow_preprocessed.csv",
        index=False,
    )

    activations = np.column_stack([biceps, biceps, triceps, deltoid])
    rollout_path = _rollout_file(
        tmp_path,
        joint_names=np.array(["shoulder_elv_r"]),
        episode_0_traj_length=np.array(8),
        episode_0_joint_positions=np.ones((8, 1)),
        episode_0_muscle_activations=activations,
        episode_0_traj_qpos=np.zeros((8, 1)),
    )
    monkeypatch.setattr(
        "mm_evalkit.workflows.bimanual.trials.select_declared_critical_joint",
        lambda *_args, **_kwargs: (
            "shoulder_elv_r",
            1.0,
            4,
            [(0, 4), (4, 8)],
        ),
    )

    groups, muscle_columns = build_muscle_groups(
        ["BIClong", "BICshort", "TRIlong", "DELT2"],
        suffix="",
    )
    policy = PolicySpec("test", "Test policy", rollout_path, "blue")
    options = AnalysisConfig(
        policies=(policy,),
        emg_roots=(root,),
        motion_group="fixture",
        motion_paths=("ULTRA_MOCAP/P11/P11_ArmSwing_Slow",),
        plots_dir=tmp_path / "plots",
        episodes=(0,),
        arm="right",
        all_joints=True,
        raw_trials=False,
    )
    result = analyse_policy(
        policy,
        options,
        joint_names=("shoulder_elv_r",),
        activation_count=4,
        muscle_columns=muscle_columns,
        groups=groups,
        joint_selection=[0],
    )

    assert result.requested_episodes == 1
    assert result.exclusions == ()
    assert set(result.records) == {("ArmSwing", "Slow", "P11")}
    metrics = result.records[("ArmSwing", "Slow", "P11")]
    np.testing.assert_allclose(
        metrics["joint_rmse_by"],
        [[180.0 / np.pi]],
    )
    np.testing.assert_allclose(metrics["muscle_corr_by"], [[1.0, 1.0, 1.0]])
    np.testing.assert_allclose(metrics["muscle_rsa"], [1.0])


def test_trial_identity_requires_a_nonempty_declared_motion_group(tmp_path):
    metadata = RolloutMetadata(
        path=tmp_path / "rollout.npz",
        n_episodes=1,
        dt=0.01,
        joint_names=("shoulder_elv_r", "elbow_flex_r"),
        activation_count=4,
    )
    with pytest.raises(ValueError, match="empty"):
        resolve_trial_stems({}, metadata, ())


def test_emg_alignment_rejects_missing_required_channels(tmp_path):
    path = tmp_path / "emg.csv"
    pd.DataFrame({"time": [0.0, 0.01], "IM EMG4": [0.2, 0.3]}).to_csv(
        path,
        index=False,
    )

    with pytest.raises(ValueError, match="missing required EMG channel"):
        load_aligned_emg(
            path,
            n_frames=2,
            fs=100.0,
            required_channels=("IM EMG4", "IM EMG5"),
        )


def test_emg_alignment_rejects_off_grid_and_duplicate_timestamps(tmp_path):
    off_grid = tmp_path / "off_grid.csv"
    duplicate = tmp_path / "duplicate.csv"
    pd.DataFrame({"time": [0.0, 0.015], "IM EMG4": [0.2, 0.3]}).to_csv(
        off_grid,
        index=False,
    )
    pd.DataFrame({"time": [0.0, 0.0], "IM EMG4": [0.2, 0.3]}).to_csv(
        duplicate,
        index=False,
    )

    with pytest.raises(ValueError, match="expected 100 Hz grid"):
        load_aligned_emg(off_grid, n_frames=2, fs=100.0)
    with pytest.raises(ValueError, match="strictly increasing"):
        load_aligned_emg(duplicate, n_frames=2, fs=100.0)


def test_muscle_mapping_requires_every_anatomical_target():
    with pytest.raises(ValueError, match="BICshort"):
        build_muscle_groups(["BIClong", "TRIlong", "DELT2"], suffix="")


def test_arm_swing_segmentation_uses_only_declared_shoulder_elevation(
    monkeypatch,
):
    qpos = np.column_stack(
        (
            [0.0, 10.0, 0.0, 10.0, 0.0, 10.0],
            [0.0, 50.0, 0.0, 50.0, 0.0, 50.0],
        )
    )
    monkeypatch.setattr(
        "mm_evalkit.workflows.bimanual.segmentation.segment_cycles",
        lambda signal, _fs: [(0, 2), (2, 4), (4, 6)],
    )
    monkeypatch.setattr(
        "mm_evalkit.workflows.bimanual.segmentation.cycle_shape_consistency",
        lambda _signal, _boundaries: 1.0,
    )
    monkeypatch.setattr(
        "mm_evalkit.workflows.bimanual.segmentation.periodicity_strength",
        lambda _signal, _fs: (0.9, 2),
    )

    selected = select_declared_critical_joint(
        qpos,
        ["shoulder_elv_r", "shoulder_rot_r"],
        "ArmSwing",
    )

    assert selected[0] == "shoulder_elv_r"


def test_reference_selection_accepts_identical_real_records():
    first = {"ref_cycle": np.full((1, 3), 1.0)}
    second = {"ref_cycle": np.full((1, 3), 1.0)}
    entries = [
        (PolicySpec("first", "First", Path("first.npz"), "blue"), first),
        (PolicySpec("second", "Second", Path("second.npz"), "red"), second),
    ]

    assert select_reference_record(entries) is first


def test_reference_selection_rejects_different_real_records():
    first = {"ref_cycle": np.full((1, 3), 1.0)}
    second = {"ref_cycle": np.full((1, 3), 2.0)}
    entries = [
        (PolicySpec("first", "First", Path("first.npz"), "blue"), first),
        (PolicySpec("second", "Second", Path("second.npz"), "red"), second),
    ]

    with pytest.raises(ValueError, match="different real-data field"):
        select_reference_record(entries)


def test_policy_comparison_requires_identical_trial_and_cycle_sets():
    policies = [
        PolicySpec("first", "First", Path("first.npz"), "blue"),
        PolicySpec("second", "Second", Path("second.npz"), "red"),
    ]
    key = ("ArmSwing", "Slow", "P11")
    baseline_trial = SimpleNamespace(
        cycle_boundaries=((0, 4), (4, 8)),
        muscle_cycle_boundaries=((0, 4),),
    )
    matching_trial = SimpleNamespace(
        cycle_boundaries=((0, 4), (4, 8)),
        muscle_cycle_boundaries=((0, 4),),
    )
    results = {
        "first": SimpleNamespace(trials={key: baseline_trial}),
        "second": SimpleNamespace(trials={key: matching_trial}),
    }

    validate_comparable_trials(results, policies)

    matching_trial.muscle_cycle_boundaries = ((4, 8),)
    with pytest.raises(ValueError, match="different paired muscle boundaries"):
        validate_comparable_trials(results, policies)


def test_trial_requires_a_preprocessed_emg_recording(tmp_path):
    groups, muscle_columns = build_muscle_groups(
        ["BIClong", "BICshort", "TRIlong", "DELT2"],
        suffix="",
    )
    trajectory = {
        "activation_sample_phase": np.array("pre_step"),
        "episode_0_traj_length": np.array(8),
        "episode_0_joint_positions": np.zeros((8, 1)),
        "episode_0_traj_qpos": np.zeros((8, 1)),
        "episode_0_muscle_activations": np.zeros((8, 4)),
    }

    with pytest.raises(FileNotFoundError, match="No preprocessed EMG file matches"):
        load_trial(
            trajectory,
            episode=0,
            trial_stem="P11_ArmSwing_Slow",
            joint_names=["elbow_flex_r"],
            muscle_columns=muscle_columns,
            groups=groups,
            arm="right",
            emg_roots=[tmp_path],
        )


def test_trial_uses_identical_complete_cycles_for_activation_and_emg(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "emg"
    emg_dir = root / "P11" / "Arm Swing" / "EMG_preprocessed"
    emg_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "time": np.arange(8) / 100.0,
            "IM EMG4": np.arange(8, dtype=float),
            "IM EMG5": np.arange(8, dtype=float) + 1.0,
            "IM EMG6": np.arange(8, dtype=float) + 2.0,
        }
    ).to_csv(
        emg_dir / "P11_ArmSwing_Slow_preprocessed.csv",
        index=False,
    )

    muscle_names = ["BIClong", "BICshort", "TRIlong", "DELT2"]
    groups, muscle_columns = build_muscle_groups(muscle_names, suffix="")
    trajectory = {
        "activation_sample_phase": np.array("pre_step"),
        "episode_0_traj_length": np.array(8),
        "episode_0_joint_positions": np.zeros((8, 1)),
        "episode_0_traj_qpos": np.zeros((8, 1)),
        "episode_0_muscle_activations": np.arange(32, dtype=float).reshape(8, 4),
    }
    selector_call = {}

    def fake_select_declared_critical_joint(
        qpos,
        names,
        movement,
        *,
        arm,
        fs,
    ):
        selector_call.update(
            qpos=qpos,
            names=names,
            movement=movement,
            arm=arm,
            fs=fs,
        )
        return (
            "shoulder_elv_r",
            1.0,
            4,
            [(0, 4), (4, 8)],
        )

    monkeypatch.setattr(
        "mm_evalkit.workflows.bimanual.trials.select_declared_critical_joint",
        fake_select_declared_critical_joint,
    )

    trial = load_trial(
        trajectory,
        episode=0,
        trial_stem="P11_ArmSwing_Slow",
        joint_names=["shoulder_elv_r"],
        muscle_columns=muscle_columns,
        groups=groups,
        arm="right",
        emg_roots=[str(root)],
    )

    assert trial.cycle_boundaries == ((0, 4), (4, 8))
    assert trial.muscle_cycle_boundaries == trial.cycle_boundaries
    np.testing.assert_array_equal(
        trial.policy_activation,
        trajectory["episode_0_muscle_activations"],
    )
    assert selector_call["names"] == ["shoulder_elv_r"]
    assert selector_call["movement"] == "ArmSwing"
    assert selector_call["arm"] == "right"
    assert selector_call["fs"] == pytest.approx(100.0)
    assert trial.activation_cycles.shape == (2, 4, 100)
    assert trial.emg_cycles.shape == (2, 3, 100)
