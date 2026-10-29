"""Focused behavior tests for the full-body neuromuscular analysis core.

These tests intentionally use tiny in-memory profiles.  They protect statistical
and input-contract behavior without depending on the Gait120/Wang datasets or on
large rollout archives.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from mm_evalkit.datasets import (
    gait120 as gait120_loader,
)
from mm_evalkit.datasets import wang as wang_loader
from mm_evalkit.workflows.gait import analysis as gait_analysis
from mm_evalkit.workflows.gait import human_data, main, policy_data
from mm_evalkit.workflows.gait.channels import joint_names_myo, muscle_names_myo
from mm_evalkit.workflows.gait.cli import build_parser
from mm_evalkit.workflows.gait.config import FullBodyConfig
from mm_evalkit.workflows.gait.constants import MOTION_SETS
from mm_evalkit.workflows.gait.references import (
    compute_subject_profile_metrics,
)
from mm_evalkit.workflows.gait.rollout_validation import validate_rollout_archive

from .conftest import make_analysis, make_channels, make_cohort, make_run

CHANNELS = make_channels(["hip", "knee"], ["m1", "m2", "m3"])


def test_population_profiles_weight_subjects_equally():
    """Extra cycles from one subject must not give that subject extra weight."""
    subject_ids = np.array([11, 11, 11, 22])

    # Subject 11 has three cycles with a mean of 2; subject 22 has one cycle at
    # 10.  Equal-subject weighting therefore gives 6, whereas pooling all four
    # cycles directly would incorrectly give 4.
    cycle_offsets = np.array([0.0, 2.0, 4.0, 10.0])
    joint = np.stack([np.full((2, 5), offset) for offset in cycle_offsets])
    joint_reference = joint + 100.0

    muscle_template = np.array(
        [
            [0.0, 1.0, 2.0, 1.0, 0.0],
            [0.0, 1.0, 4.0, 2.0, 0.0],
            [1.0, 0.0, 2.0, 3.0, 1.0],
        ]
    )
    muscle = np.stack([muscle_template + offset for offset in cycle_offsets])

    profiles = human_data.build_policy_profiles(
        joint,
        joint_reference,
        muscle,
        subject_ids,
    )

    assert profiles.per_subject_joint[11] == pytest.approx(2.0)
    assert profiles.per_subject_joint[22] == pytest.approx(10.0)
    assert profiles.avg_joint_gaits == pytest.approx(6.0)
    assert profiles.avg_joint_gaits_ref == pytest.approx(106.0)
    assert profiles.avg_muscle_gaits == pytest.approx(muscle_template + 6.0)

    cycle_weighted_joint_mean = np.mean(joint, axis=0)
    assert not np.allclose(profiles.avg_joint_gaits, cycle_weighted_joint_mean)


def test_profile_correlations_use_finite_pairs_and_preserve_undefined_values():
    """Missing samples are paired; constant profiles remain undefined."""
    time = np.linspace(0.0, 1.0, 5)
    model_joint = np.stack((time, time**2))
    human_joint = model_joint.copy()
    model_muscle = np.stack(
        (
            np.array([0.0, 1.0, np.nan, 3.0, 4.0]),
            np.zeros_like(time),
            np.sin(time),
        )
    )
    human_muscle = np.stack((time, time, np.sin(time)))

    _, _, muscle_corr, _, _ = compute_subject_profile_metrics(
        model_joint,
        model_muscle,
        human_joint,
        human_muscle,
    )

    assert muscle_corr[0] == pytest.approx(1.0)
    assert np.isnan(muscle_corr[1])
    assert muscle_corr[2] == pytest.approx(1.0)


def test_gait120_mot_reader_has_a_strict_local_contract(tmp_path: Path):
    """The adapter parses and normalizes a valid OpenSim MOT step itself."""
    path = tmp_path / "step01.mot"
    path.write_text(
        "name fixture\nendheader\ntime hip_flexion_r knee_angle_r\n0.0 1.0 4.0\n0.5 2.0 5.0\n1.0 3.0 6.0\n",
        encoding="utf-8",
    )

    profile, names = gait120_loader._load_mot_step(path)

    assert names == ("time", "hip_flexion_r", "knee_angle_r")
    assert profile.shape == (3, 100)
    np.testing.assert_allclose(profile[1, [0, -1]], [1.0, 3.0])
    assert gait120_loader._load_mot_step(tmp_path / "missing.mot") is None


def test_gait120_loader_selects_paired_steps_and_reports_missing_subjects(
    tmp_path: Path,
    monkeypatch,
):
    """Only locally parsed joint/EMG pairs enter the selected profiles."""
    primary = gait120_loader._TrialGroupData(
        joint_by_step={1: {0: np.stack([np.full(100, 1.0), np.full(100, 3.0)])}},
        muscle_by_step={1: {0: np.stack([np.full(100, 5.0), np.full(100, 7.0)])}},
        joint_names=("hip_flexion_r", "knee_angle_r"),
        muscle_names=("emg1", "emg2"),
        omissions=[],
    )
    earlier = gait120_loader._TrialGroupData(
        joint_by_step={1: {0: np.stack([np.full(100, 11.0), np.full(100, 13.0)])}},
        muscle_by_step={1: {0: np.stack([np.full(100, 15.0), np.full(100, 17.0)])}},
        joint_names=("hip_flexion_r", "knee_angle_r"),
        muscle_names=("emg1", "emg2"),
        omissions=[],
    )
    monkeypatch.setattr(
        gait120_loader,
        "_load_subject_movement",
        lambda _path: object(),
    )
    monkeypatch.setattr(
        gait120_loader,
        "_load_trial_group",
        lambda _root, *, trials, **_kwargs: primary if tuple(trials) == (5,) else earlier,
    )

    loaded = gait120_loader.load_gait120(
        tmp_path,
        requested_subjects=[1, 2],
        selected_joint_names=["knee_angle_r"],
        selected_muscle_names=["m1"],
        muscle_map={"m1": "emg2"},
        missing_subject_policy="drop-and-report",
    )

    assert loaded.subjects == (1,)
    assert loaded.joint_names == ("knee_angle_r",)
    assert loaded.muscle_names == ("m1",)
    # The documented OpenSim sign correction is applied after selection.
    np.testing.assert_allclose(loaded.joint_by_step[1][0], -3.0)
    np.testing.assert_allclose(loaded.muscle_by_step[1][0], 7.0)
    np.testing.assert_allclose(loaded.earlier_joint_by_step[1][0], -13.0)
    np.testing.assert_allclose(loaded.earlier_muscle_by_step[1][0], 17.0)
    assert loaded.joint_data.shape == (1, 1, 100)
    assert loaded.muscle_data.shape == (1, 1, 100)
    assert loaded.omissions[0].subject == 2
    assert loaded.omissions[0].reason == "no_valid_paired_step"


def test_gait120_strict_loading_rejects_a_missing_requested_subject(
    tmp_path: Path,
    monkeypatch,
):
    """The explicit error policy rejects an unavailable subject."""
    monkeypatch.setattr(
        gait120_loader,
        "_load_subject_movement",
        lambda _path: None,
    )

    with pytest.raises(ValueError, match="could not retain requested subject"):
        gait120_loader.load_gait120(
            tmp_path,
            requested_subjects=[1],
            selected_joint_names=["hip_flexion_r"],
            selected_muscle_names=["m1"],
            muscle_map={"m1": "emg1"},
            missing_subject_policy="error",
        )


def test_wang_loader_returns_selected_population_contract(monkeypatch):
    """Wang channel selection is independent of Gait120 step semantics."""

    def fake_profiles(_dataset_dir):
        joints = np.arange(12, dtype=float).reshape(2, 3, 2)
        muscles = np.arange(16, dtype=float).reshape(2, 4, 2)
        return joints, muscles, ("a", "b", "c"), ("x", "y", "z", "w")

    monkeypatch.setattr(wang_loader, "load_wang_profiles", fake_profiles)
    loaded = wang_loader.load_wang(
        "unused",
        selected_joint_names=["c", "a"],
        selected_muscle_names=["m2", "m1"],
        muscle_map={"m1": "x", "m2": "z"},
    )

    assert loaded.joint_names == ("c", "a")
    assert loaded.muscle_names == ("m2", "m1")
    assert loaded.joint_data == pytest.approx(np.arange(12, dtype=float).reshape(2, 3, 2)[:, [2, 0], :])
    assert loaded.muscle_data == pytest.approx(np.arange(16, dtype=float).reshape(2, 4, 2)[:, [2, 0], :])


def test_wang_loader_requires_declared_source_files(tmp_path: Path):
    """Stale generated arrays must not substitute for a missing Wang MAT file."""
    np.save(tmp_path / "EMG_subject_04_walk_45_avg_r.npy", np.zeros((100, 10)))

    with pytest.raises(FileNotFoundError, match=r"Subj04_walk_45\.mat"):
        wang_loader.load_wang_profiles(tmp_path)


def test_cli_requires_dataset_and_policy_config_paths(capsys):
    """A full-body run has no implicit human-data or policy-config input."""
    with pytest.raises(SystemExit) as exc_info:
        main([])

    assert exc_info.value.code == 2
    error = capsys.readouterr().err
    assert "--dataset_dir" in error
    assert "--policy_config_file" in error


def test_fullbody_constructor_accepts_the_legacy_motion_list_keyword(monkeypatch):
    received = {}
    config = object()
    state = object()

    def from_arguments(**arguments):
        received.update(arguments)
        return config

    monkeypatch.setattr(gait_analysis.FullBodyConfig, "from_arguments", staticmethod(from_arguments))
    monkeypatch.setattr(gait_analysis, "load_gait_analysis", lambda supplied: state if supplied is config else None)

    analysis = gait_analysis.FullBodyAnalysis(motion_list_Gait120="GAIT120_TESTING")

    assert received["motion_set_name"] == "GAIT120_TESTING"
    assert analysis.state is state


def test_fullbody_constructor_rejects_conflicting_motion_group_aliases():
    with pytest.raises(ValueError, match="must match"):
        gait_analysis.FullBodyAnalysis(
            motion_group="GAIT120_TESTING",
            motion_list_Gait120="GAIT120_TESTING_SUBJECTS",
        )


def test_policy_loader_requires_validated_policy_spec():
    """The rollout loader accepts only policy records validated from YAML."""
    with pytest.raises(TypeError, match="validated PolicySpec"):
        policy_data.load_policy_run(
            CHANNELS,
            {"label": "policy without a trajectory"},
            subject_based=True,
        )


def test_missing_trajectory_fails_explicitly(tmp_path: Path):
    """Missing rollout data must not silently become all-NaN profiles."""
    missing_path = tmp_path / "missing_rollout.npz"

    with pytest.raises(FileNotFoundError, match=r"missing_rollout\.npz"):
        policy_data.extract_trajectory_data(
            CHANNELS,
            missing_path,
            subject_based=True,
            motion_list=[],
        )


def test_malformed_rollout_fails_before_extraction(tmp_path: Path):
    """A present NPZ with missing scientific arrays is invalid, not an empty policy."""
    malformed_path = tmp_path / "malformed.npz"
    np.savez(malformed_path, n_episodes=1)

    with pytest.raises(ValueError, match="missing required key"):
        policy_data.extract_trajectory_data(
            CHANNELS,
            malformed_path,
            subject_based=True,
            motion_list=[object()],
        )


def test_early_termination_is_an_audited_scientific_exclusion(tmp_path: Path):
    """A structurally valid but short rollout remains a per-episode skip record."""
    rollout_path = tmp_path / "early.npz"
    np.savez(
        rollout_path,
        n_episodes=1,
        dt=0.01,
        joint_names=np.asarray(joint_names_myo),
        episode_0_traj_id=0,
        episode_0_traj_length=10,
        episode_0_joint_positions=np.zeros((5, len(joint_names_myo) + 6)),
        episode_0_traj_qpos=np.zeros((10, len(joint_names_myo) + 6)),
        episode_0_muscle_activations=np.zeros((5, len(muscle_names_myo))),
        episode_0_muscle_commands=np.zeros((5, len(muscle_names_myo))),
        episode_0_touch_observations=np.zeros((5, 4)),
    )

    extracted = policy_data.extract_trajectory_data(
        CHANNELS,
        rollout_path,
        subject_based=True,
        motion_list=MOTION_SETS["GAIT120_TESTING_SUBJECTS"][:1],
    )

    assert extracted.all_gait_lengths.size == 0
    assert extracted.skipped_episodes == [
        {
            "episode": 0,
            "traj_id": 0,
            "subject_id": 51,
            "reason": "early_termination",
            "episode_length": 5,
            "reference_length": 10,
        }
    ]
    assert extracted.episode_gait_records[0]["reason"] == "early_termination"


def test_excitation_is_clipped_muscle_command(tmp_path: Path):
    """Excitation is the [0, 1]-limited control, not the exported activation."""
    from mm_evalkit.activation import excitation_from_commands

    commands = np.array([[-0.5, 0.25, 1.0], [0.0, 0.75, 1.0]])
    np.testing.assert_array_equal(excitation_from_commands(commands), [[0.0, 0.25, 1.0], [0.0, 0.75, 1.0]])


def test_excitation_analysis_requires_muscle_commands(tmp_path: Path):
    """An archive without commands cannot silently fall back to activation."""
    rollout_path = tmp_path / "no_commands.npz"
    np.savez(
        rollout_path,
        n_episodes=1,
        dt=0.01,
        joint_names=np.asarray(joint_names_myo),
        episode_0_traj_id=0,
        episode_0_traj_length=5,
        episode_0_joint_positions=np.zeros((5, len(joint_names_myo) + 6)),
        episode_0_traj_qpos=np.zeros((5, len(joint_names_myo) + 6)),
        episode_0_muscle_activations=np.zeros((5, len(muscle_names_myo))),
        episode_0_touch_observations=np.zeros((5, 4)),
    )
    with pytest.raises(ValueError, match="episode_0_muscle_commands"):
        policy_data.extract_trajectory_data(
            CHANNELS,
            rollout_path,
            subject_based=True,
            motion_list=MOTION_SETS["GAIT120_TESTING_SUBJECTS"][:1],
            muscle_signal="excitation",
        )


def test_muscle_signal_is_validated(tmp_path: Path):
    with pytest.raises(ValueError, match="muscle_signal"):
        FullBodyConfig.from_arguments(
            dataset_name="Wang",
            dataset_dir=tmp_path,
            policy_config_file=tmp_path / "policies.yaml",
            subject_based=False,
            motion_set_name=None,
            plots_root=tmp_path / "plots",
            kinematics_source="opensim-ik",
            human_subject_policy="drop-and-report",
            muscle_signal="emg",
        )


def test_wang_rejects_subject_based_mode(tmp_path: Path):
    """Wang cannot silently downgrade a requested subject-based analysis."""
    with pytest.raises(ValueError, match="set subject_based=False explicitly"):
        FullBodyConfig.from_arguments(
            dataset_name="Wang",
            dataset_dir=tmp_path,
            policy_config_file=tmp_path / "policies.yaml",
            subject_based=True,
            motion_set_name="GAIT120_TESTING_SUBJECTS",
            plots_root=tmp_path / "plots",
            kinematics_source="opensim-ik",
            human_subject_policy="drop-and-report",
        )


def test_gait120_requires_explicit_trial_five_provenance(
    tmp_path: Path,
    capsys,
):
    """A rollout must never be decoded through an implicit motion-list default."""
    with pytest.raises(SystemExit) as exc_info:
        build_parser(fixed_dataset="Gait120").parse_args(
            [
                "--dataset_dir",
                str(tmp_path),
                "--policy_config_file",
                str(tmp_path / "policies.yaml"),
            ]
        )
    assert exc_info.value.code == 2
    assert "--motion_group" in capsys.readouterr().err

    args = build_parser(fixed_dataset="Gait120").parse_args(
        [
            "--dataset_dir",
            str(tmp_path),
            "--policy_config_file",
            str(tmp_path / "policies.yaml"),
            "--motion_group",
            "GAIT120_TESTING",
        ]
    )
    assert args.motion_group == "GAIT120_TESTING"


def test_gait120_cli_accepts_portable_kinetics_reference_root(tmp_path: Path):
    args = build_parser(fixed_dataset="Gait120").parse_args(
        [
            "--dataset_dir",
            str(tmp_path),
            "--policy_config_file",
            str(tmp_path / "policies.yaml"),
            "--motion_group",
            "GAIT120_TESTING",
            "--kinetics-reference-dir",
            str(tmp_path / "kinetics"),
        ]
    )

    assert args.kinetics_reference_dir == str(tmp_path / "kinetics")


def test_kinetics_must_be_configured_for_every_compared_policy(tmp_path: Path):
    rollout_a = tmp_path / "a.npz"
    rollout_b = tmp_path / "b.npz"
    summary = tmp_path / "kinetics.npz"
    rollout_a.touch()
    rollout_b.touch()
    summary.touch()
    policy_config = tmp_path / "policies.yaml"
    policy_config.write_text(
        yaml.safe_dump(
            {
                "policies": [
                    {
                        "id": "a",
                        "label": "A",
                        "trajectory_data_path": str(rollout_a),
                        "kinetics_summary_path": str(summary),
                    },
                    {
                        "id": "b",
                        "label": "B",
                        "trajectory_data_path": str(rollout_b),
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    config = FullBodyConfig.from_arguments(
        dataset_name="Gait120",
        dataset_dir=tmp_path,
        policy_config_file=policy_config,
        subject_based=True,
        motion_set_name="GAIT120_TESTING",
        plots_root=tmp_path / "plots",
        kinematics_source="opensim-ik",
        human_subject_policy="drop-and-report",
    )

    with pytest.raises(ValueError, match="every compared policy"):
        config.load_policies()


def test_kinetics_reference_dir_uses_the_normal_rollout_without_policy_summaries(tmp_path: Path):
    rollout = tmp_path / "rollout.npz"
    rollout.touch()
    references = tmp_path / "references"
    references.mkdir()
    policy_config = tmp_path / "policies.yaml"
    policy_config.write_text(
        yaml.safe_dump(
            {
                "policies": [
                    {
                        "id": "policy",
                        "label": "Policy",
                        "trajectory_data_path": str(rollout),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    config = FullBodyConfig.from_arguments(
        dataset_name="Gait120",
        dataset_dir=tmp_path,
        policy_config_file=policy_config,
        subject_based=True,
        motion_set_name="GAIT120_TESTING",
        plots_root=tmp_path / "plots",
        kinematics_source="opensim-ik",
        human_subject_policy="drop-and-report",
        kinetics_reference_dir=references,
    )

    policies = config.load_policies()

    assert len(policies) == 1
    assert policies[0].kinetics_summary_path is None


def test_primary_gait120_analysis_rejects_an_all_trials_motion_set(
    tmp_path: Path,
):
    """Human trial 5 cannot be paired with rollout IDs decoded as trials 1-5."""
    with pytest.raises(ValueError, match="requires a trial-5 motion set"):
        FullBodyConfig.from_arguments(
            dataset_name="Gait120",
            dataset_dir=tmp_path,
            policy_config_file=tmp_path / "policies.yaml",
            subject_based=True,
            motion_set_name="GAIT120_TESTING_SUBJECTS",
            plots_root=tmp_path / "plots",
            kinematics_source="opensim-ik",
            human_subject_policy="drop-and-report",
        )


def test_mfb_ground_truth_requires_subject_based_gait120(tmp_path: Path):
    """An incompatible MFB mode is a config error, not an OpenSim-IK downgrade."""
    with pytest.raises(ValueError, match="only defined for subject-based Gait120"):
        FullBodyConfig.from_arguments(
            dataset_name="Gait120",
            dataset_dir=tmp_path,
            policy_config_file=tmp_path / "policies.yaml",
            subject_based=False,
            motion_set_name="GAIT120_TESTING_SUBJECTS",
            plots_root=tmp_path / "plots",
            kinematics_source="mfb-retargeted",
            human_subject_policy="drop-and-report",
        )


def test_mfb_reference_requires_every_selected_subject():
    """A mixed OpenSim/MFB ground truth is rejected."""
    run = make_run("Test policy")
    run.profiles = replace(run.profiles, per_subject_joint_ref={11: np.ones((2, 100))})
    cohort = make_cohort([11, 22], joints=["hip", "knee"])
    analysis = make_analysis(policies=[run], cohort=cohort, channels=CHANNELS)
    analysis.gait_position = 0

    with pytest.raises(ValueError, match="subject 22 is missing"):
        human_data.use_reference_as_joint_ground_truth(analysis)


def _rollout_archive() -> dict[str, np.ndarray]:
    return {
        "n_episodes": np.array(1),
        "dt": np.array(0.01),
        "joint_names": np.array(["hip", "knee"]),
        "episode_0_traj_id": np.array(0),
        "episode_0_traj_length": np.array(4),
        "episode_0_joint_positions": np.zeros((4, 8)),
        "episode_0_traj_qpos": np.zeros((4, 8)),
        "episode_0_muscle_activations": np.zeros((4, 12)),
        "episode_0_touch_observations": np.zeros((4, 4)),
    }


def test_fullbody_rollout_schema_is_validated_before_segmentation():
    archive = _rollout_archive()

    assert (
        validate_rollout_archive(
            archive,
            "rollout.npz",
            subject_based=True,
            motion_count=1,
            expected_joint_names=("hip", "knee"),
            expected_joint_width=8,
            expected_activation_width=12,
        )
        == 1
    )

    del archive["episode_0_muscle_activations"]
    with pytest.raises(ValueError, match="episode_0_muscle_activations"):
        validate_rollout_archive(
            archive,
            "rollout.npz",
            subject_based=True,
            motion_count=1,
            expected_joint_names=("hip", "knee"),
            expected_joint_width=8,
            expected_activation_width=12,
        )


def test_fullbody_rollout_rejects_wrong_motion_identity_and_time_axis():
    archive = _rollout_archive()
    archive["episode_0_traj_id"] = np.array(3)
    with pytest.raises(ValueError, match="outside the supplied motion list"):
        validate_rollout_archive(
            archive,
            "rollout.npz",
            subject_based=True,
            motion_count=1,
        )

    archive = _rollout_archive()
    archive["episode_0_touch_observations"] = np.zeros((3, 4))
    with pytest.raises(ValueError, match="4 joint frames but 3 touch_observations"):
        validate_rollout_archive(
            archive,
            "rollout.npz",
            subject_based=False,
            motion_count=None,
        )


def test_fullbody_rollout_validates_optional_embedded_kinetics():
    archive = _rollout_archive()
    archive.update(
        {
            "kinetics_sample_phase": np.array("post_step"),
            "generalized_actuator_force_joint_names": np.array(["hip", "knee"]),
            "generalized_actuator_force_units": np.array(["Nm", "Nm"]),
            "force_sensor_names": np.array(["r_foot", "r_toes", "l_foot", "l_toes"]),
            "force_sensor_units": np.array(["N", "N", "N", "N"]),
            "model_mass_kg": np.array(80.0),
            "episode_0_generalized_actuator_forces": np.zeros((4, 2)),
            "episode_0_force_sensor_values": np.zeros((4, 4)),
            "episode_0_kinetics_reference_steps": np.arange(4, dtype=int),
        }
    )

    assert validate_rollout_archive(archive, "rollout.npz", subject_based=False, motion_count=None) == 1

    archive["episode_0_kinetics_reference_steps"] = np.array([0, 1, 1, 3])
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_rollout_archive(archive, "rollout.npz", subject_based=False, motion_count=None)
