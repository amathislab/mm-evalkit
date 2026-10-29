"""Keep EMG analysis and video signal selection consistent."""

from types import SimpleNamespace

import numpy as np
import pytest

from mm_evalkit.workflows.gait import analysis, render
from mm_evalkit.workflows.gait.channels import muscle_names_myo
from mm_evalkit.workflows.gait.cli import build_parser


@pytest.mark.parametrize("signal", [None, "activation"])
def test_analysis_defaults_to_excitation_and_accepts_activation(tmp_path, monkeypatch, signal):
    monkeypatch.setattr(analysis, "load_gait_analysis", lambda config: object())
    options = {} if signal is None else {"muscle_signal": signal}
    result = analysis.FullBodyAnalysis(
        dataset_dir=tmp_path,
        policy_config_file=tmp_path / "policies.yaml",
        motion_group="GAIT120_TESTING",
        **options,
    )
    assert result.config.muscle_signal == (signal or "excitation")


def test_analysis_cli_defaults_to_excitation():
    args = build_parser(fixed_dataset="Gait120").parse_args(
        ["--motion_group", "GAIT120_TESTING", "--policy_config_file", "policies.yaml"]
    )
    assert args.muscle_signal == "excitation"


@pytest.fixture
def video_rollout(tmp_path, monkeypatch):
    frames = 12
    muscles = len(muscle_names_myo)
    qpos = np.arange(frames * 89, dtype=float).reshape(frames, 89)
    activation = np.broadcast_to(np.linspace(0.1, 0.7, frames)[:, None], (frames, muscles)).copy()
    commands = np.broadcast_to(np.linspace(-0.5, 1.5, frames)[:, None], (frames, muscles)).copy()
    arrays = {
        "episode_0_joint_positions": qpos,
        "episode_0_muscle_activations": activation,
        "episode_0_muscle_commands": commands,
        "episode_0_touch_observations": np.zeros((frames, 4)),
        "episode_0_traj_id": 0,
        "episode_0_traj_length": frames,
        "activation_sample_phase": "post_step",
        "muscle_command_sample_phase": "post_step",
    }
    path = tmp_path / "rollout.npz"
    np.savez(path, **arrays)
    monkeypatch.setattr(
        render, "extract_valid_strides_flexible", lambda *_args, **_kwargs: [SimpleNamespace(start=2, end=8)]
    )
    monkeypatch.setattr(
        render,
        "load_gait120_emg_step",
        lambda *_args, **kwargs: np.zeros((len(kwargs["selected_muscle_names"]), 100)),
    )
    return path, arrays


@pytest.mark.parametrize("signal", [None, "activation"])
def test_video_uses_selected_signal_on_the_state_timeline(video_rollout, tmp_path, signal):
    path, arrays = video_rollout
    options = {} if signal is None else {"muscle_signal": signal}
    clip = render.prepare_clip(trajectory_path=path, episode=0, dataset_dir=tmp_path, **options)
    expected = (
        arrays["episode_0_muscle_activations"][1:7]
        if signal == "activation"
        else np.clip(arrays["episode_0_muscle_commands"][2:8], 0, 1)
    )
    np.testing.assert_array_equal(clip.qpos, arrays["episode_0_joint_positions"][2:8])
    np.testing.assert_array_equal(clip.activations, expected)
    for comparison, channel in zip(clip.comparisons, render.DEFAULT_CHANNELS, strict=True):
        np.testing.assert_array_equal(comparison.policy, expected[:, muscle_names_myo.index(channel)])
        assert comparison.policy_label == f"Policy {signal or 'excitation'}"
    assert clip.metadata["muscle_signal"] == (signal or "excitation")


def test_video_requires_commands_without_falling_back_to_activation(video_rollout, tmp_path):
    path, arrays = video_rollout
    del arrays["episode_0_muscle_commands"]
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="episode_0_muscle_commands"):
        render.prepare_clip(trajectory_path=path, episode=0, dataset_dir=tmp_path)


def test_video_rejects_unknown_muscle_signal(video_rollout, tmp_path):
    path, _ = video_rollout
    with pytest.raises(ValueError, match="muscle_signal"):
        render.prepare_clip(trajectory_path=path, episode=0, dataset_dir=tmp_path, muscle_signal="emg")
