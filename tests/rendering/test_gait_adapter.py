"""Gait demo controls remain unshifted; legacy activation follows state time."""
from types import SimpleNamespace

import numpy as np
import pytest

from mm_evalkit.workflows.gait import render


@pytest.mark.parametrize('signal', ['ctrl', 'activations'])
def test_gait_clip_uses_selected_signal(tmp_path, monkeypatch, signal):
    values = np.arange(6*len(render.muscle_names_myo), dtype=float).reshape(6, -1)
    path = tmp_path/'rollout.npz'
    np.savez(path, episode_0_joint_positions=np.zeros((6, len(render.joint_names_myo)+6)),
             episode_0_muscle_activations=values, episode_0_muscle_commands=values+1000,
             episode_0_touch_observations=np.zeros((6, 4)), episode_0_traj_id=0, episode_0_traj_length=6)
    monkeypatch.setattr(render, 'get_feet_contacts', lambda **_: None)
    monkeypatch.setattr(render, 'extract_valid_strides_flexible', lambda *_, **__: [SimpleNamespace(start=1, end=5)])
    monkeypatch.setattr(render, 'subject_from_trajectory', lambda *_: 52)
    monkeypatch.setattr(render, 'load_gait120_emg_step', lambda *_, **__: np.tile(np.arange(100), (2, 1)))
    options = {} if signal == 'ctrl' else {'emg_signal':signal}
    clip = render.prepare_clip(trajectory_path=path, episode=0, dataset_dir=tmp_path,
                               channels=('vaslat_r', 'bflh_r'), **options)
    expected = np.clip(values+1000, 0.0, 1.0)[1:5] if signal=='ctrl' else values[:4]
    np.testing.assert_array_equal(clip.activations, expected)
    np.testing.assert_array_equal(clip.comparisons[0].policy, expected[:, render.muscle_names_myo.index('vaslat_r')])
    assert clip.metadata['emg_signal'] == signal
    assert ('ctrl' in clip.comparisons[0].policy_label) == (signal=='ctrl')
