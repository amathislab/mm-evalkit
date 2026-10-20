"""Render signal selection keeps controls and legacy activations on their timelines."""
import numpy as np
import pandas as pd
import pytest

from mm_evalkit.workflows.bimanual import render


@pytest.mark.parametrize('signal', ['ctrl', 'activations'])
def test_clip_uses_selected_signal_and_labels(tmp_path, monkeypatch, signal):
    values = np.arange(24, dtype=float).reshape(6, 4)
    path = tmp_path / 'rollout.npz'
    np.savez(path, joint_names=['shoulder_elv_r'],
             episode_0_joint_positions=np.zeros((6, 1)), episode_0_traj_qpos=np.zeros((6, 1)),
             episode_0_muscle_activations=values, episode_0_muscle_commands=values+100)
    monkeypatch.setattr(render, 'load_muscle_names', lambda: ['BIClong', 'BICshort', 'TRIlong', 'DELT2'])
    monkeypatch.setattr(render, 'find_preprocessed_emg', lambda *_: tmp_path/'emg.csv')
    monkeypatch.setattr(render, 'load_aligned_emg', lambda *_, **__: pd.DataFrame(
        {channel:np.arange(6, dtype=float) for channel in render.CHANNELS}))
    monkeypatch.setattr(render, 'select_declared_critical_joint', lambda *_, **__: (
        'shoulder_elv_r', 1.0, 2, [(1, 3), (3, 5)]))
    options = {} if signal == 'ctrl' else {'emg_signal':signal}
    clip = render.prepare_clip(trajectory_path=path, episode=0, trial_stem='P11_ArmSwing_Slow',
                               dataset_dir=tmp_path, cycles=2, **options)
    expected = np.clip(values+100, 0.0, 1.0)[1:6] if signal == 'ctrl' else values[:5]
    np.testing.assert_array_equal(clip.activations, expected)
    np.testing.assert_array_equal(clip.comparisons[0].policy, expected[:, :2].mean(axis=1))
    assert clip.metadata['emg_signal'] == signal
    assert ('ctrl' in clip.comparisons[0].policy_label) == (signal == 'ctrl')
