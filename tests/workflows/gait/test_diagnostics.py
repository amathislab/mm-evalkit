"""Policy identities must survive identical display labels in diagnostics."""
from types import SimpleNamespace

from mm_evalkit.policies import PolicySpec
from mm_evalkit.workflows.gait import diagnostics


def test_gait_count_table_keeps_duplicate_labels_separate(tmp_path, monkeypatch):
    captured = {}
    def capture(*args):
        captured['cells'] = args[1]
    monkeypatch.setattr(diagnostics, 'render_gait_count_table_png', capture)
    monkeypatch.setattr(diagnostics, 'save_event_sequences_txt', lambda _: None)
    runs = []
    for signal, count in [('activations', 1), ('ctrl', 2)]:
        policy = PolicySpec(id=signal, label='Same label', trajectory_path=tmp_path/'rollout.npz',
                            color='blue')
        runs.append(SimpleNamespace(policy=policy, episode_gait_records=[
            {'subject_id':52,'n_gaits':count,'gait_indexes':[(1,100)]*count}]))
    analysis=SimpleNamespace(subject_based=True,human=SimpleNamespace(subjects=[52]),
                             policies=runs,plots_dir=tmp_path)
    diagnostics.build_and_save_gait_count_table(analysis)
    assert analysis.gait_count_table.shape == (1,4)
    assert analysis.gait_count_table.loc[52,('activations','total')] == 1
    assert analysis.gait_count_table.loc[52,('ctrl','total')] == 2
    assert len(captured['cells'][0]) == 6
