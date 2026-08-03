"""Cross-policy identity and cohort contracts for the full-body adapter."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mm_evalkit.policies import PolicySpec
from mm_evalkit.workflows.gait.human_data import (
    use_reference_as_joint_ground_truth,
)
from mm_evalkit.workflows.gait.rollout_validation import (
    validate_policy_rollouts,
)

from .conftest import make_analysis, make_channels, make_cohort, make_run


def _write_rollout(
    path: Path,
    *,
    reference: np.ndarray | None = None,
    motion_group: str = "TEST_GROUP",
) -> None:
    reference = np.zeros((4, 8), dtype=float) if reference is None else np.asarray(reference, dtype=float)
    np.savez(
        path,
        n_episodes=np.array(1),
        dt=np.array(0.01),
        joint_names=np.array(["hip", "knee"]),
        motion_group=np.array(motion_group),
        episode_0_traj_id=np.array(0),
        episode_0_traj_length=np.array(len(reference)),
        episode_0_joint_positions=np.zeros((len(reference), 8)),
        episode_0_traj_qpos=reference,
        episode_0_muscle_activations=np.zeros((len(reference), 12)),
        episode_0_touch_observations=np.zeros((len(reference), 4)),
    )


def _policy(policy_id: str, path: Path) -> PolicySpec:
    return PolicySpec(policy_id, policy_id.title(), path, "blue")


def _validate(policies: list[PolicySpec]) -> None:
    validate_policy_rollouts(
        policies,
        subject_based=True,
        motion_count=1,
        expected_joint_names=("hip", "knee"),
        expected_joint_width=8,
        expected_activation_width=12,
        expected_motion_group="TEST_GROUP",
    )


def test_policy_rollouts_require_identical_references(tmp_path):
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    _write_rollout(first)
    _write_rollout(second, reference=np.ones((4, 8)))

    with pytest.raises(ValueError, match="trajectory ID or reference waveform"):
        _validate([_policy("first", first), _policy("second", second)])


def test_policy_rollout_motion_group_is_explicit_provenance(tmp_path):
    path = tmp_path / "rollout.npz"
    _write_rollout(path, motion_group="OTHER_GROUP")

    with pytest.raises(ValueError, match="expected 'TEST_GROUP'"):
        _validate([_policy("policy", path)])


def test_mfb_reference_rejects_policy_dependent_segmentations():
    """Two policies segmenting the same subject differently is an error, not an average."""
    reference = np.zeros((2, 100))

    def policy_run(label, subject_reference):
        run = make_run(label)
        run.profiles = replace(run.profiles, per_subject_joint_ref={11: subject_reference})
        return run

    analysis = make_analysis(
        policies=[policy_run("first", reference), policy_run("second", reference + 1.0)],
        cohort=make_cohort([11], joints=["hip", "knee"]),
        channels=make_channels(["hip", "knee"], ["m1"]),
    )

    with pytest.raises(ValueError, match="differs between policy"):
        use_reference_as_joint_ground_truth(analysis)
