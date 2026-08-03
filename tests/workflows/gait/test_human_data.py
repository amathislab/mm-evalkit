"""Gait selection: one position at a time, or every gait pooled."""

import numpy as np
import pytest

from mm_kit.workflows.gait import human_data
from mm_kit.workflows.gait.state import PolicyCycles

from .conftest import JOINTS, MUSCLES, make_analysis, make_cohort, make_run


def _step(seed, n_channels):
    """A distinguishable 100-sample profile per channel."""
    return np.tile(np.arange(100, dtype=float) / 100.0, (n_channels, 1)) + seed


def _two_subject_analysis():
    """Subject 51 walks two steps, subject 52 only one."""
    joint_by_step = {
        51: {0: _step(0.0, len(JOINTS)), 1: _step(2.0, len(JOINTS))},
        52: {0: _step(1.0, len(JOINTS))},
    }
    muscle_by_step = {
        51: {0: _step(0.0, len(MUSCLES)), 1: _step(2.0, len(MUSCLES))},
        52: {0: _step(1.0, len(MUSCLES))},
    }
    cohort = make_cohort([51, 52], by_step=(joint_by_step, muscle_by_step))

    # One cycle per (subject, gait position), matching the human steps available.
    joints = np.stack([_step(float(10 * i), len(JOINTS)) for i in range(3)])
    muscles = np.stack([_step(float(10 * i), len(MUSCLES)) for i in range(3)])
    cycles = PolicyCycles(
        joint=joints,
        joint_ref=joints,
        muscle=muscles,
        subject_ids=np.array([51, 51, 52]),
        trial_ids=np.array([5, 5, 5]),
        gait_positions=np.array([0, 1, 0]),
    )
    analysis = make_analysis(policies=[make_run("policy", cycles=cycles)], cohort=cohort)
    return analysis, joints


def test_select_gait_subset_restricts_to_one_position():
    analysis, joints = _two_subject_analysis()

    human_data.select_gait_subset(analysis, 1)

    # Only subject 51 has a second human step, so only it is analysed.
    assert analysis.human.subjects == [51]
    assert analysis.human.joint_data[0] == pytest.approx(analysis.human.joint_by_step[51][1])
    assert analysis.policies[0].profiles.per_subject_joint[51] == pytest.approx(joints[1])


def test_select_gait_subset_pools_every_gait():
    analysis, joints = _two_subject_analysis()

    human_data.select_gait_subset(analysis, None)

    # Every subject with any step takes part, and the human ground truth is the mean over
    # that subject's steps -- matching how the policy side pools its cycles.
    assert analysis.human.subjects == [51, 52]
    assert analysis.human.joint_data[0] == pytest.approx(
        np.mean([analysis.human.joint_by_step[51][0], analysis.human.joint_by_step[51][1]], axis=0)
    )
    assert analysis.human.joint_data[1] == pytest.approx(analysis.human.joint_by_step[52][0])
    profiles = analysis.policies[0].profiles.per_subject_joint
    assert profiles[51] == pytest.approx(np.mean(joints[:2], axis=0))
    assert profiles[52] == pytest.approx(joints[2])


def test_policy_comparison_requires_shared_subjects():
    """Policies without a shared subject cannot form a paired comparison."""
    analysis, _ = _two_subject_analysis()
    source = analysis.policies[0].cycles

    def policy_subset(policy_id, mask):
        return make_run(
            policy_id,
            cycles=PolicyCycles(
                joint=source.joint[mask],
                joint_ref=source.joint_ref[mask],
                muscle=source.muscle[mask],
                subject_ids=source.subject_ids[mask],
                trial_ids=source.trial_ids[mask],
                gait_positions=source.gait_positions[mask],
            ),
        )

    analysis.policies = [
        policy_subset("only-51", source.subject_ids == 51),
        policy_subset("only-52", source.subject_ids == 52),
    ]

    with pytest.raises(ValueError, match="no shared subjects"):
        human_data.select_gait_subset(analysis, None)


def test_gait_selection_names():
    assert human_data.gait_selection_name(0) == "gait1"
    assert human_data.gait_selection_name(1) == "gait2"
    assert human_data.gait_selection_name(None) == "all_gaits"


def test_state_rejects_an_undeclared_field():
    """The schema is enforced: a misspelled stage output fails immediately."""
    analysis, _ = _two_subject_analysis()

    with pytest.raises(AttributeError):
        analysis.ref_within_subject = {}
