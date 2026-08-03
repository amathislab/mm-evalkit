"""Sensitivity of a policy scored against the human's earlier trials."""

import numpy as np
import pytest

from mm_evalkit.workflows.gait.metrics import compute_rsm
from mm_evalkit.workflows.gait.references import compute_policy_to_earlier_human_sensitivity
from mm_evalkit.workflows.gait.state import PolicyProfiles

from .conftest import make_analysis, make_channels, make_cohort, make_run

JOINTS = ["hip", "knee"]
MUSCLES = ["m1", "m2", "m3"]


def _muscle_profile(offset=0.0):
    time = np.linspace(0, 2 * np.pi, 100)
    return np.stack(
        [
            np.sin(time) + offset,
            np.cos(time) + offset,
            np.sin(2 * time) + 0.3 * np.cos(time) + offset,
        ]
    )


def _analysis_with_synthetic_profiles():
    """One policy with profiles for subjects 51 and 52; 53 is human-only."""
    time = np.linspace(0, 1, 100)
    joint_51 = np.stack([time, time**2])
    joint_52 = np.stack([2 * time, 2 * time**2])
    muscle_51 = _muscle_profile()
    muscle_52 = _muscle_profile(offset=0.2)

    run = make_run("test policy")
    run.profiles = PolicyProfiles(
        avg_joint_gaits=joint_51,
        std_joint_gaits=np.zeros_like(joint_51),
        avg_joint_gaits_ref=joint_51,
        std_joint_gaits_ref=np.zeros_like(joint_51),
        avg_muscle_gaits=muscle_51,
        std_muscle_gaits=np.zeros_like(muscle_51),
        population_muscle_rsm=compute_rsm(muscle_51),
        subject_ids=np.array([51, 52]),
        per_subject_joint={51: joint_51, 52: joint_52},
        per_subject_muscle={51: muscle_51, 52: muscle_52},
        per_subject_muscle_rsm={51: compute_rsm(muscle_51), 52: compute_rsm(muscle_52)},
    )

    cohort = make_cohort([51, 52, 53], joints=JOINTS, muscles=MUSCLES)
    # The policy is scored against each subject's earlier-trial profile.
    cohort.earlier_joint_by_step = {51: {0: joint_51}, 52: {0: joint_52 + 1.0}, 53: {0: joint_51}}
    cohort.earlier_muscle_by_step = {
        51: {0: 2 * muscle_51 + 0.5},
        52: {0: 2 * muscle_52 - 0.5},
        53: {0: muscle_51},
    }

    analysis = make_analysis(
        policies=[run],
        cohort=cohort,
        channels=make_channels(JOINTS, MUSCLES),
    )
    analysis.gait_position = 0
    return analysis


def test_policy_to_earlier_human_sensitivity_is_subject_matched():
    analysis = _analysis_with_synthetic_profiles()

    compute_policy_to_earlier_human_sensitivity(analysis)

    result = analysis.policies[0].earlier_human_sensitivity
    # Subject 53 has no policy profile, so it is skipped rather than scored.
    assert result.n == 2
    assert result.subjects == [51, 52]
    assert result.joint_rmse["all_joints"] == pytest.approx([0.0, 1.0])
    assert result.muscle_corr["all_muscles"] == pytest.approx([1.0, 1.0])
    assert result.rsa_spearman == pytest.approx([1.0, 1.0])
    assert result.rsa_pearson == pytest.approx([1.0, 1.0])


def test_policy_to_earlier_human_sensitivity_skips_missing_human_gait():
    analysis = _analysis_with_synthetic_profiles()
    analysis.human.earlier_joint_by_step[52] = {}
    analysis.human.earlier_muscle_by_step[52] = {}

    compute_policy_to_earlier_human_sensitivity(analysis)

    result = analysis.policies[0].earlier_human_sensitivity
    assert result.n == 1
    assert result.subjects == [51]
