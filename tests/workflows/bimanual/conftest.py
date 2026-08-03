"""Synthetic per-trial records shared by the ULTra-MoCap wiring tests."""

import pytest

from mm_evalkit.stats.rsa import compute_rsm


class Policy:
    """The policy metadata the reporting stages read, without a rollout file."""

    def __init__(self, label, color="#2a78d6"):
        self.id = label
        self.label = label
        self.color = color


def rsm(rng, n_channels):
    """A plausible symmetric RSM: correlations of random channel patterns."""
    return compute_rsm(rng.normal(0.0, 1.0, (n_channels, 60)))


def bimanual_records(rng, offset, keys, n_joints, n_channels, with_reference=False):
    """One metrics dict per trial key, optionally with the real-data references."""
    records = {}
    for key in keys:
        metrics = {
            "joint_rmse_by": rng.normal(6.0 + offset, 1.0, (1, n_joints)),
            "muscle_corr_by": rng.normal(0.5, 0.1, (1, n_channels)),
            "muscle_rsa": rng.normal(0.4 - 0.1 * offset, 0.1, 1),
            "act_rsm": rsm(rng, n_channels),
        }
        if with_reference:
            metrics["joint_rmse_within_by"] = rng.normal(2.0, 0.4, (1, n_joints))
            metrics["muscle_corr_within_by"] = rng.normal(0.85, 0.05, (1, n_channels))
            metrics["muscle_rsa_within"] = rng.normal(0.9, 0.03, 1)
            metrics["joint_rmse_pairwise_by"] = rng.normal(3.0, 0.4, n_joints)
            metrics["muscle_corr_pairwise_by"] = rng.normal(0.7, 0.05, n_channels)
            metrics["muscle_rsa_pairwise_by"] = rng.normal(0.75, 0.05, 1)
            metrics["emg_rsm"] = rsm(rng, n_channels)
        records[key] = metrics
    return records


@pytest.fixture
def make_records():
    return bimanual_records


@pytest.fixture
def policy_factory():
    return Policy
