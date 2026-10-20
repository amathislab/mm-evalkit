"""Synthetic per-trial records shared by the ULTRA-MoCap wiring tests."""

import numpy as np
import pytest

from mm_evalkit.stats.rsa import compute_rsm


class Policy:
    """The policy metadata the reporting stages read, without a rollout file."""

    def __init__(self, label, color="#2a78d6", linestyle="-"):
        self.id = label
        self.label = label
        self.color = color
        # The figures read the linestyle to pick the policy's fill texture, exactly as
        # they do from a real PolicySpec.
        self.linestyle = linestyle


def rsm(rng, n_channels):
    """A plausible symmetric RSM: correlations of random channel patterns."""
    return compute_rsm(rng.normal(0.0, 1.0, (n_channels, 60)))


def bimanual_records(rng, offset, keys, n_joints, n_channels, with_reference=False, with_cycles=False):
    """One metrics dict per trial key, optionally with the real-data references.

    ``with_cycles`` adds the average-cycle waveforms a real record carries, which the
    grid figures plot; the scores stay the same either way.
    """
    records = {}
    n_points = 100
    for key in keys:
        joint_rmse_by = rng.normal(6.0 + offset, 1.0, (1, n_joints))
        muscle_corr_by = rng.normal(0.5, 0.1, (1, n_channels))
        metrics = {
            "joint_rmse_by": joint_rmse_by,
            # The aggregate keys real records carry beside the per-channel ones.
            "joint_rmse": joint_rmse_by.mean(axis=1),
            "muscle_corr_by": muscle_corr_by,
            "muscle_corr": muscle_corr_by.mean(axis=1),
            "muscle_rsa": rng.normal(0.4 - 0.1 * offset, 0.1, 1),
            "act_rsm": rsm(rng, n_channels),
        }
        if with_cycles:
            phase = np.linspace(0, 2 * np.pi, n_points)
            metrics["pol_cycle"] = np.stack([np.sin(phase + index) for index in range(n_joints)])
            metrics["pol_std"] = np.full((n_joints, n_points), 0.1)
            metrics["act_cycle"] = np.stack([0.5 + 0.4 * np.sin(phase + index) for index in range(n_channels)])
            metrics["act_std"] = np.full((n_channels, n_points), 0.05)
        if with_reference:
            if with_cycles:
                phase = np.linspace(0, 2 * np.pi, n_points)
                metrics["ref_cycle"] = np.stack([np.cos(phase + index) for index in range(n_joints)])
                metrics["ref_std"] = np.full((n_joints, n_points), 0.2)
                metrics["emg_cycle"] = np.stack([0.4 + 0.3 * np.cos(phase + index) for index in range(n_channels)])
                metrics["emg_std"] = np.full((n_channels, n_points), 0.05)
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
