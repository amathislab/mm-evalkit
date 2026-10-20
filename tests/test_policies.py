"""Tests for the explicit policy configuration shared by both workflows."""

from pathlib import Path

import pytest
import yaml

from mm_evalkit import config as legacy_config
from mm_evalkit.policies import (
    PolicySpec,
    load_policy_specs,
    validate_policy_files,
)
from mm_evalkit.workflows.bimanual.config import (
    load_policies as load_bimanual_policies,
)


def test_original_config_module_reexports_the_policy_api():
    assert legacy_config.PolicySpec is PolicySpec
    assert legacy_config.load_policy_specs is load_policy_specs
    assert legacy_config.validate_policy_files is validate_policy_files


def _write_config(tmp_path: Path, policies: list[dict], **extra) -> Path:
    path = tmp_path / "policies.yaml"
    path.write_text(
        yaml.safe_dump({"policies": policies, **extra}, sort_keys=False),
        encoding="utf-8",
    )
    return path


def _policy(rollout: Path, **overrides) -> dict:
    values = {
        "id": "baseline",
        "label": "Baseline policy",
        "trajectory_data_path": str(rollout),
        "color": "#123456",
        "linestyle": "-",
    }
    values.update(overrides)
    return values


def test_policy_config_preserves_identity_separately_from_label(tmp_path: Path):
    rollout = tmp_path / "rollout.npz"
    rollout.touch()
    config = _write_config(tmp_path, [_policy(rollout)])

    policies = load_policy_specs(config)

    assert len(policies) == 1
    assert policies[0].id == "baseline"
    assert policies[0].label == "Baseline policy"
    assert policies[0].slug == "baseline"
    assert policies[0].trajectory_path == rollout
    validate_policy_files(policies)


@pytest.mark.parametrize("field", ["id", "label", "trajectory_data_path"])
def test_policy_config_requires_research_identity_fields(tmp_path: Path, field: str):
    entry = _policy(tmp_path / "rollout.npz")
    entry.pop(field)
    config = _write_config(tmp_path, [entry])

    with pytest.raises(ValueError, match=field):
        load_policy_specs(config)


def test_policy_config_rejects_unknown_fields(tmp_path: Path):
    config = _write_config(
        tmp_path,
        [_policy(tmp_path / "rollout.npz", trajectory_dat_path="typo.npz")],
    )

    with pytest.raises(ValueError, match=r"unknown field.*trajectory_dat_path"):
        load_policy_specs(config)


def test_policy_ids_must_be_unique_and_filesystem_safe(tmp_path: Path):
    first = _policy(tmp_path / "a.npz")
    duplicate = _policy(tmp_path / "b.npz", label="A different label")
    config = _write_config(tmp_path, [first, duplicate])

    with pytest.raises(ValueError, match="duplicates policy id"):
        load_policy_specs(config)

    invalid = _write_config(
        tmp_path,
        [_policy(tmp_path / "a.npz", id="Not a path / label")],
    )
    with pytest.raises(ValueError, match="lowercase"):
        load_policy_specs(invalid)


def test_earlier_trial_fields_are_an_explicit_pair(tmp_path: Path):
    config = _write_config(
        tmp_path,
        [
            _policy(
                tmp_path / "rollout.npz",
                trajectory_data_path_trials_1_4="earlier.npz",
            )
        ],
    )

    with pytest.raises(ValueError, match=r"must provide.*together"):
        load_policy_specs(config)


def test_configured_rollouts_are_validated_before_analysis(tmp_path: Path):
    missing = tmp_path / "missing.npz"
    policies = load_policy_specs(_write_config(tmp_path, [_policy(missing)]))

    with pytest.raises(FileNotFoundError, match=r"missing\.npz"):
        validate_policy_files(policies)


def test_policy_config_loads_and_validates_explicit_kinetics_metadata(tmp_path: Path):
    rollout = tmp_path / "rollout.npz"
    kinetics = tmp_path / "kinetics.npz"
    rollout.touch()
    kinetics.touch()
    policies = load_policy_specs(
        _write_config(
            tmp_path,
            [
                _policy(
                    rollout,
                    kinetics_summary_path=str(kinetics),
                    kinetics_swap_left_right=True,
                )
            ],
        )
    )

    assert policies[0].kinetics_summary_path == kinetics
    assert policies[0].kinetics_swap_left_right is True
    validate_policy_files(policies)


def test_policy_config_rejects_non_boolean_kinetics_side_swap(tmp_path: Path):
    config = _write_config(
        tmp_path,
        [_policy(tmp_path / "rollout.npz", kinetics_swap_left_right="yes")],
    )

    with pytest.raises(ValueError, match="true or false"):
        load_policy_specs(config)


def test_bimanual_rejects_gait120_only_policy_fields(tmp_path: Path):
    rollout = tmp_path / "rollout.npz"
    rollout.touch()
    config = _write_config(
        tmp_path,
        [
            _policy(
                rollout,
                trajectory_data_path_trials_1_4=str(tmp_path / "earlier.npz"),
                motion_list_trials_1_4="GAIT120_TESTING_SUBJECTS_TRIALS0104",
            )
        ],
    )

    with pytest.raises(ValueError, match="not valid for ULTRA-MoCap"):
        load_bimanual_policies(config, None)


def test_bimanual_rejects_gait120_kinetics_fields(tmp_path: Path):
    rollout = tmp_path / "rollout.npz"
    kinetics = tmp_path / "kinetics.npz"
    rollout.touch()
    kinetics.touch()
    config = _write_config(
        tmp_path,
        [_policy(rollout, kinetics_summary_path=str(kinetics))],
    )

    with pytest.raises(ValueError, match="kinetics fields"):
        load_bimanual_policies(config, None)


@pytest.mark.parametrize("signal", ["activations", "ctrl"])
def test_policy_config_rejects_analysis_wide_emg_signal(tmp_path, signal):
    config = _write_config(tmp_path, [_policy(tmp_path / "rollout.npz", emg_signal=signal)])
    with pytest.raises(ValueError, match="unknown field.*emg_signal"):
        load_policy_specs(config)
