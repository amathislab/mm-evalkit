import pytest
import yaml

from mm_evalkit.paths import load_path_config, resolve_dataset_path, set_dataset_path


def test_explicit_path_wins_over_environment(tmp_path, monkeypatch):
    explicit = tmp_path / "explicit"
    configured = tmp_path / "configured"
    explicit.mkdir()
    configured.mkdir()
    monkeypatch.setenv("MUSCLEMIMIC_GAIT120_DATA_PATH", str(configured))

    assert resolve_dataset_path("gait120", explicit) == explicit.resolve()


def test_environment_wins_over_yaml(tmp_path, monkeypatch):
    environment = tmp_path / "environment"
    configured = tmp_path / "configured"
    environment.mkdir()
    configured.mkdir()
    config = tmp_path / "paths.yaml"
    config.write_text(
        yaml.safe_dump({"MUSCLEMIMIC_GAIT120_DATA_PATH": str(configured)}),
        encoding="utf-8",
    )
    monkeypatch.setenv("MUSCLEMIMIC_CONFIG_PATH", str(config))
    monkeypatch.setenv("MUSCLEMIMIC_GAIT120_DATA_PATH", str(environment))

    assert resolve_dataset_path("gait120") == environment.resolve()


def test_missing_configuration_explains_all_options(tmp_path, monkeypatch):
    monkeypatch.delenv("MUSCLEMIMIC_GAIT120_DATA_PATH", raising=False)
    monkeypatch.setenv("MUSCLEMIMIC_CONFIG_PATH", str(tmp_path / "missing.yaml"))

    with pytest.raises(FileNotFoundError, match="MUSCLEMIMIC_GAIT120_DATA_PATH"):
        resolve_dataset_path("gait120")


def test_set_dataset_path_preserves_other_keys(tmp_path, monkeypatch):
    dataset = tmp_path / "gait120"
    dataset.mkdir()
    config = tmp_path / "paths.yaml"
    config.write_text("OTHER_PATH: /keep/me\n", encoding="utf-8")
    monkeypatch.setenv("MUSCLEMIMIC_CONFIG_PATH", str(config))

    assert set_dataset_path("gait120", dataset) == dataset.resolve()
    assert load_path_config() == {
        "OTHER_PATH": "/keep/me",
        "MUSCLEMIMIC_GAIT120_DATA_PATH": str(dataset.resolve()),
    }
