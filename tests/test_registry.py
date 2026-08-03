"""Dataset selection, plugin discovery, and lazy command dispatch.

These are the seams a new dataset is added through, so they are covered here
rather than only through the workflows they happen to reach.
"""

from types import SimpleNamespace

import pytest

from mm_kit import registry
from mm_kit.cli import main as cli_main


def test_registry_exposes_supported_research_datasets():
    adapters = registry.registered_datasets()
    assert [adapter.name for adapter in adapters] == ["gait120", "wang", "ultra-mocap"]
    assert [adapter.workflow for adapter in adapters] == ["gait", "gait", "bimanual"]
    assert [adapter.body_model for adapter in adapters] == ["fullbody", "fullbody", "bimanual"]


def test_registry_targets_public_workflow_entrypoints():
    adapters = registry.registered_datasets()
    assert {(adapter.module, adapter.entrypoint) for adapter in adapters} == {
        ("mm_kit.workflows.gait", "gait120_main"),
        ("mm_kit.workflows.gait", "wang_main"),
        ("mm_kit.workflows.bimanual", "main"),
    }


def test_dataset_names_are_case_insensitive_and_aliased():
    assert registry.get_dataset("GAIT120").name == "gait120"
    assert registry.get_dataset("ultramocap").name == "ultra-mocap"


def test_registry_rejects_unknown_dataset():
    with pytest.raises(KeyError, match="Unknown neuromuscular dataset"):
        registry.get_dataset("unknown")


def test_adapter_forwards_arguments_without_rewriting_them(monkeypatch):
    received = []
    module = SimpleNamespace(wang_main=lambda argv: received.append(argv) or 7)
    monkeypatch.setattr(registry.importlib, "import_module", lambda _name: module)

    assert registry.run_dataset("wang", ["--dataset_dir", "/data"]) == 7
    assert received == [["--dataset_dir", "/data"]]


def test_plugin_datasets_are_discovered_without_being_imported(monkeypatch):
    """An installed third-party adapter appears in the listing; loading it is deferred."""
    loaded = []

    def load():
        loaded.append(True)
        return lambda argv: 0

    entry_point = SimpleNamespace(
        name="external-set",
        module="third_party.adapter",
        attr="main",
        value="third_party.adapter:main",
        load=load,
    )
    monkeypatch.setattr(
        registry.metadata,
        "entry_points",
        lambda: SimpleNamespace(select=lambda group: (entry_point,) if group == "mm_kit.datasets" else ()),
    )

    adapter = registry.get_dataset("external-set")

    assert adapter.workflow == "plugin"
    assert not loaded, "listing a plugin must not import it"
    assert adapter.run([]) == 0
    assert loaded == [True]


def test_a_plugin_may_not_shadow_a_built_in_dataset(monkeypatch):
    entry_point = SimpleNamespace(
        name="gait120",
        module="third_party.adapter",
        attr="main",
        value="third_party.adapter:main",
        load=lambda: (_ for _ in ()).throw(AssertionError("built-in must win")),
    )
    monkeypatch.setattr(
        registry.metadata,
        "entry_points",
        lambda: SimpleNamespace(select=lambda group: (entry_point,) if group == "mm_kit.datasets" else ()),
    )

    assert registry.get_dataset("gait120").module == "mm_kit.workflows.gait"


def test_cli_lists_datasets_without_loading_implementations(capsys):
    assert cli_main(["evaluate", "--list-datasets"]) == 0
    output = capsys.readouterr().out
    assert "gait120: gait" in output
    assert "ultra-mocap: bimanual" in output


def test_cli_rejects_an_unknown_command(capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli_main(["analyse"])

    assert exc_info.value.code == 2
    assert "unknown command" in capsys.readouterr().err
