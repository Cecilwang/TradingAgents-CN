import json
from pathlib import Path


def test_install_export_package_contains_codex_seed():
    package_path = Path("install/database_export_config_2025-11-13.json")
    export_data = json.loads(package_path.read_text(encoding="utf-8"))
    collections = export_data["data"]

    active_config = next(
        doc for doc in collections["system_configs"] if doc.get("is_active")
    )
    codex_llms = [
        config
        for config in active_config.get("llm_configs", [])
        if config.get("provider") == "codex" and config.get("model_name") == "gpt-5.4"
    ]
    codex_providers = [
        provider
        for provider in collections["llm_providers"]
        if provider.get("name") == "codex"
    ]
    codex_catalogs = [
        catalog
        for catalog in collections["model_catalog"]
        if catalog.get("provider") == "codex"
    ]

    assert len(codex_llms) == 1
    assert len(codex_providers) == 1
    assert len(codex_catalogs) == 1
