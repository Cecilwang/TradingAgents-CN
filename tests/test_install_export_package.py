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
        if config.get("provider") == "codex" and config.get("model_name") == "codex-gpt-5.4"
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

    codex_llm = codex_llms[0]
    codex_catalog_models = codex_catalogs[0]["models"]
    codex_catalog_model = next(
        model for model in codex_catalog_models if model["name"] == "codex-gpt-5.4"
    )

    assert codex_llm["input_price_per_1k"] == 0.0025
    assert codex_llm["output_price_per_1k"] == 0.015
    assert codex_llm["currency"] == "USD"
    assert codex_catalog_model["input_price_per_1k"] == 0.0025
    assert codex_catalog_model["output_price_per_1k"] == 0.015
    assert codex_catalog_model["currency"] == "USD"
