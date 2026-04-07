from app.scripts.init_providers import build_providers_data


def test_build_providers_data_disables_codex_when_cli_missing(monkeypatch):
    monkeypatch.setattr(
        "app.scripts.init_providers.get_codex_cli_status",
        lambda: {"available": False, "version": None},
    )

    providers = build_providers_data()
    codex_provider = next(
        provider for provider in providers if provider["name"] == "codex"
    )

    assert codex_provider["is_active"] is False
    assert codex_provider["extra_config"]["source"] == "local_cli"
    assert codex_provider["extra_config"]["codex_version"] is None
