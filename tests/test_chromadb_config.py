from tradingagents.agents.utils import chromadb_config


def test_create_ephemeral_client_prefers_explicit_api(monkeypatch):
    called = {}

    class FakeChroma:
        @staticmethod
        def EphemeralClient(settings=None):
            called["ephemeral"] = True
            called["settings"] = settings
            return "ephemeral-client"

    monkeypatch.setattr(chromadb_config, "chromadb", FakeChroma)

    client = chromadb_config._create_ephemeral_client()

    assert client == "ephemeral-client"
    assert called["ephemeral"] is True
    assert called["settings"] is not None


def test_non_windows_uses_ephemeral_client(monkeypatch):
    monkeypatch.setattr(chromadb_config.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        chromadb_config,
        "_create_ephemeral_client",
        lambda: "linux-ephemeral-client",
    )

    client = chromadb_config.get_optimal_chromadb_client()

    assert client == "linux-ephemeral-client"
