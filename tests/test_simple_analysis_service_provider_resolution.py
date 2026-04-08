import asyncio

from app.models.config import LLMConfig, SystemConfig, CODEX_DEEP_MODEL_NAME
from app.services import simple_analysis_service


def test_get_provider_by_model_name_prefers_default_provider(monkeypatch):
    async def run_test():
        system_config = SystemConfig(
            config_name="test",
            config_type="system",
            system_settings={"default_provider": "codex"},
            llm_configs=[
                LLMConfig(provider="openai", model_name="gpt-4o"),
                LLMConfig(provider="codex", model_name=CODEX_DEEP_MODEL_NAME),
            ],
        )

        async def mock_get_system_config():
            return system_config

        monkeypatch.setattr(
            simple_analysis_service.config_service,
            "get_system_config",
            mock_get_system_config,
        )

        provider = await simple_analysis_service.get_provider_by_model_name(CODEX_DEEP_MODEL_NAME)
        assert provider == "codex"

    asyncio.run(run_test())


class _FakeProvidersCollection:
    def find_one(self, query):
        provider = query["name"]
        if provider == "codex":
            return {"name": "codex", "default_base_url": "local://codex-cli"}
        if provider == "openai":
            return {"name": "openai", "default_base_url": "https://api.openai.com/v1"}
        return None


class _FakeSystemConfigsCollection:
    def find_one(self, *_args, **_kwargs):
        return {
            "is_active": True,
            "system_settings": {"default_provider": "codex"},
            "llm_configs": [
                {"provider": "openai", "model_name": "gpt-4o", "api_base": "https://api.openai.com/v1"},
                {"provider": "codex", "model_name": CODEX_DEEP_MODEL_NAME, "api_base": "local://codex-cli"},
            ],
        }


class _FakeMongoDb:
    def __init__(self):
        self.system_configs = _FakeSystemConfigsCollection()
        self.llm_providers = _FakeProvidersCollection()

    def __getitem__(self, _name):
        return self


class _FakeMongoClient:
    def __init__(self, *_args, **_kwargs):
        self._db = _FakeMongoDb()

    def __getitem__(self, _name):
        return self._db

    def close(self):
        return None


def test_get_provider_and_url_by_model_sync_prefers_default_provider(monkeypatch):
    monkeypatch.setattr("pymongo.MongoClient", _FakeMongoClient)
    monkeypatch.setattr(
        simple_analysis_service,
        "_get_env_api_key_for_provider",
        lambda _provider: None,
    )

    provider_info = simple_analysis_service.get_provider_and_url_by_model_sync(CODEX_DEEP_MODEL_NAME)

    assert provider_info["provider"] == "codex"
    assert provider_info["backend_url"] == "local://codex-cli"
