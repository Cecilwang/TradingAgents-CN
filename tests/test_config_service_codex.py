import asyncio

from app.models.config import LLMConfig, CODEX_DEEP_MODEL_NAME
from app.services.config_service import ConfigService


def test_test_llm_config_uses_local_codex_cli_path(monkeypatch):
    """Codex 模型测试不应要求 API Key 或 API Base。"""

    async def run_test():
        service = ConfigService()

        async def mock_get_db():
            raise AssertionError("Codex 模型测试不应访问数据库")

        monkeypatch.setattr(service, "_get_db", mock_get_db)
        monkeypatch.setattr(
            service,
            "_test_codex_cli",
            lambda display_name: {
                "success": True,
                "message": f"{display_name} 本地 CLI 可用",
                "data": {"version": "codex 0.117.0", "source": "local_cli"},
            },
        )

        result = await service.test_llm_config(
            LLMConfig(
                provider="codex",
                model_name=CODEX_DEEP_MODEL_NAME,
                api_key="",
                api_base="",
            )
        )

        assert result["success"] is True
        assert "本地 CLI 可用" in result["message"]
        assert "response_time" in result

    asyncio.run(run_test())


def test_get_llm_providers_sets_codex_active_from_local_cli(monkeypatch):
    class FakeCursor:
        async def to_list(self, length=None):
            return [
                {
                    "name": "codex",
                    "display_name": "Codex CLI",
                    "default_base_url": "local://codex-cli",
                    "is_active": False,
                    "supported_features": ["chat"],
                    "extra_config": {},
                }
            ]

    class FakeCollection:
        def find(self):
            return FakeCursor()

    class FakeDb:
        llm_providers = FakeCollection()

    async def run_test():
        service = ConfigService()

        async def mock_get_db():
            return FakeDb()

        monkeypatch.setattr(service, "_get_db", mock_get_db)
        monkeypatch.setattr(
            "app.services.config_service.get_codex_cli_status",
            lambda: {"available": True, "version": "codex 0.117.0"},
        )

        providers = await service.get_llm_providers()
        codex = next(provider for provider in providers if provider.name == "codex")

        assert codex.is_active is True
        assert codex.extra_config["source"] == "local_cli"
        assert codex.extra_config["codex_version"] == "codex 0.117.0"

    asyncio.run(run_test())
