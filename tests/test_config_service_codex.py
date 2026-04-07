import asyncio

from app.models.config import LLMConfig
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
                model_name="gpt-5.4",
                api_key="",
                api_base="",
            )
        )

        assert result["success"] is True
        assert "本地 CLI 可用" in result["message"]
        assert "response_time" in result

    asyncio.run(run_test())
