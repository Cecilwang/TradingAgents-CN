import asyncio

from app.models.config import LLMConfig, SystemConfig, CODEX_DEEP_MODEL_NAME
from app.services.config_service import ConfigService


def test_delete_llm_config_accepts_string_provider(monkeypatch):
    """删除逻辑应兼容数据库里 provider 为字符串的旧配置。"""

    async def run_test():
        service = ConfigService()
        config = SystemConfig(
            config_name="test",
            config_type="system",
            llm_configs=[
                LLMConfig(provider="codex", model_name=CODEX_DEEP_MODEL_NAME),
                LLMConfig(provider="dashscope", model_name="qwen-plus"),
            ],
        )
        saved_configs = []

        async def mock_get_system_config():
            return config

        async def mock_save_system_config(updated_config):
            saved_configs.append(updated_config)
            return True

        monkeypatch.setattr(service, "get_system_config", mock_get_system_config)
        monkeypatch.setattr(service, "save_system_config", mock_save_system_config)

        result = await service.delete_llm_config("codex", CODEX_DEEP_MODEL_NAME)

        assert result is True
        assert len(config.llm_configs) == 1
        assert config.llm_configs[0].provider == "dashscope"
        assert config.llm_configs[0].model_name == "qwen-plus"
        assert len(saved_configs) == 1

    asyncio.run(run_test())


def test_delete_llm_config_returns_false_when_model_missing(monkeypatch):
    """删除不存在的配置时不应报错，也不应触发保存。"""

    async def run_test():
        service = ConfigService()
        config = SystemConfig(
            config_name="test",
            config_type="system",
            llm_configs=[LLMConfig(provider="codex", model_name=CODEX_DEEP_MODEL_NAME)],
        )
        save_called = False

        async def mock_get_system_config():
            return config

        async def mock_save_system_config(_updated_config):
            nonlocal save_called
            save_called = True
            return True

        monkeypatch.setattr(service, "get_system_config", mock_get_system_config)
        monkeypatch.setattr(service, "save_system_config", mock_save_system_config)

        result = await service.delete_llm_config("codex", "missing-model")

        assert result is False
        assert len(config.llm_configs) == 1
        assert save_called is False

    asyncio.run(run_test())
