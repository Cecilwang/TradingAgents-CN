import os
import sys
from pathlib import Path

import pytest

# 该测试只验证路由层清洗逻辑，不需要在导入阶段初始化 MongoDB 存储。
os.environ["USE_MONGODB_STORAGE"] = "false"

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.models.config import LLMProviderRequest  # noqa: E402
from app.models.operation_log import ActionType  # noqa: E402
from app.models.user import User  # noqa: E402
from app.routers import config as config_router  # noqa: E402
from app.services.config_service import config_service  # noqa: E402


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_add_llm_provider_sanitizes_api_key(monkeypatch):
    captured = {}

    async def mock_add_llm_provider(provider):
        captured["api_key"] = provider.api_key
        return "mock-id-123"

    async def mock_log_operation(**kwargs):
        captured["action"] = kwargs["action"]
        captured["action_type"] = kwargs["action_type"]
        return None

    monkeypatch.setattr(config_service, "add_llm_provider", mock_add_llm_provider)
    monkeypatch.setattr(config_router, "log_operation", mock_log_operation)

    payload = LLMProviderRequest(
        name="openai",
        display_name="OpenAI",
        description="desc",
        website="https://openai.com",
        api_doc_url=None,
        logo_url=None,
        is_active=True,
        supported_features=[],
        default_base_url=None,
        api_key="SHOULD_BE_STRIPPED",
        api_secret=None,
        extra_config={},
    )

    result = await config_router.add_llm_provider(
        request=payload,
        current_user=User(
            username="tester", email="t@example.com", hashed_password="x"
        ),
    )

    assert result["success"] is True
    assert result["data"]["id"] == "mock-id-123"
    assert captured["api_key"] == ""
    assert captured["action"] == "add_llm_provider"
    assert captured["action_type"] == ActionType.CONFIG_MANAGEMENT


@pytest.mark.anyio
async def test_update_llm_provider_sanitizes_api_key(monkeypatch):
    captured = {}

    async def mock_update_llm_provider(provider_id, update_data):
        captured["provider_id"] = provider_id
        captured["api_key"] = update_data.get("api_key")
        return True

    async def mock_log_operation(**kwargs):
        captured["action"] = kwargs["action"]
        captured["action_type"] = kwargs["action_type"]
        return None

    monkeypatch.setattr(config_service, "update_llm_provider", mock_update_llm_provider)
    monkeypatch.setattr(config_router, "log_operation", mock_log_operation)

    payload = LLMProviderRequest(
        name="openai",
        display_name="OpenAI",
        description="desc",
        website="https://openai.com",
        api_doc_url=None,
        logo_url=None,
        is_active=True,
        supported_features=[],
        default_base_url=None,
        api_key="your_api_key_here",
        api_secret=None,
        extra_config={"k": "v"},
    )

    result = await config_router.update_llm_provider(
        provider_id="abc123",
        request=payload,
        current_user=User(
            username="tester", email="t@example.com", hashed_password="x"
        ),
    )

    assert result["success"] is True
    assert captured["provider_id"] == "abc123"
    assert captured["api_key"] is None
    assert captured["action"] == "update_llm_provider"
    assert captured["action_type"] == ActionType.CONFIG_MANAGEMENT
