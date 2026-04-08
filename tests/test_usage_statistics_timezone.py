from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from tradingagents.config.config_manager import ConfigManager
from tradingagents.config.usage_models import UsageRecord


def test_get_usage_statistics_handles_aware_and_naive_timestamps(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("USE_MONGODB_STORAGE", "false")

    config_manager = ConfigManager(str(tmp_path))
    timezone = ZoneInfo("Asia/Shanghai")
    now = datetime.now(timezone)

    config_manager.save_usage_records(
        [
            UsageRecord(
                timestamp=(now - timedelta(hours=1)).isoformat(),
                provider="codex",
                model_name="gpt-5.4",
                input_tokens=12,
                output_tokens=6,
                cost=0.0,
            ),
            UsageRecord(
                timestamp=(now - timedelta(days=3)).replace(tzinfo=None).isoformat(),
                provider="openai",
                model_name="gpt-4",
                input_tokens=8,
                output_tokens=4,
                cost=1.0,
            ),
        ]
    )

    stats = config_manager.get_usage_statistics(1)

    assert stats["total_requests"] == 1
    assert stats["total_input_tokens"] == 12
    assert stats["total_output_tokens"] == 6
    assert stats["provider_stats"] == {
        "codex": {
            "cost": 0.0,
            "input_tokens": 12,
            "cached_input_tokens": 0,
            "output_tokens": 6,
            "requests": 1,
        }
    }
