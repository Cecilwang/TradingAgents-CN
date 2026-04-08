import asyncio

from app.services.config_service import ConfigService
from tradingagents.config.config_manager import ConfigManager
from tradingagents.config.usage_models import PricingConfig


def test_config_manager_calculate_cost_normalizes_codex_exec_model_names(tmp_path):
    manager = ConfigManager(config_dir=str(tmp_path))
    manager.load_pricing = lambda: [  # type: ignore[method-assign]
        PricingConfig("codex", "codex-gpt-5.4", 0.0025, 0.015, "USD"),
        PricingConfig("codex", "codex-gpt-5.4-mini", 0.00075, 0.0045, "USD"),
    ]

    cost, currency = manager.calculate_cost("codex", "gpt-5.4", 1000, 1000)

    assert cost == 0.0175
    assert currency == "USD"


def test_config_manager_calculate_cost_normalizes_codex_reasoning_variants(tmp_path):
    manager = ConfigManager(config_dir=str(tmp_path))
    manager.load_pricing = lambda: [  # type: ignore[method-assign]
        PricingConfig("codex", "codex-gpt-5.4", 0.0025, 0.015, "USD"),
        PricingConfig("codex", "codex-gpt-5.4-mini", 0.00075, 0.0045, "USD"),
    ]

    cost, currency = manager.calculate_cost("codex", "codex-gpt-5.4-medium", 1000, 1000)

    assert cost == 0.0175
    assert currency == "USD"


def test_config_manager_calculate_cost_normalizes_codex_mini_variants(tmp_path):
    manager = ConfigManager(config_dir=str(tmp_path))
    manager.load_pricing = lambda: [  # type: ignore[method-assign]
        PricingConfig("codex", "codex-gpt-5.4", 0.0025, 0.015, "USD"),
        PricingConfig("codex", "codex-gpt-5.4-mini", 0.00075, 0.0045, "USD"),
    ]

    cost, currency = manager.calculate_cost("codex", "codex-gpt-5.4-mini-preview", 1000, 1000)

    assert cost == 0.00525
    assert currency == "USD"


def test_default_config_contains_codex_usd_pricing():
    service = ConfigService()
    default_config = asyncio.run(service._create_default_config())

    codex_models = {
        llm.model_name: llm
        for llm in default_config.llm_configs
        if getattr(llm.provider, "value", llm.provider) == "codex"
    }

    assert codex_models["codex-gpt-5.4-mini"].input_price_per_1k == 0.00075
    assert codex_models["codex-gpt-5.4-mini"].output_price_per_1k == 0.0045
    assert codex_models["codex-gpt-5.4-mini"].currency == "USD"
    assert codex_models["codex-gpt-5.4"].input_price_per_1k == 0.0025
    assert codex_models["codex-gpt-5.4"].output_price_per_1k == 0.015
    assert codex_models["codex-gpt-5.4"].currency == "USD"
