from app.models.config import LLMConfig, ModelCatalog, ModelInfo
from app.services.config_service import ConfigService


def test_apply_model_catalog_pricing_to_llm_configs_updates_matching_models():
    service = ConfigService()
    llm_configs = [
        LLMConfig(
            provider="codex",
            model_name="codex-gpt-5.4-mini",
            input_price_per_1k=None,
            cached_input_price_per_1k=None,
            output_price_per_1k=None,
            currency="CNY",
        ),
        LLMConfig(
            provider="codex",
            model_name="codex-gpt-5.4",
            input_price_per_1k=None,
            cached_input_price_per_1k=None,
            output_price_per_1k=None,
            currency="CNY",
        ),
        LLMConfig(
            provider="dashscope",
            model_name="qwen-plus",
            input_price_per_1k=0.1,
            cached_input_price_per_1k=0.1,
            output_price_per_1k=0.2,
            currency="CNY",
        ),
    ]
    catalog = ModelCatalog(
        provider="codex",
        provider_name="Codex CLI",
        models=[
            ModelInfo(
                name="codex-gpt-5.4-mini",
                display_name="codex-gpt-5.4-mini",
                input_price_per_1k=0.00075,
                cached_input_price_per_1k=0.000075,
                output_price_per_1k=0.0045,
                currency="USD",
            ),
            ModelInfo(
                name="codex-gpt-5.4",
                display_name="codex-gpt-5.4",
                input_price_per_1k=0.0025,
                cached_input_price_per_1k=0.00025,
                output_price_per_1k=0.015,
                currency="USD",
            ),
        ],
    )

    updated_count = service._apply_model_catalog_pricing_to_llm_configs(
        llm_configs,
        catalog,
    )

    assert updated_count == 2
    assert llm_configs[0].input_price_per_1k == 0.00075
    assert llm_configs[0].cached_input_price_per_1k == 0.000075
    assert llm_configs[0].output_price_per_1k == 0.0045
    assert llm_configs[0].currency == "USD"
    assert llm_configs[1].input_price_per_1k == 0.0025
    assert llm_configs[1].cached_input_price_per_1k == 0.00025
    assert llm_configs[1].output_price_per_1k == 0.015
    assert llm_configs[1].currency == "USD"
    assert llm_configs[2].input_price_per_1k == 0.1
    assert llm_configs[2].cached_input_price_per_1k == 0.1
    assert llm_configs[2].output_price_per_1k == 0.2
    assert llm_configs[2].currency == "CNY"
