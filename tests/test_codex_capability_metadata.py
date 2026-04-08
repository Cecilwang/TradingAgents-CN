from app.core.unified_config import UnifiedConfigManager
from app.models.config import LLMConfig, CODEX_DEEP_MODEL_NAME
from app.services.model_capability_service import ModelCapabilityService
from app.constants.model_capabilities import ModelFeature


class _FakeCollection:
    """最小 Mongo 集合桩。"""

    def __init__(self, document):
        self._document = document

    def find_one(self, *_args, **_kwargs):
        return self._document


class _FakeMongoDb:
    """最小 Mongo 数据库桩。"""

    def __init__(self, document):
        self.system_configs = _FakeCollection(document)


class _FakeMongoClient:
    """最小 MongoClient 桩。"""

    def __init__(self, *_args, **_kwargs):
        self._db = _FakeMongoDb(
            {
                "is_active": True,
                "llm_configs": [
                    {
                        "provider": "codex",
                        "model_name": CODEX_DEEP_MODEL_NAME,
                        "capability_level": 5,
                        "suitable_roles": ["both"],
                        "features": ["tool_calling", "reasoning", "long_context"],
                        "recommended_depths": ["标准", "深度", "全面"],
                        "performance_metrics": {"speed": 3, "cost": 2, "quality": 5},
                    }
                ],
            }
        )

    def __getitem__(self, _name):
        return self._db

    def close(self):
        return None


def test_unified_config_preserves_codex_capability_metadata(tmp_path):
    """文件回退配置需要完整保留 Codex 的能力元数据。"""
    manager = UnifiedConfigManager()
    manager.paths.models_json = tmp_path / "models.json"

    llm_config = LLMConfig(
        provider="codex",
        model_name=CODEX_DEEP_MODEL_NAME,
        model_display_name=CODEX_DEEP_MODEL_NAME,
        api_base="",
        enabled=True,
        capability_level=5,
        suitable_roles=["both"],
        features=["tool_calling", "reasoning", "long_context"],
        recommended_depths=["标准", "深度", "全面"],
        performance_metrics={"speed": 3, "cost": 2, "quality": 5},
    )

    assert manager.save_llm_config(llm_config) is True

    saved_config = manager.get_llm_configs()[0]
    assert saved_config.provider == "codex"
    assert saved_config.model_name == CODEX_DEEP_MODEL_NAME
    assert saved_config.capability_level == 5
    assert saved_config.suitable_roles == ["both"]
    assert saved_config.features == ["tool_calling", "reasoning", "long_context"]
    assert saved_config.recommended_depths == ["标准", "深度", "全面"]
    assert saved_config.performance_metrics == {"speed": 3, "cost": 2, "quality": 5}


def test_model_capability_service_reads_codex_metadata_from_db(monkeypatch):
    """能力服务应直接读取已保存的 Codex 能力元数据。"""
    monkeypatch.setattr("pymongo.MongoClient", _FakeMongoClient)

    service = ModelCapabilityService()
    config = service.get_model_config(CODEX_DEEP_MODEL_NAME)

    assert config["capability_level"] == 5
    assert ModelFeature.TOOL_CALLING in config["features"]
    assert ModelFeature.REASONING in config["features"]


def test_validate_model_pair_accepts_saved_codex_capability_metadata(monkeypatch):
    """能力校验应接受当前保存的 Codex 配置，不再误切回其他模型。"""
    monkeypatch.setattr("pymongo.MongoClient", _FakeMongoClient)

    service = ModelCapabilityService()
    validation = service.validate_model_pair(CODEX_DEEP_MODEL_NAME, CODEX_DEEP_MODEL_NAME, "全面")

    assert validation["valid"] is True
    assert not any("不支持工具调用" in warning for warning in validation["warnings"])


def test_model_capability_service_maps_codex_variants_to_deep_profile():
    service = ModelCapabilityService()

    config = service.get_model_config("codex-gpt-5.4-medium")

    assert config["capability_level"] == 5
    assert config["_mapped_from"] == CODEX_DEEP_MODEL_NAME
