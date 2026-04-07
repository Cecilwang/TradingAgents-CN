from tradingagents.agents.utils import memory as memory_mod


class _FakeCollection:
    """最小集合替身，避免测试触发真实 ChromaDB 初始化。"""

    def count(self):
        return 0

    def add(self, **kwargs):
        return None

    def query(self, **kwargs):
        return {"documents": [[]], "distances": [[]], "ids": [[]]}


class _FakeChromaDBManager:
    """最小 ChromaDB 管理器替身。"""

    def get_or_create_collection(self, name: str):
        return _FakeCollection()


def test_codex_memory_is_disabled_without_default_embedding(monkeypatch):
    """Codex 当前没有默认 embedding 配置时，应直接禁用记忆向量化。"""
    monkeypatch.setattr(memory_mod, "ChromaDBManager", _FakeChromaDBManager)

    memory = memory_mod.FinancialSituationMemory(
        "test_codex_memory_disabled",
        {"llm_provider": "codex", "backend_url": "local://codex-cli"},
    )

    assert memory.client == "DISABLED"
    assert memory.get_embedding("示例文本") == [0.0] * 1024


def test_memory_keeps_openai_embedding_when_capability_exists(monkeypatch):
    """声明了 embedding 能力的厂家仍应走原有 embedding 初始化。"""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setattr(memory_mod, "ChromaDBManager", _FakeChromaDBManager)

    memory = memory_mod.FinancialSituationMemory(
        "test_openai_memory_enabled",
        {"llm_provider": "openai", "backend_url": "https://api.openai.com/v1"},
    )

    assert memory.client != "DISABLED"
    assert memory.embedding == "text-embedding-3-small"
