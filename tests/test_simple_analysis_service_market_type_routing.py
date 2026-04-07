import asyncio

from app.models.analysis import AnalysisParameters, SingleAnalysisRequest
from app.services import simple_analysis_service


class _FakeTaskState:
    def __init__(self, task_id: str):
        self.task_id = task_id


class _FakeMemoryManager:
    def __init__(self):
        self.create_task_calls = []

    async def create_task(self, **kwargs):
        self.create_task_calls.append(kwargs)
        return _FakeTaskState(kwargs["task_id"])

    async def get_task(self, task_id: str):
        return _FakeTaskState(task_id)


class _FakeUpdateResult:
    matched_count = 1
    upserted_id = None


class _FakeAnalysisTasksCollection:
    def __init__(self):
        self.calls = []

    async def update_one(self, query, update, upsert=False):
        self.calls.append({
            "query": query,
            "update": update,
            "upsert": upsert,
        })
        return _FakeUpdateResult()


class _FakeMongoDb:
    def __init__(self):
        self.analysis_tasks = _FakeAnalysisTasksCollection()


def test_resolve_market_type_for_symbol_prefers_symbol_over_requested_value():
    assert simple_analysis_service._resolve_market_type_for_symbol("00700.HK", "A股") == "港股"
    assert simple_analysis_service._resolve_market_type_for_symbol("AAPL", "A股") == "美股"
    assert simple_analysis_service._resolve_market_type_for_symbol("600036.SH", "港股") == "A股"


def test_create_analysis_task_overrides_default_market_type_for_hk_symbol(monkeypatch):
    async def run_test():
        fake_db = _FakeMongoDb()
        fake_memory_manager = _FakeMemoryManager()

        service = object.__new__(simple_analysis_service.SimpleAnalysisService)
        service.memory_manager = fake_memory_manager
        service._stock_name_cache = {}
        service._resolve_stock_name = lambda code: f"名称-{code}"

        monkeypatch.setattr(simple_analysis_service, "get_mongo_db", lambda: fake_db)

        request = SingleAnalysisRequest(
            symbol="00700.HK",
            stock_code="00700.HK",
            parameters=AnalysisParameters(),
        )

        result = await service.create_analysis_task("user-1", request)

        assert result["status"] == "pending"
        assert fake_memory_manager.create_task_calls[0]["parameters"]["market_type"] == "港股"
        assert fake_db.analysis_tasks.calls[0]["update"]["$setOnInsert"]["market_type"] == "港股"

    asyncio.run(run_test())


def test_create_analysis_task_overrides_default_market_type_for_us_symbol(monkeypatch):
    async def run_test():
        fake_db = _FakeMongoDb()
        fake_memory_manager = _FakeMemoryManager()

        service = object.__new__(simple_analysis_service.SimpleAnalysisService)
        service.memory_manager = fake_memory_manager
        service._stock_name_cache = {}
        service._resolve_stock_name = lambda code: f"名称-{code}"

        monkeypatch.setattr(simple_analysis_service, "get_mongo_db", lambda: fake_db)

        request = SingleAnalysisRequest(
            symbol="AAPL",
            stock_code="AAPL",
            parameters=AnalysisParameters(),
        )

        result = await service.create_analysis_task("user-1", request)

        assert result["status"] == "pending"
        assert fake_memory_manager.create_task_calls[0]["parameters"]["market_type"] == "美股"
        assert fake_db.analysis_tasks.calls[0]["update"]["$setOnInsert"]["market_type"] == "美股"

    asyncio.run(run_test())
