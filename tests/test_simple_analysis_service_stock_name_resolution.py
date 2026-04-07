from typing import Any, Dict, Optional

from app.services import simple_analysis_service


class _FakeCollection:
    def __init__(self, document: Optional[Dict[str, Any]]):
        self.document = document
        self.calls = []

    def find_one(self, query, sort=None):
        self.calls.append({"query": query, "sort": sort})
        return self.document


class _FakeMongoDb:
    def __init__(self, **collections):
        self.collections = collections

    def __getitem__(self, name: str):
        return self.collections[name]


def test_resolve_stock_name_by_market_reads_hk_cache(monkeypatch):
    fake_hk_collection = _FakeCollection({"code": "00700", "name": "腾讯控股"})
    fake_db = _FakeMongoDb(stock_basic_info_hk=fake_hk_collection)

    monkeypatch.setattr(simple_analysis_service, "get_mongo_db_sync", lambda: fake_db)

    result = simple_analysis_service._resolve_stock_name_by_market("00700.HK")

    assert result == "腾讯控股"
    assert fake_hk_collection.calls == [{
        "query": {"$or": [{"code": "00700"}, {"symbol": "00700.HK"}]},
        "sort": [("updated_at", -1)],
    }]


def test_resolve_stock_name_by_market_reads_us_cache(monkeypatch):
    fake_us_collection = _FakeCollection({"code": "AAPL", "name": "苹果公司"})
    fake_db = _FakeMongoDb(stock_basic_info_us=fake_us_collection)

    monkeypatch.setattr(simple_analysis_service, "get_mongo_db_sync", lambda: fake_db)

    result = simple_analysis_service._resolve_stock_name_by_market("AAPL")

    assert result == "苹果公司"
    assert fake_us_collection.calls == [{
        "query": {"$or": [{"code": "AAPL"}, {"symbol": "AAPL"}]},
        "sort": [("updated_at", -1)],
    }]


def test_enrich_stock_names_rewrites_non_a_share_names(monkeypatch):
    service = object.__new__(simple_analysis_service.SimpleAnalysisService)
    service._stock_name_cache = {}
    service._resolve_stock_name = lambda code: {
        "00700.HK": "腾讯控股",
        "AAPL": "苹果公司",
    }[code]

    tasks = [
        {"stock_code": "00700.HK", "stock_name": "锦江在线"},
        {"stock_code": "AAPL", "stock_name": "美股AAPL"},
        {"stock_code": "600036.SH", "stock_name": "招商银行"},
    ]

    enriched = service._enrich_stock_names(tasks)

    assert enriched[0]["stock_name"] == "腾讯控股"
    assert enriched[1]["stock_name"] == "苹果公司"
    assert enriched[2]["stock_name"] == "招商银行"
