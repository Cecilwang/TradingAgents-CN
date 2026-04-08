import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.services.favorites_service import FavoritesService


class _FakeCursor:
    def __init__(self, docs):
        self._docs = docs

    async def to_list(self, length=None):
        return list(self._docs)


class _FakeCollection:
    def __init__(self, *, find_one_result=None, docs=None):
        self._find_one_result = find_one_result
        self._docs = list(docs or [])
        self.updated = []

    async def find_one(self, query):
        return self._find_one_result

    def find(self, query, projection=None):
        docs = list(self._docs)

        code_filter = query.get("code")
        if isinstance(code_filter, dict) and "$in" in code_filter:
            allowed = set(code_filter["$in"])
            docs = [doc for doc in docs if doc.get("code") in allowed]

        source = query.get("source")
        if source is not None:
            docs = [doc for doc in docs if doc.get("source") == source]

        return _FakeCursor(docs)

    async def update_one(self, query, update, upsert=False):
        self.updated.append({
            "query": query,
            "update": update,
            "upsert": upsert,
        })


class _FakeDB:
    def __init__(self):
        self.user_favorites = _FakeCollection(
            find_one_result={
                "user_id": "user-1",
                "favorites": [
                    {
                        "stock_code": "600089",
                        "stock_name": "特变电工",
                        "market": "A股",
                        "tags": [],
                        "notes": "",
                    },
                    {
                        "stock_code": "0700",
                        "stock_name": "腾讯控股",
                        "market": "港股",
                        "tags": [],
                        "notes": "",
                    },
                    {
                        "stock_code": "AAPL",
                        "stock_name": "Apple Inc.",
                        "market": "美股",
                        "tags": [],
                        "notes": "",
                    },
                ],
            }
        )
        self._collections = {
            "stock_basic_info": _FakeCollection(
                docs=[{"code": "600089", "source": "tushare", "market": "主板", "sse": "上海证券交易所"}]
            ),
            "market_quotes": _FakeCollection(docs=[]),
            "market_quotes_hk": _FakeCollection(docs=[]),
            "market_quotes_us": _FakeCollection(docs=[]),
        }

    def __getitem__(self, name):
        return self._collections[name]


class _FakeConfigManager:
    async def get_data_source_configs_async(self):
        return [SimpleNamespace(type="tushare", enabled=True, priority=1)]


def test_get_user_favorites_reads_quotes_from_market_specific_mongo(monkeypatch):
    service = FavoritesService()
    service.db = _FakeDB()

    service.db["market_quotes"]._docs = [
        {"code": "600089", "close": 12.34, "pct_chg": 1.2, "volume": 1000}
    ]
    service.db["market_quotes_hk"]._docs = [
        {"code": "00700", "close": 320.5, "pct_chg": 2.3, "volume": 2000}
    ]
    service.db["market_quotes_us"]._docs = [
        {"code": "AAPL", "close": 180.1, "pct_chg": -0.5, "volume": 3000}
    ]

    online_quotes = AsyncMock(return_value={})

    monkeypatch.setattr("app.core.unified_config.UnifiedConfigManager", _FakeConfigManager)
    monkeypatch.setattr(
        "app.services.favorites_service.get_quotes_service",
        lambda: SimpleNamespace(get_quotes=online_quotes),
    )

    items = asyncio.run(service.get_user_favorites("user-1", allow_online_quote_fallback=False))

    assert [item["current_price"] for item in items] == [12.34, 320.5, 180.1]
    assert [item["change_percent"] for item in items] == [1.2, 2.3, -0.5]
    online_quotes.assert_not_awaited()


def test_get_user_favorites_fetches_missing_quotes_without_mongo_writeback(monkeypatch):
    service = FavoritesService()
    service.db = _FakeDB()

    online_quotes = AsyncMock(return_value={
        "600089": {"close": 12.34, "pct_chg": 1.2, "volume": 1000}
    })
    foreign_calls = []

    class _FakeForeignStockService:
        def __init__(self, db=None):
            self.db = db

        async def get_quote(self, market, code, force_refresh=False):
            foreign_calls.append((market, code, force_refresh))
            payloads = {
                ("HK", "00700"): {"price": 320.5, "change_percent": 2.3, "volume": 2000},
                ("US", "AAPL"): {"price": 180.1, "change_percent": -0.5, "volume": 3000},
            }
            return payloads.get((market, code))

    monkeypatch.setattr("app.core.unified_config.UnifiedConfigManager", _FakeConfigManager)
    monkeypatch.setattr(
        "app.services.favorites_service.get_quotes_service",
        lambda: SimpleNamespace(get_quotes=online_quotes),
    )
    monkeypatch.setattr(
        "app.services.foreign_stock_service.ForeignStockService",
        _FakeForeignStockService,
    )

    items = asyncio.run(service.get_user_favorites("user-1", allow_online_quote_fallback=True))

    assert [item["current_price"] for item in items] == [12.34, 320.5, 180.1]
    assert [item["change_percent"] for item in items] == [1.2, 2.3, -0.5]
    online_quotes.assert_awaited_once_with(["600089"])
    assert foreign_calls == [("HK", "00700", False), ("US", "AAPL", False)]
    assert service.db["market_quotes"].updated == []
    assert service.db["market_quotes_hk"].updated == []
    assert service.db["market_quotes_us"].updated == []
