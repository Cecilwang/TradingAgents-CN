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
    def __init__(self, *, find_one_result=None, find_result=None):
        self._find_one_result = find_one_result
        self._find_result = find_result or []

    async def find_one(self, query):
        return self._find_one_result

    def find(self, query, projection=None):
        return _FakeCursor(self._find_result)


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
                    }
                ],
            }
        )
        self._collections = {
            "stock_basic_info": _FakeCollection(
                find_result=[
                    {"code": "600089", "market": "主板", "sse": "上海证券交易所"}
                ]
            ),
            "market_quotes": _FakeCollection(find_result=[]),
        }

    def __getitem__(self, name):
        return self._collections[name]


class _FakeConfig:
    async def get_data_source_configs_async(self):
        return [SimpleNamespace(type="tushare", enabled=True, priority=1)]


def test_get_user_favorites_skips_online_quote_fallback_by_default(monkeypatch):
    service = FavoritesService()
    service.db = _FakeDB()

    online_quotes = AsyncMock(return_value={"600089": {"close": 12.34, "pct_chg": 1.2}})

    monkeypatch.setattr("app.core.unified_config.UnifiedConfigManager", _FakeConfig)
    monkeypatch.setattr(
        "app.services.favorites_service.get_quotes_service",
        lambda: SimpleNamespace(get_quotes=online_quotes),
    )

    items = asyncio.run(service.get_user_favorites("user-1"))

    assert len(items) == 1
    assert items[0]["stock_code"] == "600089"
    assert items[0]["current_price"] is None
    online_quotes.assert_not_awaited()


def test_get_user_favorites_can_opt_in_online_quote_fallback(monkeypatch):
    service = FavoritesService()
    service.db = _FakeDB()

    online_quotes = AsyncMock(return_value={"600089": {"close": 12.34, "pct_chg": 1.2}})

    monkeypatch.setattr("app.core.unified_config.UnifiedConfigManager", _FakeConfig)
    monkeypatch.setattr(
        "app.services.favorites_service.get_quotes_service",
        lambda: SimpleNamespace(get_quotes=online_quotes),
    )

    items = asyncio.run(
        service.get_user_favorites("user-1", allow_online_quote_fallback=True)
    )

    assert len(items) == 1
    assert items[0]["current_price"] == 12.34
    assert items[0]["change_percent"] == 1.2
    online_quotes.assert_awaited_once_with(["600089"])
