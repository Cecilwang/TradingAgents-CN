import pandas as pd

from tradingagents.dataflows.data_source_manager import DataSourceManager


def test_get_akshare_stock_info_normalizes_ts_code(monkeypatch):
    manager = DataSourceManager.__new__(DataSourceManager)
    called = {}

    class FakeAkshare:
        @staticmethod
        def stock_individual_info_em(symbol):
            called["symbol"] = symbol
            return pd.DataFrame(
                [
                    {"item": "股票简称", "value": "九丰能源"},
                ]
            )

    monkeypatch.setitem(__import__("sys").modules, "akshare", FakeAkshare)

    result = manager._get_akshare_stock_info("600989.SH")

    assert called["symbol"] == "sh600989"
    assert result["name"] == "九丰能源"
    assert result["source"] == "akshare"


def test_get_baostock_stock_info_normalizes_ts_code(monkeypatch):
    manager = DataSourceManager.__new__(DataSourceManager)
    called = {"logout": 0}

    class FakeLoginResult:
        error_code = "0"
        error_msg = ""

    class FakeQueryResult:
        error_code = "0"
        error_msg = ""

        def __init__(self):
            self._rows = [["sh.600989", "九丰能源", "2021-05-25", "", "1", "1"]]
            self._index = -1

        def next(self):
            self._index += 1
            return self._index < len(self._rows)

        def get_row_data(self):
            return self._rows[self._index]

    class FakeBaoStock:
        @staticmethod
        def login():
            return FakeLoginResult()

        @staticmethod
        def logout():
            called["logout"] += 1

        @staticmethod
        def query_stock_basic(code):
            called["code"] = code
            return FakeQueryResult()

    monkeypatch.setitem(__import__("sys").modules, "baostock", FakeBaoStock)

    result = manager._get_baostock_stock_info("600989.SH")

    assert called["code"] == "sh.600989"
    assert called["logout"] == 1
    assert result["name"] == "九丰能源"
    assert result["list_date"] == "2021-05-25"
    assert result["source"] == "baostock"


def test_split_cn_stock_symbol_supports_multiple_input_formats():
    manager = DataSourceManager.__new__(DataSourceManager)

    assert manager._split_cn_stock_symbol("600989.SH") == ("600989", "sh")
    assert manager._split_cn_stock_symbol("sh.600989") == ("600989", "sh")
    assert manager._split_cn_stock_symbol("sz000001") == ("000001", "sz")
