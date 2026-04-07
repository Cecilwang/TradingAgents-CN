from tradingagents.utils.stock_utils import StockMarket, StockUtils


def test_identify_a_share_full_symbols() -> None:
    assert StockUtils.identify_stock_market("000001.SZ") == StockMarket.CHINA_A
    assert StockUtils.identify_stock_market("600036.SH") == StockMarket.CHINA_A
    assert StockUtils.identify_stock_market("430001.BJ") == StockMarket.CHINA_A


def test_get_market_info_for_a_share_full_symbol() -> None:
    market_info = StockUtils.get_market_info("300750.SZ")

    assert market_info["market"] == "china_a"
    assert market_info["market_name"] == "中国A股"
    assert market_info["data_source"] == "china_unified"
    assert market_info["is_china"] is True
