from app.utils.report_helpers import (
    extract_report_action,
    extract_report_target_price,
)


def test_extract_report_action_prefers_decision_action():
    report = {
        "decision": {"action": "BUY"},
        "recommendation": "投资建议：卖出。",
    }

    assert extract_report_action(report) == "买入"


def test_extract_report_action_returns_empty_without_structured_decision():
    report = {
        "recommendation": "投资建议：持有。决策依据：估值合理。"
    }

    assert extract_report_action(report) == ""


def test_extract_report_target_price_prefers_decision_target_price():
    report = {
        "decision": {"target_price": "¥128.50"},
        "recommendation": "目标价格：120元。"
    }

    assert extract_report_target_price(report) == 128.5


def test_extract_report_target_price_returns_none_without_structured_decision():
    report = {
        "reports": {
            "final_trade_decision": "最终建议：买入\n目标价：$245.80\n理由：增长稳健"
        }
    }

    assert extract_report_target_price(report) is None
