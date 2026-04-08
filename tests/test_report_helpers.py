from app.utils.report_helpers import (
    extract_report_action,
)


def test_extract_report_action_prefers_decision_action():
    report = {
        "decision": {"action": "BUY"},
        "recommendation": "投资建议：卖出。",
    }

    assert extract_report_action(report) == "买入"


def test_extract_report_action_falls_back_to_recommendation_text():
    report = {
        "recommendation": "投资建议：持有。决策依据：估值合理。"
    }

    assert extract_report_action(report) == "持有"
