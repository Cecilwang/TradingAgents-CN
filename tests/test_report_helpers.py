from app.utils.report_helpers import (
    extract_report_action,
    extract_report_summary_markdown,
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


def test_extract_report_summary_markdown_prefers_investment_plan():
    report = {
        "reports": {
            "investment_plan": "# 投资组合经理决策\n\n## 投资建议\n\n建议买入并分批建仓。\n",
            "final_trade_decision": "# 风险管理委员会决策\n\n建议持有。\n",
        },
    }

    summary = extract_report_summary_markdown(report)

    assert "# 投资组合经理决策" not in summary
    assert "## 投资建议" in summary
    assert "建议买入并分批建仓。" in summary
    assert "建议持有。" not in summary


def test_extract_report_summary_markdown_falls_back_to_summary_string():
    report = {
        "summary": '{"context":"腾讯广告恢复，游戏现金流稳定。","action":"买入"}'
    }

    summary = extract_report_summary_markdown(report)

    assert summary == '{"context":"腾讯广告恢复，游戏现金流稳定。","action":"买入"}'
