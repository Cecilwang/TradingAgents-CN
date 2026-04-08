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


def test_extract_report_summary_markdown_uses_full_final_decision():
    report = {
        "summary": "这是一段被截断的摘要...",
        "reports": {
            "final_trade_decision": "# 00700.HK 最终投资决策\n\n## 投资建议\n\n**行动**: 买入\n\n## 分析推理\n\n长期逻辑稳定。"
        },
    }

    summary = extract_report_summary_markdown(report)

    assert "# 00700.HK 最终投资决策" not in summary
    assert "## 投资建议" in summary
    assert "长期逻辑稳定。" in summary
