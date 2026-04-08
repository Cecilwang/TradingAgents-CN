"""
分析报告展示辅助函数。
"""

from __future__ import annotations

from typing import Any, Dict


def normalize_report_action(value: Any) -> str:
    """把动作值统一成中文买卖建议。"""
    if value is None:
        return ""

    text = str(value).strip()
    if not text:
        return ""

    lower = text.lower()
    exact_map = {
        "buy": "买入",
        "sell": "卖出",
        "hold": "持有",
        "买入": "买入",
        "卖出": "卖出",
        "持有": "持有",
        "增持": "买入",
        "减持": "卖出",
        "观望": "持有",
    }
    if lower in exact_map:
        return exact_map[lower]
    if text in exact_map:
        return exact_map[text]

    return parse_action_from_text(text)


def parse_action_from_text(text: str) -> str:
    """从自由文本中提取买卖持有动作。"""
    if not text:
        return ""

    lower = text.lower()
    if (
        "买入" in text
        or "增持" in text
        or "做多" in text
        or "buy" in lower
    ):
        return "买入"
    if (
        "卖出" in text
        or "减持" in text
        or "清仓" in text
        or "做空" in text
        or "sell" in lower
    ):
        return "卖出"
    if (
        "持有" in text
        or "观望" in text
        or "hold" in lower
    ):
        return "持有"
    return ""


def extract_report_action(report: Dict[str, Any]) -> str:
    """从报告文档中提取执行建议动作。"""
    if not isinstance(report, dict):
        return ""

    decision = report.get("decision")
    if isinstance(decision, dict):
        action = normalize_report_action(decision.get("action"))
        if action:
            return action

    recommendation = parse_action_from_text(str(report.get("recommendation", "")))
    if recommendation:
        return recommendation

    reports = report.get("reports") or {}
    if isinstance(reports, dict):
        for key in ("final_trade_decision", "trader_investment_plan", "investment_plan"):
            action = parse_action_from_text(str(reports.get(key, "")))
            if action:
                return action

    return ""
