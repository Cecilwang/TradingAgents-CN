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

    return ""


def normalize_report_target_price(value: Any) -> float | None:
    """把目标价统一转换为数字。"""
    if value is None:
        return None

    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        return None

    cleaned = (
        text.replace("HK$", "")
        .replace("US$", "")
        .replace("$", "")
        .replace("¥", "")
        .replace("￥", "")
        .replace(",", "")
        .strip()
    )
    try:
        return float(cleaned)
    except (TypeError, ValueError):
        return None


def extract_report_target_price(report: Dict[str, Any]) -> float | None:
    """从报告文档中提取参考价格/目标价。"""
    if not isinstance(report, dict):
        return None

    decision = report.get("decision")
    if isinstance(decision, dict):
        price = normalize_report_target_price(decision.get("target_price"))
        if price is not None:
            return price

    return None
