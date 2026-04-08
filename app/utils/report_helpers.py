"""
分析报告展示辅助函数。
"""

from __future__ import annotations

import re
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


def parse_target_price_from_text(text: str) -> float | None:
    """从自由文本中提取目标价。"""
    if not text:
        return None

    patterns = [
        r"(?:目标价|目标价格|参考价格|目标价位)\s*[:：]\s*([A-Za-z$¥￥HKUS\.\-0-9, ]+)",
        r"(?:target\s*price)\s*[:：]\s*([A-Za-z$¥￥HKUS\.\-0-9, ]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        price = normalize_report_target_price(match.group(1))
        if price is not None:
            return price
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

    recommendation = parse_target_price_from_text(str(report.get("recommendation", "")))
    if recommendation is not None:
        return recommendation

    reports = report.get("reports") or {}
    if isinstance(reports, dict):
        for key in ("final_trade_decision", "risk_management_decision", "investment_plan", "trader_investment_plan"):
            price = parse_target_price_from_text(str(reports.get(key, "")))
            if price is not None:
                return price

    return None
