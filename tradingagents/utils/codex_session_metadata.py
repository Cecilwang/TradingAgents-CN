"""Codex session 元数据辅助函数。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

MODULE_ROLE_MAPPING: Dict[str, str] = {
    "market_report": "Market Analyst",
    "sentiment_report": "Social Media Analyst",
    "news_report": "News Analyst",
    "fundamentals_report": "Fundamentals Analyst",
    "bull_researcher": "Bull Researcher",
    "bear_researcher": "Bear Researcher",
    "research_team_decision": "Research Manager",
    "investment_plan": "Research Manager",
    "trader_investment_plan": "Trader",
    "risky_analyst": "Risky Analyst",
    "safe_analyst": "Safe Analyst",
    "neutral_analyst": "Neutral Analyst",
    "risk_management_decision": "Risk Judge",
    "final_trade_decision": "Risk Judge",
    "investment_debate_state": "Research Manager",
    "risk_debate_state": "Risk Judge",
}

ROLE_DISPLAY_NAMES: Dict[str, str] = {
    "Market Analyst": "市场技术分析师",
    "Social Media Analyst": "市场情绪分析师",
    "News Analyst": "新闻分析师",
    "Fundamentals Analyst": "基本面分析师",
    "Bull Researcher": "多头研究员",
    "Bear Researcher": "空头研究员",
    "Research Manager": "研究经理",
    "Trader": "交易员",
    "Risky Analyst": "激进分析师",
    "Safe Analyst": "保守分析师",
    "Neutral Analyst": "中性分析师",
    "Risk Judge": "投资组合经理",
}


def normalize_codex_role_sessions(value: Any) -> Dict[str, List[str]]:
    """清洗并返回合法的角色会话映射。"""
    if not isinstance(value, dict):
        return {}

    normalized_sessions: Dict[str, List[str]] = {}
    for role_name, session_ids in value.items():
        if not isinstance(role_name, str) or not isinstance(session_ids, list):
            continue

        normalized_session_ids = [
            session_id
            for session_id in session_ids
            if isinstance(session_id, str) and session_id.strip()
        ]
        if normalized_session_ids:
            normalized_sessions[role_name] = normalized_session_ids

    return normalized_sessions


def extract_codex_role_sessions(payload: Any) -> Dict[str, List[str]]:
    """从结果对象或 state 中提取 Codex 角色会话映射。"""
    if not isinstance(payload, dict):
        return {}

    direct_sessions = normalize_codex_role_sessions(payload.get("codex_role_sessions"))
    if direct_sessions:
        return direct_sessions

    nested_state = payload.get("state")
    if isinstance(nested_state, dict):
        return normalize_codex_role_sessions(nested_state.get("codex_role_sessions"))

    return {}


def get_report_codex_sessions(
    report_key: str,
    codex_role_sessions: Dict[str, List[str]],
) -> Optional[Tuple[str, List[str]]]:
    """根据报告模块返回该模块自己的 Codex sessions。"""
    normalized_sessions = normalize_codex_role_sessions(codex_role_sessions)
    role_name = MODULE_ROLE_MAPPING.get(report_key)
    if not role_name:
        return None

    session_ids = normalized_sessions.get(role_name)
    if not session_ids:
        return None

    return ROLE_DISPLAY_NAMES.get(role_name, role_name), session_ids
