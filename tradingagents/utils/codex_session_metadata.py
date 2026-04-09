"""Codex session 元数据辅助函数。"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

# 报告模块到相关 Codex 角色的固定映射。
# 只映射当前真正接入了 Codex 会话复用的角色，避免展示伪 session。
MODULE_ROLE_MAPPING: Dict[str, List[str]] = {
    "bull_researcher": ["Bull Researcher"],
    "bear_researcher": ["Bear Researcher"],
    "research_team_decision": ["Bull Researcher", "Bear Researcher"],
    "investment_debate_state": ["Bull Researcher", "Bear Researcher"],
    "investment_plan": ["Bull Researcher", "Bear Researcher"],
    "risky_analyst": ["Risky Analyst"],
    "safe_analyst": ["Safe Analyst"],
    "neutral_analyst": ["Neutral Analyst"],
    "risk_management_decision": ["Risky Analyst", "Safe Analyst", "Neutral Analyst"],
    "risk_debate_state": ["Risky Analyst", "Safe Analyst", "Neutral Analyst"],
    "final_trade_decision": ["Risky Analyst", "Safe Analyst", "Neutral Analyst"],
}

# 统一前端展示名称，避免直接暴露英文角色名。
ROLE_DISPLAY_NAMES: Dict[str, str] = {
    "Bull Researcher": "多头研究员",
    "Bear Researcher": "空头研究员",
    "Risky Analyst": "激进分析师",
    "Safe Analyst": "保守分析师",
    "Neutral Analyst": "中性分析师",
}


def normalize_codex_role_sessions(value: Any) -> Dict[str, str]:
    """清洗并返回合法的角色会话映射。"""
    if not isinstance(value, dict):
        return {}

    return {
        role_name: session_id
        for role_name, session_id in value.items()
        if isinstance(role_name, str) and isinstance(session_id, str) and session_id.strip()
    }


def extract_codex_role_sessions(payload: Any) -> Dict[str, str]:
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


def get_related_codex_sessions(
    report_key: str,
    codex_role_sessions: Dict[str, str],
) -> List[Tuple[str, str]]:
    """根据报告模块返回可展示的相关 Codex sessions。"""
    normalized_sessions = normalize_codex_role_sessions(codex_role_sessions)
    related_roles = MODULE_ROLE_MAPPING.get(report_key, [])

    return [
        (ROLE_DISPLAY_NAMES.get(role_name, role_name), normalized_sessions[role_name])
        for role_name in related_roles
        if role_name in normalized_sessions
    ]
