from __future__ import annotations

from typing import Any, Dict, List, Optional


RoleSessionMap = Dict[str, List[str]]
CodexSessionEvent = Dict[str, str]


def is_codex_cli_llm(llm: Any) -> bool:
    """判断当前 LLM 是否为本地 Codex CLI 适配器。"""
    return getattr(llm, "_llm_type", "") == "codex_cli"


def get_task_id(state: Dict[str, Any]) -> str:
    """读取当前分析任务 ID。"""
    task_id = state.get("task_id")
    if isinstance(task_id, str):
        return task_id
    return ""


def get_latest_codex_session(state: Dict[str, Any], role_name: str) -> str:
    """返回当前角色最后一个可复用的 Codex session。"""
    role_sessions = copy_role_sessions(state).get(role_name, [])
    return role_sessions[-1] if role_sessions else ""


def copy_role_sessions(state: Dict[str, Any]) -> RoleSessionMap:
    """复制当前任务中的角色会话映射。"""
    codex_role_sessions = state.get("codex_role_sessions")
    if not isinstance(codex_role_sessions, dict):
        return {}

    copied_sessions: RoleSessionMap = {}
    for role_name, session_ids in codex_role_sessions.items():
        if not isinstance(role_name, str) or not isinstance(session_ids, list):
            continue
        normalized_session_ids = [
            session_id
            for session_id in session_ids
            if isinstance(session_id, str) and session_id
        ]
        if normalized_session_ids:
            copied_sessions[role_name] = normalized_session_ids

    return copied_sessions


def build_codex_invoke_kwargs(
    state: Dict[str, Any],
    role_name: str,
    *,
    analysis_type: str = "stock_analysis",
) -> Dict[str, Any]:
    """为 Codex 调用构造通用 invoke 参数。"""
    invoke_kwargs: Dict[str, Any] = {
        "analysis_type": analysis_type,
    }

    task_id = get_task_id(state)
    if task_id:
        invoke_kwargs["task_id"] = task_id

    latest_codex_session = get_latest_codex_session(state, role_name)
    if latest_codex_session:
        invoke_kwargs["resume_session_id"] = latest_codex_session

    return invoke_kwargs


def build_invoke_kwargs(
    llm: Any,
    state: Dict[str, Any],
    role_name: str,
    *,
    analysis_type: str = "stock_analysis",
) -> Dict[str, Any]:
    """统一构造 LLM 调用参数；非 Codex 模型返回空字典。"""
    if not is_codex_cli_llm(llm):
        return {}
    return build_codex_invoke_kwargs(
        state,
        role_name,
        analysis_type=analysis_type,
    )


def extract_codex_session_id(response: Any) -> str:
    """从 LLM 响应元数据中提取 Codex session。"""
    response_metadata = getattr(response, "response_metadata", None)
    if not isinstance(response_metadata, dict):
        return ""

    codex_session_id = response_metadata.get("codex_session_id")
    if isinstance(codex_session_id, str):
        return codex_session_id
    return ""


def build_codex_session_event(
    role_name: str,
    response: Any,
) -> Optional[CodexSessionEvent]:
    """从本次响应提取单次 Codex session 事件。"""
    codex_session_id = extract_codex_session_id(response)
    if not codex_session_id:
        return None
    return {
        "role": role_name,
        "codex_session_id": codex_session_id,
    }


def merge_codex_session_event(
    codex_role_sessions: Dict[str, Any],
    codex_session: Any,
) -> RoleSessionMap:
    """把单次 session 事件合并到最终角色 session 列表。"""
    merged_sessions = copy_role_sessions(
        {"codex_role_sessions": codex_role_sessions}
    )
    if not isinstance(codex_session, dict):
        return merged_sessions

    role_name = codex_session.get("role")
    codex_session_id = codex_session.get("codex_session_id")
    if not isinstance(role_name, str) or not isinstance(codex_session_id, str):
        return merged_sessions
    if not role_name or not codex_session_id:
        return merged_sessions

    role_sessions = merged_sessions.setdefault(role_name, [])
    if not role_sessions or role_sessions[-1] != codex_session_id:
        role_sessions.append(codex_session_id)
    return merged_sessions
