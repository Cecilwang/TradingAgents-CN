from __future__ import annotations

from typing import Any, Callable, Dict, Tuple, Union

from tradingagents.utils.logging_init import get_logger

logger = get_logger("default")

PromptFactory = Union[str, Callable[[], str]]


def is_codex_cli_llm(llm: Any) -> bool:
    """判断当前 LLM 是否为本地 Codex CLI 适配器。"""
    return getattr(llm, "_llm_type", "") == "codex_cli"


def get_role_session_id(state: Dict[str, Any], role_name: str) -> str:
    """从当前任务状态中读取角色对应的 Codex session。"""
    codex_role_sessions = state.get("codex_role_sessions")
    if not isinstance(codex_role_sessions, dict):
        return ""

    session_id = codex_role_sessions.get(role_name)
    if isinstance(session_id, str):
        return session_id
    return ""


def invoke_role_with_codex_session(
    *,
    llm: Any,
    state: Dict[str, Any],
    role_name: str,
    full_prompt: PromptFactory,
    continuation_prompt: str,
    analysis_type: str = "stock_analysis",
) -> Tuple[Any, Dict[str, str]]:
    """在单任务内按角色复用 Codex session；非 Codex 模型保持原样调用。"""
    if not is_codex_cli_llm(llm):
        return llm.invoke(_resolve_prompt(full_prompt)), _copy_role_sessions(state)

    task_id = _get_task_id(state)
    existing_session_id = get_role_session_id(state, role_name)
    updated_sessions = _copy_role_sessions(state)
    invoke_kwargs: Dict[str, Any] = {
        "analysis_type": analysis_type,
    }
    if task_id:
        invoke_kwargs["session_id"] = task_id

    if existing_session_id:
        logger.info(
            "🔁 [Codex Session] 复用角色会话: role=%s, task_id=%s, session_id=%s",
            role_name,
            task_id or "-",
            existing_session_id,
        )
        invoke_kwargs["resume_session_id"] = existing_session_id
        try:
            response = llm.invoke(continuation_prompt, **invoke_kwargs)
        except Exception as exc:
            logger.warning(
                "⚠️ [Codex Session] 角色会话恢复失败，改为新建: role=%s, task_id=%s, session_id=%s, error=%s",
                role_name,
                task_id or "-",
                existing_session_id,
                exc,
            )
            invoke_kwargs.pop("resume_session_id", None)
            response = llm.invoke(_resolve_prompt(full_prompt), **invoke_kwargs)
    else:
        logger.info(
            "🆕 [Codex Session] 新建角色会话: role=%s, task_id=%s",
            role_name,
            task_id or "-",
        )
        response = llm.invoke(_resolve_prompt(full_prompt), **invoke_kwargs)

    response_session_id = _extract_response_session_id(response)
    if response_session_id:
        updated_sessions[role_name] = response_session_id

    return response, updated_sessions


def _resolve_prompt(prompt: PromptFactory) -> str:
    """按需构建完整 prompt，避免在 resume 路径重复拼重上下文。"""
    if callable(prompt):
        return prompt()
    return prompt


def _copy_role_sessions(state: Dict[str, Any]) -> Dict[str, str]:
    """复制当前任务里的角色会话映射，避免原地修改输入状态。"""
    codex_role_sessions = state.get("codex_role_sessions")
    if not isinstance(codex_role_sessions, dict):
        return {}
    return {
        role_name: session_id
        for role_name, session_id in codex_role_sessions.items()
        if isinstance(role_name, str) and isinstance(session_id, str)
    }


def _get_task_id(state: Dict[str, Any]) -> str:
    """读取当前任务 ID，作为 Codex token 统计上的分析任务标识。"""
    task_id = state.get("task_id")
    if isinstance(task_id, str):
        return task_id
    return ""


def _extract_response_session_id(response: Any) -> str:
    """从 LLM 响应元数据中提取最新的 Codex session id。"""
    response_metadata = getattr(response, "response_metadata", None)
    if not isinstance(response_metadata, dict):
        return ""

    session_id = response_metadata.get("session_id")
    if isinstance(session_id, str):
        return session_id
    return ""
