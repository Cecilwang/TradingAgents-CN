"""
Codex CLI 本地适配器

将本机 `codex exec` 包装成 LangChain ChatModel，并兼容现有的工具调用链路。
"""

from __future__ import annotations

import asyncio
import json
import os
import select
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Sequence, Union

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict, Field

from tradingagents.utils.logging_manager import get_logger, get_logger_manager

logger = get_logger("agents")

try:
    from tradingagents.config.config_manager import token_tracker

    TOKEN_TRACKING_ENABLED = True
except ImportError:
    token_tracker = None
    TOKEN_TRACKING_ENABLED = False


_CODEX_CLI_PRICING_BY_MODEL: Dict[str, Dict[str, Any]] = {
    "codex-gpt-5.4-mini": {
        "input_price_per_1k": 0.00075,
        "output_price_per_1k": 0.0045,
        "currency": "USD",
    },
    "codex-gpt-5.4": {
        "input_price_per_1k": 0.0025,
        "output_price_per_1k": 0.015,
        "currency": "USD",
    },
}


def _default_working_dir() -> str:
    """返回项目根目录，供 Codex CLI 作为只读工作目录使用。"""
    return str(Path(__file__).resolve().parents[2] / "codex")


def get_codex_cli_status(command: str = "codex") -> Dict[str, Any]:
    """获取本机 Codex CLI 的可用状态和版本号。"""
    try:
        result = subprocess.run(
            [command, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode != 0:
            return {"available": False, "version": None}

        version = (result.stdout or result.stderr).strip() or None
        return {"available": bool(version), "version": version}
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return {"available": False, "version": None}


def is_codex_cli_available(command: str = "codex") -> bool:
    """检查本机是否可用 Codex CLI。"""
    return bool(get_codex_cli_status(command)["available"])


def get_codex_cli_version(command: str = "codex") -> Optional[str]:
    """获取 Codex CLI 版本号，便于在配置页展示本地状态。"""
    return get_codex_cli_status(command)["version"]


def infer_codex_exec_model_name(model_name: str) -> str:
    """根据 Codex 配置键推断实际执行模型。"""
    normalized = str(model_name or "").strip()
    if not normalized or normalized.lower() in {"auto", "default"}:
        raise ValueError("Codex 模型必须显式指定模型代码，不能使用 auto/default。")
    if normalized.startswith("codex-gpt-5.4-mini"):
        return "gpt-5.4-mini"
    if normalized.startswith("codex-gpt-5.4"):
        return "gpt-5.4"
    return normalized


def get_codex_cli_profile_name() -> Optional[str]:
    """读取可选的 Codex profile；未设置时不加载 profile。"""
    profile_name = os.getenv("TA_CODEX_PROFILE", "").strip()
    return profile_name or None


def get_codex_cli_pricing(model_name: str) -> Optional[Dict[str, Any]]:
    """返回 Codex 模型的默认定价元信息。"""
    normalized = str(model_name or "").strip()
    if not normalized:
        return None
    if normalized.startswith("codex-gpt-5.4-mini") or normalized == "gpt-5.4-mini":
        return dict(_CODEX_CLI_PRICING_BY_MODEL["codex-gpt-5.4-mini"])
    if normalized.startswith("codex-gpt-5.4") or normalized == "gpt-5.4":
        return dict(_CODEX_CLI_PRICING_BY_MODEL["codex-gpt-5.4"])
    return None


class ChatCodexCLI(BaseChatModel):
    """基于本地 Codex CLI 的 LangChain ChatModel 适配器。"""

    model_name: str = Field(..., alias="model")
    codex_command: str = Field(default="codex")
    working_dir: str = Field(default_factory=_default_working_dir)
    request_timeout: int = Field(default=180)
    reasoning_effort: str = Field(default="medium")
    fast_mode: bool = Field(default=False)
    ask_for_approval: str = Field(default="never")
    temperature: float = Field(default=0.1)
    max_tokens: Optional[int] = Field(default=None)
    sandbox_mode: str = Field(default="read-only")

    model_config = ConfigDict(populate_by_name=True)

    @property
    def _llm_type(self) -> str:
        """标识当前适配器类型，便于 LangChain 追踪。"""
        return "codex_cli"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """返回识别当前实例的关键参数。"""
        return {
            "model_name": self.model_name,
            "codex_command": self.codex_command,
            "working_dir": self.working_dir,
            "reasoning_effort": self.reasoning_effort,
            "fast_mode": self.fast_mode,
            "profile_name": get_codex_cli_profile_name(),
            "ask_for_approval": self.ask_for_approval,
            "sandbox_mode": self.sandbox_mode,
        }

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], type, Any, BaseTool]],
        *,
        tool_choice: Optional[Union[str, Dict[str, Any], bool]] = None,
        strict: Optional[bool] = None,
        parallel_tool_calls: Optional[bool] = None,
        **kwargs: Any,
    ) -> Runnable[Any, BaseMessage]:
        """绑定工具定义，让 Codex CLI 走统一的 tool-call JSON 输出协议。"""
        formatted_tools = self._format_tools(tools, strict=strict)
        normalized_choice = self._normalize_tool_choice(tool_choice, formatted_tools)
        bind_kwargs = dict(kwargs)
        if parallel_tool_calls is not None:
            bind_kwargs["parallel_tool_calls"] = parallel_tool_calls
        return super().bind(
            tools=formatted_tools,
            tool_choice=normalized_choice,
            **bind_kwargs,
        )

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """调用本地 Codex CLI，并将结果转换为 LangChain ChatResult。"""
        start_time = time.time()
        request = self._prepare_request(messages, kwargs)
        prompt_text = self._build_cli_prompt(
            messages,
            tools=request["tools"],
            tool_choice=request["tool_choice"],
            parallel_tool_calls=request["parallel_tool_calls"],
            resume_session_id=request["resume_session_id"],
        )
        execution_kwargs = {
            "tools": request["tools"],
            "tool_choice": request["tool_choice"],
            "parallel_tool_calls": request["parallel_tool_calls"],
        }
        if request["resume_session_id"]:
            execution_kwargs["resume_session_id"] = request["resume_session_id"]
        execution_result = self._normalize_execution_result(
            self._run_codex_exec(prompt_text, **execution_kwargs)
        )
        parsed_response = self._parse_codex_response(
            execution_result["raw_output"],
            tools=request["tools"],
            tool_choice=request["tool_choice"],
            parallel_tool_calls=request["parallel_tool_calls"],
        )

        return self._build_chat_result(
            parsed_response=parsed_response,
            execution_metadata=execution_result["execution_metadata"],
            session_id=request["session_id"],
            analysis_type=request["analysis_type"],
            start_time=start_time,
        )

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """异步调用本地 Codex CLI，并将结果转换为 LangChain ChatResult。"""
        start_time = time.time()
        request = self._prepare_request(messages, kwargs)
        prompt_text = self._build_cli_prompt(
            messages,
            tools=request["tools"],
            tool_choice=request["tool_choice"],
            parallel_tool_calls=request["parallel_tool_calls"],
            resume_session_id=request["resume_session_id"],
        )
        execution_kwargs = {
            "tools": request["tools"],
            "tool_choice": request["tool_choice"],
            "parallel_tool_calls": request["parallel_tool_calls"],
        }
        if request["resume_session_id"]:
            execution_kwargs["resume_session_id"] = request["resume_session_id"]
        execution_result = self._normalize_execution_result(
            await self._arun_codex_exec(prompt_text, **execution_kwargs)
        )
        parsed_response = self._parse_codex_response(
            execution_result["raw_output"],
            tools=request["tools"],
            tool_choice=request["tool_choice"],
            parallel_tool_calls=request["parallel_tool_calls"],
        )

        return self._build_chat_result(
            parsed_response=parsed_response,
            execution_metadata=execution_result["execution_metadata"],
            session_id=request["session_id"],
            analysis_type=request["analysis_type"],
            start_time=start_time,
        )

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """流式调用本地 Codex CLI，并将 JSON 事件转换为 LangChain chunks。"""
        start_time = time.time()
        request = self._prepare_request(messages, kwargs)
        prompt_text = self._build_cli_prompt(
            messages,
            tools=request["tools"],
            tool_choice=request["tool_choice"],
            parallel_tool_calls=request["parallel_tool_calls"],
            resume_session_id=request["resume_session_id"],
        )
        stream_state: Dict[str, Any] = {
            "saw_text": False,
            "saw_tool_call": False,
            "raw_response": None,
            "execution_metadata": {},
        }

        execution_kwargs = {
            "tools": request["tools"],
            "tool_choice": request["tool_choice"],
            "parallel_tool_calls": request["parallel_tool_calls"],
            "stream_state": stream_state,
            "run_manager": run_manager,
        }
        if request["resume_session_id"]:
            execution_kwargs["resume_session_id"] = request["resume_session_id"]

        yield from self._stream_codex_exec(prompt_text, **execution_kwargs)

        parsed_response = self._parse_codex_response(
            stream_state["raw_response"],
            tools=request["tools"],
            tool_choice=request["tool_choice"],
            parallel_tool_calls=request["parallel_tool_calls"],
        )
        fallback_chunk = self._build_stream_fallback_chunk(
            parsed_response=parsed_response,
            saw_text=stream_state["saw_text"],
            saw_tool_call=stream_state["saw_tool_call"],
        )
        if fallback_chunk is not None:
            self._notify_sync_stream_chunk(run_manager, fallback_chunk)
            yield fallback_chunk

        self._track_codex_usage(
            execution_metadata=stream_state["execution_metadata"],
            session_id=request["session_id"],
            analysis_type=request["analysis_type"],
        )

        elapsed = time.time() - start_time
        logger.info(
            "✅ [Codex CLI] 流式调用完成: model=%s, codex_session_id=%s, tool_calls=%s, elapsed=%.2fs",
            self.model_name,
            stream_state["execution_metadata"].get("thread_id") or "-",
            len(parsed_response["tool_calls"]),
            elapsed,
        )

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """异步流式调用本地 Codex CLI，并将 JSON 事件转换为 LangChain chunks。"""
        start_time = time.time()
        request = self._prepare_request(messages, kwargs)
        prompt_text = self._build_cli_prompt(
            messages,
            tools=request["tools"],
            tool_choice=request["tool_choice"],
            parallel_tool_calls=request["parallel_tool_calls"],
            resume_session_id=request["resume_session_id"],
        )
        stream_state: Dict[str, Any] = {
            "saw_text": False,
            "saw_tool_call": False,
            "raw_response": None,
            "execution_metadata": {},
        }

        execution_kwargs = {
            "tools": request["tools"],
            "tool_choice": request["tool_choice"],
            "parallel_tool_calls": request["parallel_tool_calls"],
            "stream_state": stream_state,
            "run_manager": run_manager,
        }
        if request["resume_session_id"]:
            execution_kwargs["resume_session_id"] = request["resume_session_id"]

        async for chunk in self._astream_codex_exec(prompt_text, **execution_kwargs):
            yield chunk

        parsed_response = self._parse_codex_response(
            stream_state["raw_response"],
            tools=request["tools"],
            tool_choice=request["tool_choice"],
            parallel_tool_calls=request["parallel_tool_calls"],
        )
        fallback_chunk = self._build_stream_fallback_chunk(
            parsed_response=parsed_response,
            saw_text=stream_state["saw_text"],
            saw_tool_call=stream_state["saw_tool_call"],
        )
        if fallback_chunk is not None:
            await self._notify_async_stream_chunk(run_manager, fallback_chunk)
            yield fallback_chunk

        self._track_codex_usage(
            execution_metadata=stream_state["execution_metadata"],
            session_id=request["session_id"],
            analysis_type=request["analysis_type"],
        )

        elapsed = time.time() - start_time
        logger.info(
            "✅ [Codex CLI] 异步流式调用完成: model=%s, codex_session_id=%s, tool_calls=%s, elapsed=%.2fs",
            self.model_name,
            stream_state["execution_metadata"].get("thread_id") or "-",
            len(parsed_response["tool_calls"]),
            elapsed,
        )

    def _build_chat_result(
        self,
        parsed_response: Dict[str, Any],
        execution_metadata: Dict[str, Any],
        session_id: Optional[str],
        analysis_type: Optional[str],
        start_time: float,
    ) -> ChatResult:
        """将规整后的响应构造成 LangChain ChatResult。"""
        response_metadata = self._build_response_metadata(
            execution_metadata=execution_metadata,
            session_id=session_id,
        )
        usage_metadata = self._build_usage_metadata(execution_metadata)
        ai_message = AIMessage(
            content=parsed_response["content"],
            tool_calls=parsed_response["tool_calls"],
            response_metadata=response_metadata,
            usage_metadata=usage_metadata,
        )
        generation = ChatGeneration(message=ai_message)
        self._track_codex_usage(
            execution_metadata=execution_metadata,
            session_id=session_id,
            analysis_type=analysis_type,
        )

        elapsed = time.time() - start_time
        logger.info(
            "✅ [Codex CLI] 调用完成: model=%s, codex_session_id=%s, tool_calls=%s, elapsed=%.2fs",
            self.model_name,
            execution_metadata.get("thread_id") or "-",
            len(parsed_response["tool_calls"]),
            elapsed,
        )

        return ChatResult(
            generations=[generation],
            llm_output={
                "session_id": execution_metadata.get("thread_id") or "",
                "analysis_session_id": session_id or "",
                "token_usage": response_metadata.get("token_usage", {}),
            },
        )

    def _normalize_execution_result(
        self,
        execution_result: Union[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """兼容旧测试桩和新版执行结果结构。"""
        if isinstance(execution_result, str):
            return {
                "raw_output": execution_result,
                "execution_metadata": {},
            }

        if isinstance(execution_result, dict):
            raw_output = execution_result.get("raw_output")
            execution_metadata = execution_result.get("execution_metadata")
            if isinstance(raw_output, str) and isinstance(execution_metadata, dict):
                return {
                    "raw_output": raw_output,
                    "execution_metadata": execution_metadata,
                }

        raise ValueError(
            "Codex CLI 执行结果格式无效，必须包含 raw_output 和 execution_metadata。"
        )

    def _build_response_metadata(
        self,
        *,
        execution_metadata: Dict[str, Any],
        session_id: Optional[str],
    ) -> Dict[str, Any]:
        """构造 LangChain 消息级别的响应元数据。"""
        token_usage = self._build_token_usage(execution_metadata)
        codex_session_id = execution_metadata.get("thread_id") or ""

        return {
            "provider": "codex",
            "session_id": codex_session_id,
            "analysis_session_id": session_id or "",
            "token_usage": token_usage,
        }

    def _build_usage_metadata(
        self,
        execution_metadata: Dict[str, Any],
    ) -> Dict[str, int]:
        """构造 LangChain usage_metadata，供上层统一读取。"""
        usage = execution_metadata.get("usage") or {}
        if not isinstance(usage, dict):
            usage = {}

        return {
            "input_tokens": int(usage.get("input_tokens", 0) or 0),
            "output_tokens": int(usage.get("output_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
            "cached_input_tokens": int(usage.get("cached_input_tokens", 0) or 0),
        }

    def _build_token_usage(
        self,
        execution_metadata: Dict[str, Any],
    ) -> Dict[str, int]:
        """构造兼容现有调用方的 token_usage 结构。"""
        usage_metadata = self._build_usage_metadata(execution_metadata)
        return {
            "prompt_tokens": usage_metadata["input_tokens"],
            "completion_tokens": usage_metadata["output_tokens"],
            "total_tokens": usage_metadata["total_tokens"],
            "cached_input_tokens": usage_metadata["cached_input_tokens"],
        }

    def _track_codex_usage(
        self,
        *,
        execution_metadata: Dict[str, Any],
        session_id: Optional[str],
        analysis_type: Optional[str],
    ) -> None:
        """记录 Codex CLI 的 token 使用和会话标识。"""
        usage_metadata = self._build_usage_metadata(execution_metadata)
        input_tokens = usage_metadata["input_tokens"]
        output_tokens = usage_metadata["output_tokens"]
        cached_input_tokens = usage_metadata["cached_input_tokens"]
        codex_session_id = execution_metadata.get("thread_id") or ""

        if not codex_session_id and input_tokens == 0 and output_tokens == 0:
            return

        effective_session_id = session_id or codex_session_id or ""
        logger.info(
            "📊 [Codex CLI] session_id=%s, analysis_session_id=%s, input_tokens=%s, cached_input_tokens=%s, output_tokens=%s",
            codex_session_id or "-",
            session_id or "-",
            input_tokens,
            cached_input_tokens,
            output_tokens,
        )

        if (
            not TOKEN_TRACKING_ENABLED
            or token_tracker is None
            or (input_tokens == 0 and output_tokens == 0)
        ):
            return

        try:
            usage_record = token_tracker.track_usage(
                provider="codex",
                model_name=self.model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_input_tokens=cached_input_tokens,
                session_id=effective_session_id or codex_session_id,
                analysis_type=analysis_type or "stock_analysis",
                provider_session_id=codex_session_id,
            )

            if usage_record:
                logger_manager = get_logger_manager()
                logger_manager.log_token_usage(
                    logger,
                    "codex",
                    self.model_name,
                    input_tokens,
                    output_tokens,
                    usage_record.cost,
                    effective_session_id or codex_session_id,
                    cached_input_tokens=cached_input_tokens,
                    provider_session_id=codex_session_id,
                )
        except Exception as exc:
            logger.warning("⚠️ [Codex CLI] Token统计失败: %s", exc, exc_info=True)

    def _normalize_tool_choice(
        self,
        tool_choice: Optional[Union[str, Dict[str, Any], bool]],
        tools: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Optional[str]:
        """将 LangChain 的 tool_choice 归一化为内部易处理的字符串。"""
        tool_names = self._extract_tool_names(tools or [])
        if tool_choice in (None, False):
            return None
        if tool_choice == "none":
            return "none"
        if tool_choice == "auto":
            return "auto"
        if tool_choice in (True, "any", "required"):
            if not tool_names:
                raise ValueError("tool_choice=require/any 时必须先提供至少一个工具。")
            return "required"
        if isinstance(tool_choice, dict):
            function_info = tool_choice.get("function", {})
            tool_name = function_info.get("name")
            if not isinstance(tool_name, str) or not tool_name:
                raise ValueError(f"无效的 tool_choice 配置: {tool_choice}")
            if tool_names and tool_name not in tool_names:
                raise ValueError(f"tool_choice 指定了未注册工具: {tool_name}")
            return tool_name
        if isinstance(tool_choice, str):
            if (
                tool_names
                and tool_choice not in {"required", "auto", "none"}
                and tool_choice not in tool_names
            ):
                raise ValueError(f"tool_choice 指定了未注册工具: {tool_choice}")
            return tool_choice
        raise ValueError(f"不支持的 tool_choice 类型: {type(tool_choice)!r}")

    def _prepare_request(
        self,
        messages: List[BaseMessage],
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """抽取并规整工具调用相关参数，供同步/异步入口共享。"""
        session_id = kwargs.pop("session_id", None)
        analysis_type = kwargs.pop("analysis_type", None)
        resume_session_id = kwargs.pop("resume_session_id", None)
        strict = kwargs.pop("strict", None)
        tools = self._format_tools(kwargs.pop("tools", None), strict=strict)
        tool_choice = self._normalize_tool_choice(
            kwargs.pop("tool_choice", None), tools
        )
        parallel_tool_calls = self._normalize_parallel_tool_calls(
            kwargs.pop("parallel_tool_calls", None)
        )

        logger.info(
            "🧩 [Codex CLI] 规整请求: model=%s, ta_session_id=%s, resume_session_id=%s, tools=%s, tool_choice=%s, parallel_tool_calls=%s, messages=%s",
            self.model_name,
            session_id or "-",
            resume_session_id or "-",
            len(tools),
            tool_choice,
            parallel_tool_calls,
            len(messages),
        )

        return {
            "tools": tools,
            "tool_choice": tool_choice,
            "parallel_tool_calls": parallel_tool_calls,
            "session_id": session_id,
            "analysis_type": analysis_type,
            "resume_session_id": resume_session_id,
        }

    def _format_tools(
        self,
        tools: Optional[Sequence[Union[Dict[str, Any], type, Any, BaseTool]]],
        *,
        strict: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """将 LangChain 工具定义统一转换成 OpenAI tool schema。"""
        if not tools:
            return []
        return [convert_to_openai_tool(tool, strict=strict) for tool in tools]

    def _extract_tool_names(self, tools: Sequence[Dict[str, Any]]) -> List[str]:
        """提取工具名列表，便于校验 tool_choice 和输出结果。"""
        tool_names: List[str] = []
        for tool in tools:
            function_info = tool.get("function", {})
            tool_name = function_info.get("name")
            if isinstance(tool_name, str) and tool_name:
                tool_names.append(tool_name)
        return tool_names

    def _normalize_parallel_tool_calls(
        self,
        parallel_tool_calls: Optional[bool],
    ) -> Optional[bool]:
        """校验是否允许模型并行返回多个工具调用。"""
        if parallel_tool_calls is None:
            return None
        if isinstance(parallel_tool_calls, bool):
            return parallel_tool_calls
        raise ValueError(
            f"parallel_tool_calls 必须是 bool 或 None，当前为: {parallel_tool_calls!r}"
        )

    def _build_cli_prompt(
        self,
        messages: List[BaseMessage],
        tools: Optional[Sequence[Dict[str, Any]]],
        tool_choice: Optional[str],
        parallel_tool_calls: Optional[bool],
        resume_session_id: Optional[str] = None,
    ) -> str:
        """构建传给 `codex exec` 的统一提示词。"""
        if resume_session_id:
            return self._build_resume_cli_prompt(
                messages,
                tools=tools,
                tool_choice=tool_choice,
                parallel_tool_calls=parallel_tool_calls,
            )

        tool_definitions = list(tools or [])
        prompt_sections = [
            "你正在作为 TradingAgents 的本地 Codex CLI 模型后端工作。",
            "你只能依据下面给出的会话消息和工具定义工作，不要自行读取仓库、运行 shell、编辑文件或访问额外上下文。",
            "你的输出必须严格是一个 JSON 对象，字段只有：",
            '1. "content": 字符串，最终回复或简短说明',
            '2. "tool_calls": 数组，每项包含 "name" 和 "args"',
            "如果需要工具，返回 tool_calls；如果不需要工具，tool_calls 必须是空数组。",
            "不要输出 Markdown 代码块，不要输出 schema 之外的字段。",
        ]

        if tool_definitions:
            prompt_sections.append("可用工具定义如下（OpenAI tool schema 格式）：")
            prompt_sections.append(
                json.dumps(tool_definitions, ensure_ascii=False, indent=2)
            )
        else:
            prompt_sections.append(
                "当前没有可用工具，你必须直接给出最终回复，并保持 tool_calls 为空数组。"
            )

        if parallel_tool_calls is False:
            prompt_sections.append(
                "本轮最多只能返回一个工具调用；tool_calls 的长度必须是 0 或 1。"
            )

        if tool_choice == "required":
            prompt_sections.append("本轮至少必须请求一个工具调用；不要直接给最终分析。")
        elif tool_choice == "none":
            prompt_sections.append(
                "本轮禁止调用工具；你必须直接给出最终回复，并保持 tool_calls 为空数组。"
            )
        elif tool_choice == "auto":
            prompt_sections.append("本轮你可以自行决定是否需要工具。")
        elif tool_choice:
            prompt_sections.append(
                f"本轮必须调用指定工具：{tool_choice}；不要调用其他工具。"
            )

        prompt_sections.append("会话消息如下：")
        prompt_sections.append(
            json.dumps(self._serialize_messages(messages), ensure_ascii=False, indent=2)
        )

        return "\n\n".join(prompt_sections)

    def _build_resume_cli_prompt(
        self,
        messages: List[BaseMessage],
        *,
        tools: Optional[Sequence[Dict[str, Any]]],
        tool_choice: Optional[str],
        parallel_tool_calls: Optional[bool],
    ) -> str:
        """为 resume 路径构造精简续写提示，避免重复发送首轮包装。"""
        tool_definitions = list(tools or [])
        prompt_sections = [
            "继续当前 Codex 会话。",
            '保持 JSON {"content": string, "tool_calls": array} 输出。',
        ]

        if tool_definitions:
            prompt_sections.append("本轮工具定义如下：")
            prompt_sections.append(
                json.dumps(tool_definitions, ensure_ascii=False, indent=2)
            )
        else:
            prompt_sections.append("本轮无工具，tool_calls 必须为空数组。")

        if parallel_tool_calls is False:
            prompt_sections.append("本轮最多只能返回一个工具调用。")

        if tool_choice == "required":
            prompt_sections.append("本轮至少必须请求一个工具调用。")
        elif tool_choice == "none":
            prompt_sections.append("本轮禁止调用工具，tool_calls 必须为空数组。")
        elif tool_choice == "auto":
            prompt_sections.append("本轮可自行决定是否调用工具。")
        elif tool_choice:
            prompt_sections.append(f"本轮只能调用指定工具：{tool_choice}。")

        prompt_sections.append("新增消息如下：")
        prompt_sections.append(
            json.dumps(self._serialize_messages(messages), ensure_ascii=False, indent=2)
        )

        return "\n\n".join(prompt_sections)

    def _serialize_messages(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """将 LangChain 消息安全序列化，便于 Codex CLI 理解完整上下文。"""
        serialized: List[Dict[str, Any]] = []

        for message in messages:
            item: Dict[str, Any] = {
                "role": self._message_role(message),
                "content": self._message_content(message),
            }

            if isinstance(message, AIMessage) and message.tool_calls:
                item["tool_calls"] = [
                    {
                        "name": tool_call.get("name"),
                        "args": tool_call.get("args", {}),
                        "id": tool_call.get("id"),
                    }
                    for tool_call in message.tool_calls
                ]

            if isinstance(message, ToolMessage):
                item["tool_call_id"] = message.tool_call_id
                item["name"] = getattr(message, "name", None)

            serialized.append(item)

        return serialized

    def _message_role(self, message: BaseMessage) -> str:
        """统一消息角色名称，减少提示词中的歧义。"""
        if isinstance(message, SystemMessage):
            return "system"
        if isinstance(message, HumanMessage):
            return "user"
        if isinstance(message, AIMessage):
            return "assistant"
        if isinstance(message, ToolMessage):
            return "tool"
        if isinstance(message, ChatMessage):
            return message.role
        return getattr(message, "type", "message")

    def _message_content(self, message: BaseMessage) -> str:
        """将消息内容转换为字符串，兼容富文本/列表内容。"""
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        return json.dumps(content, ensure_ascii=False)

    def _build_output_schema(
        self,
        *,
        tools: Sequence[Dict[str, Any]],
        tool_choice: Optional[str],
        parallel_tool_calls: Optional[bool],
    ) -> Dict[str, Any]:
        """按当前工具集合生成 Codex 可接受的严格输出 schema。"""
        tool_names = self._extract_tool_names(tools)
        tool_calls_schema = self._build_tool_calls_schema(
            tools=tools,
            tool_names=tool_names,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
        )
        return {
            "type": "object",
            "properties": {
                "content": {"type": "string"},
                "tool_calls": tool_calls_schema,
            },
            "required": ["content", "tool_calls"],
            "additionalProperties": False,
        }

    def _build_tool_calls_schema(
        self,
        *,
        tools: Sequence[Dict[str, Any]],
        tool_names: Sequence[str],
        tool_choice: Optional[str],
        parallel_tool_calls: Optional[bool],
    ) -> Dict[str, Any]:
        """构造 `tool_calls` 字段的严格 schema。"""
        if not tools or tool_choice == "none":
            return {
                "type": "array",
                "maxItems": 0,
                "items": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            }

        selected_tools = list(tools)
        if tool_choice not in {None, "required", "auto", "none"}:
            selected_tools = [
                tool
                for tool in tools
                if tool.get("function", {}).get("name") == tool_choice
            ]

        item_schemas = [
            self._build_tool_call_item_schema(tool)
            for tool in selected_tools
            if tool.get("function", {}).get("name") in tool_names
        ]
        if not item_schemas:
            raise ValueError("无法根据当前工具配置构造 Codex 输出 schema。")

        items_schema: Dict[str, Any]
        if len(item_schemas) == 1:
            items_schema = item_schemas[0]
        else:
            items_schema = {"oneOf": item_schemas}

        tool_calls_schema: Dict[str, Any] = {
            "type": "array",
            "items": items_schema,
        }

        if parallel_tool_calls is False:
            tool_calls_schema["maxItems"] = 1

        if tool_choice == "required" or tool_choice not in {None, "auto", "none"}:
            tool_calls_schema["minItems"] = 1

        return tool_calls_schema

    def _build_tool_call_item_schema(self, tool: Dict[str, Any]) -> Dict[str, Any]:
        """为单个工具调用生成严格 schema。"""
        function_info = tool.get("function", {})
        tool_name = function_info.get("name")
        if not isinstance(tool_name, str) or not tool_name:
            raise ValueError(f"工具定义缺少合法 name: {tool}")

        parameters = function_info.get("parameters") or {
            "type": "object",
            "properties": {},
        }
        strict_parameters = self._strictify_json_schema(parameters)

        return {
            "type": "object",
            "properties": {
                "name": {"type": "string", "const": tool_name},
                "args": strict_parameters,
            },
            "required": ["name", "args"],
            "additionalProperties": False,
        }

    def _strictify_json_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """递归补齐/收紧 JSON schema，满足 Codex 对 object 的严格要求。"""
        strict_schema: Dict[str, Any] = {}
        original_required = schema.get("required", [])
        if not isinstance(original_required, list):
            original_required = []

        for key, value in schema.items():
            if key in {"properties", "$defs", "definitions"} and isinstance(
                value, dict
            ):
                strict_schema[key] = {
                    item_key: self._strictify_json_schema(item_value)
                    if isinstance(item_value, dict)
                    else item_value
                    for item_key, item_value in value.items()
                }
            elif key == "items" and isinstance(value, dict):
                strict_schema[key] = self._strictify_json_schema(value)
            elif key in {"oneOf", "anyOf", "allOf"} and isinstance(value, list):
                strict_schema[key] = [
                    self._strictify_json_schema(item)
                    if isinstance(item, dict)
                    else item
                    for item in value
                ]
            elif key == "not" and isinstance(value, dict):
                strict_schema[key] = self._strictify_json_schema(value)
            else:
                strict_schema[key] = value

        if strict_schema.get("type") == "object":
            properties = strict_schema.get("properties")
            if not isinstance(properties, dict):
                properties = {}
            strict_properties: Dict[str, Any] = {}
            required_fields: List[str] = []
            for property_name, property_schema in properties.items():
                normalized_property_schema = property_schema
                if (
                    isinstance(normalized_property_schema, dict)
                    and property_name not in original_required
                ):
                    normalized_property_schema = self._make_schema_nullable(
                        normalized_property_schema
                    )
                strict_properties[property_name] = normalized_property_schema
                required_fields.append(property_name)
            strict_schema["properties"] = strict_properties
            strict_schema["required"] = required_fields
            strict_schema["additionalProperties"] = False

        return strict_schema

    def _make_schema_nullable(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """将原本可选的字段转换为 required+nullable，满足 Codex structured output 约束。"""
        nullable_schema = dict(schema)
        schema_type = nullable_schema.get("type")

        if isinstance(schema_type, list):
            if "null" not in schema_type:
                nullable_schema["type"] = [*schema_type, "null"]
            return nullable_schema

        if isinstance(schema_type, str):
            if schema_type != "null":
                nullable_schema["type"] = [schema_type, "null"]
            return nullable_schema

        if isinstance(nullable_schema.get("enum"), list):
            if None not in nullable_schema["enum"]:
                nullable_schema["enum"] = [*nullable_schema["enum"], None]
            return nullable_schema

        return {
            "anyOf": [
                nullable_schema,
                {"type": "null"},
            ]
        }

    def _create_execution_files(
        self,
        *,
        tools: Sequence[Dict[str, Any]],
        tool_choice: Optional[str],
        parallel_tool_calls: Optional[bool],
    ) -> tuple[str, str]:
        """创建 schema/output 临时文件，并返回其路径。"""
        schema = self._build_output_schema(
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
        )

        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False
        ) as schema_file:
            json.dump(schema, schema_file, ensure_ascii=False)
            schema_path = schema_file.name

        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False
        ) as output_file:
            output_path = output_file.name

        return schema_path, output_path

    def _build_codex_command(
        self,
        *,
        schema_path: str,
        output_path: str,
        json_output: bool,
        resume_session_id: Optional[str] = None,
    ) -> List[str]:
        """构造同步/异步/流式调用共用的 Codex CLI 命令。"""
        command = [
            self.codex_command,
            "-a",
            self.ask_for_approval,
        ]
        if self.working_dir:
            command.extend(["-C", self.working_dir])
        profile_name = get_codex_cli_profile_name()
        if profile_name:
            command.extend(["-p", profile_name])
        command.extend(
            [
                "-c",
                f"model_reasoning_effort={json.dumps(self.reasoning_effort)}",
            ]
        )
        if self.fast_mode:
            command.extend(["-c", "service_tier=fast"])
        command.append("exec")
        if resume_session_id:
            command.extend(["resume", resume_session_id])
        if json_output:
            command.append("--json")
        exec_model_name = infer_codex_exec_model_name(self.model_name)
        command.extend(["-m", exec_model_name])
        if self.working_dir:
            command.append("--skip-git-repo-check")
        if not resume_session_id:
            command.extend(
                [
                    "-s",
                    self.sandbox_mode,
                    "--output-schema",
                    schema_path,
                ]
            )
            command.extend(["--color", "never"])
        command.extend(["-o", output_path, "-"])
        return command

    def _run_codex_exec(
        self,
        prompt_text: str,
        *,
        tools: Sequence[Dict[str, Any]],
        tool_choice: Optional[str],
        parallel_tool_calls: Optional[bool],
        resume_session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """执行 `codex exec` 并返回结构化输出与会话元数据。"""
        if not is_codex_cli_available(self.codex_command):
            raise ValueError(
                "未检测到可用的 Codex CLI。请先安装并确认 `codex --version` 可执行。"
            )
        schema_path, output_path = self._create_execution_files(
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
        )
        command = self._build_codex_command(
            schema_path=schema_path,
            output_path=output_path,
            json_output=True,
            resume_session_id=resume_session_id,
        )

        logger.info(
            "🚀 [Codex CLI] 开始调用: command=%s, model=%s, working_dir=%s, resume_session_id=%s",
            self.codex_command,
            self.model_name,
            self.working_dir,
            resume_session_id or "-",
        )

        try:
            result = subprocess.run(
                command,
                input=prompt_text,
                text=True,
                capture_output=True,
                timeout=self.request_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            self._cleanup_temp_files(schema_path, output_path)
            raise TimeoutError(
                f"Codex CLI 调用超时（>{self.request_timeout}秒）"
            ) from exc
        except FileNotFoundError as exc:
            self._cleanup_temp_files(schema_path, output_path)
            raise ValueError("未找到 `codex` 命令，请先安装 Codex CLI。") from exc

        return self._finalize_codex_execution(
            returncode=result.returncode,
            stdout_text=result.stdout or "",
            stderr_text=result.stderr or "",
            schema_path=schema_path,
            output_path=output_path,
            resume_session_id=resume_session_id,
        )

    async def _arun_codex_exec(
        self,
        prompt_text: str,
        *,
        tools: Sequence[Dict[str, Any]],
        tool_choice: Optional[str],
        parallel_tool_calls: Optional[bool],
        resume_session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """异步执行 `codex exec` 并返回结构化输出与会话元数据。"""
        if not is_codex_cli_available(self.codex_command):
            raise ValueError(
                "未检测到可用的 Codex CLI。请先安装并确认 `codex --version` 可执行。"
            )
        schema_path, output_path = self._create_execution_files(
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
        )
        command = self._build_codex_command(
            schema_path=schema_path,
            output_path=output_path,
            json_output=True,
            resume_session_id=resume_session_id,
        )

        logger.info(
            "🚀 [Codex CLI] 开始异步调用: command=%s, model=%s, working_dir=%s, resume_session_id=%s",
            self.codex_command,
            self.model_name,
            self.working_dir,
            resume_session_id or "-",
        )

        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(prompt_text.encode("utf-8")),
                timeout=self.request_timeout,
            )
        except asyncio.TimeoutError as exc:
            self._cleanup_temp_files(schema_path, output_path)
            raise TimeoutError(
                f"Codex CLI 调用超时（>{self.request_timeout}秒）"
            ) from exc
        except FileNotFoundError as exc:
            self._cleanup_temp_files(schema_path, output_path)
            raise ValueError("未找到 `codex` 命令，请先安装 Codex CLI。") from exc

        return self._finalize_codex_execution(
            returncode=process.returncode or 0,
            stdout_text=stdout_bytes.decode("utf-8", errors="replace"),
            stderr_text=stderr_bytes.decode("utf-8", errors="replace"),
            schema_path=schema_path,
            output_path=output_path,
            resume_session_id=resume_session_id,
        )

    def _stream_codex_exec(
        self,
        prompt_text: str,
        *,
        tools: Sequence[Dict[str, Any]],
        tool_choice: Optional[str],
        parallel_tool_calls: Optional[bool],
        stream_state: Dict[str, Any],
        run_manager: Optional[CallbackManagerForLLMRun],
        resume_session_id: Optional[str] = None,
    ) -> Iterator[ChatGenerationChunk]:
        """执行同步流式 Codex 调用，并逐条产出 LangChain chunks。"""
        if not is_codex_cli_available(self.codex_command):
            raise ValueError(
                "未检测到可用的 Codex CLI。请先安装并确认 `codex --version` 可执行。"
            )
        schema_path, output_path = self._create_execution_files(
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
        )
        command = self._build_codex_command(
            schema_path=schema_path,
            output_path=output_path,
            json_output=True,
            resume_session_id=resume_session_id,
        )
        stdout_lines: List[str] = []

        logger.info(
            "🚀 [Codex CLI] 开始流式调用: command=%s, model=%s, working_dir=%s, resume_session_id=%s",
            self.codex_command,
            self.model_name,
            self.working_dir,
            resume_session_id or "-",
        )

        process: Optional[subprocess.Popen[str]] = None
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            self._write_process_input(process, prompt_text)

            yield from self._consume_streaming_process(
                process,
                stream_state=stream_state,
                stdout_lines=stdout_lines,
                run_manager=run_manager,
            )

            stderr_text = process.stderr.read() if process.stderr else ""
            returncode = process.wait()
            execution_result = self._finalize_codex_execution(
                returncode=returncode,
                stdout_text="".join(stdout_lines),
                stderr_text=stderr_text,
                schema_path=schema_path,
                output_path=output_path,
                resume_session_id=resume_session_id,
            )
            stream_state["raw_response"] = execution_result["raw_output"]
            stream_state["execution_metadata"] = execution_result["execution_metadata"]
        except FileNotFoundError as exc:
            self._cleanup_temp_files(schema_path, output_path)
            raise ValueError("未找到 `codex` 命令，请先安装 Codex CLI。") from exc
        except Exception:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()
            self._cleanup_temp_files(schema_path, output_path)
            raise

    async def _astream_codex_exec(
        self,
        prompt_text: str,
        *,
        tools: Sequence[Dict[str, Any]],
        tool_choice: Optional[str],
        parallel_tool_calls: Optional[bool],
        stream_state: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForLLMRun],
        resume_session_id: Optional[str] = None,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """执行异步流式 Codex 调用，并逐条产出 LangChain chunks。"""
        if not is_codex_cli_available(self.codex_command):
            raise ValueError(
                "未检测到可用的 Codex CLI。请先安装并确认 `codex --version` 可执行。"
            )
        schema_path, output_path = self._create_execution_files(
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
        )
        command = self._build_codex_command(
            schema_path=schema_path,
            output_path=output_path,
            json_output=True,
            resume_session_id=resume_session_id,
        )
        stdout_lines: List[str] = []

        logger.info(
            "🚀 [Codex CLI] 开始异步流式调用: command=%s, model=%s, working_dir=%s, resume_session_id=%s",
            self.codex_command,
            self.model_name,
            self.working_dir,
            resume_session_id or "-",
        )

        process: Optional[asyncio.subprocess.Process] = None
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await self._awrite_process_input(process, prompt_text)

            async for chunk in self._aconsume_streaming_process(
                process,
                stream_state=stream_state,
                stdout_lines=stdout_lines,
                run_manager=run_manager,
            ):
                yield chunk

            stderr_bytes = b""
            if process.stderr is not None:
                stderr_bytes = await process.stderr.read()
            returncode = await process.wait()
            execution_result = self._finalize_codex_execution(
                returncode=returncode,
                stdout_text="".join(stdout_lines),
                stderr_text=stderr_bytes.decode("utf-8", errors="replace"),
                schema_path=schema_path,
                output_path=output_path,
                resume_session_id=resume_session_id,
            )
            stream_state["raw_response"] = execution_result["raw_output"]
            stream_state["execution_metadata"] = execution_result["execution_metadata"]
        except FileNotFoundError as exc:
            self._cleanup_temp_files(schema_path, output_path)
            raise ValueError("未找到 `codex` 命令，请先安装 Codex CLI。") from exc
        except Exception:
            if process is not None and process.returncode is None:
                process.kill()
                await process.wait()
            self._cleanup_temp_files(schema_path, output_path)
            raise

    def _write_process_input(
        self,
        process: subprocess.Popen[str],
        prompt_text: str,
    ) -> None:
        """向同步子进程写入 prompt。"""
        if process.stdin is None:
            raise RuntimeError("Codex CLI 未提供可写入的 stdin。")
        try:
            process.stdin.write(prompt_text)
            process.stdin.close()
        except BrokenPipeError as exc:
            raise RuntimeError("Codex CLI 在接收输入前提前退出。") from exc

    async def _awrite_process_input(
        self,
        process: asyncio.subprocess.Process,
        prompt_text: str,
    ) -> None:
        """向异步子进程写入 prompt。"""
        if process.stdin is None:
            raise RuntimeError("Codex CLI 未提供可写入的 stdin。")
        process.stdin.write(prompt_text.encode("utf-8"))
        await process.stdin.drain()
        process.stdin.close()
        if hasattr(process.stdin, "wait_closed"):
            await process.stdin.wait_closed()

    def _consume_streaming_process(
        self,
        process: subprocess.Popen[str],
        *,
        stream_state: Dict[str, Any],
        stdout_lines: List[str],
        run_manager: Optional[CallbackManagerForLLMRun],
    ) -> Iterator[ChatGenerationChunk]:
        """消费同步流式 stdout，并把 JSON 事件转换成 LangChain chunks。"""
        if process.stdout is None:
            raise RuntimeError("Codex CLI 未提供可读取的 stdout。")
        deadline = time.monotonic() + self.request_timeout

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                process.kill()
                process.wait()
                raise TimeoutError(f"Codex CLI 调用超时（>{self.request_timeout}秒）")

            ready, _, _ = select.select([process.stdout], [], [], min(remaining, 0.25))
            if ready:
                line = process.stdout.readline()
                if line:
                    stdout_lines.append(line)
                    chunk = self._chunk_from_stream_line(line)
                    if chunk is not None:
                        self._mark_stream_activity(stream_state, chunk)
                        self._notify_sync_stream_chunk(run_manager, chunk)
                        yield chunk
                    continue

            if process.poll() is not None:
                break

        tail = process.stdout.read()
        if tail:
            for line in tail.splitlines(keepends=True):
                stdout_lines.append(line)
                chunk = self._chunk_from_stream_line(line)
                if chunk is not None:
                    self._mark_stream_activity(stream_state, chunk)
                    self._notify_sync_stream_chunk(run_manager, chunk)
                    yield chunk

    async def _aconsume_streaming_process(
        self,
        process: asyncio.subprocess.Process,
        *,
        stream_state: Dict[str, Any],
        stdout_lines: List[str],
        run_manager: Optional[AsyncCallbackManagerForLLMRun],
    ) -> AsyncIterator[ChatGenerationChunk]:
        """消费异步流式 stdout，并把 JSON 事件转换成 LangChain chunks。"""
        if process.stdout is None:
            raise RuntimeError("Codex CLI 未提供可读取的 stdout。")
        deadline = time.monotonic() + self.request_timeout

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                process.kill()
                await process.wait()
                raise TimeoutError(f"Codex CLI 调用超时（>{self.request_timeout}秒）")

            try:
                line_bytes = await asyncio.wait_for(
                    process.stdout.readline(),
                    timeout=min(remaining, 0.25),
                )
            except asyncio.TimeoutError:
                if process.returncode is not None:
                    break
                continue

            if line_bytes:
                line = line_bytes.decode("utf-8", errors="replace")
                stdout_lines.append(line)
                chunk = self._chunk_from_stream_line(line)
                if chunk is not None:
                    self._mark_stream_activity(stream_state, chunk)
                    await self._notify_async_stream_chunk(run_manager, chunk)
                    yield chunk
                continue

            if process.stdout.at_eof():
                break

        tail = await process.stdout.read()
        if tail:
            for line in tail.decode("utf-8", errors="replace").splitlines(
                keepends=True
            ):
                stdout_lines.append(line)
                chunk = self._chunk_from_stream_line(line)
                if chunk is not None:
                    self._mark_stream_activity(stream_state, chunk)
                    await self._notify_async_stream_chunk(run_manager, chunk)
                    yield chunk

    def _chunk_from_stream_line(self, raw_line: str) -> Optional[ChatGenerationChunk]:
        """将 Codex CLI 的单行 JSON 事件转换为 LangChain chunk。"""
        event = self._load_stream_event(raw_line)
        if event is None:
            return None

        event_type = event.get("type")
        if event_type == "response.output_text.delta":
            delta = event.get("delta")
            if not isinstance(delta, str) or not delta:
                return None
            return ChatGenerationChunk(message=AIMessageChunk(content=delta))

        if event_type == "response.created":
            response_id = event.get("response", {}).get("id")
            if not isinstance(response_id, str) or not response_id:
                return None
            return ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    response_metadata={"id": response_id},
                )
            )

        if event_type == "response.output_item.added":
            item = event.get("item", {})
            if item.get("type") != "function_call":
                return None
            return ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {
                            "type": "tool_call_chunk",
                            "name": item.get("name"),
                            "args": item.get("arguments", ""),
                            "id": item.get("call_id"),
                            "index": event.get("output_index", 0),
                        }
                    ],
                    id=item.get("id"),
                )
            )

        if event_type == "response.function_call_arguments.delta":
            delta = event.get("delta")
            if not isinstance(delta, str):
                return None
            return ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {
                            "type": "tool_call_chunk",
                            "args": delta,
                            "index": event.get("output_index", 0),
                        }
                    ],
                )
            )

        return None

    def _load_stream_event(self, raw_line: str) -> Optional[Dict[str, Any]]:
        """尽量鲁棒地解析 Codex CLI `--json` 输出中的单行事件。"""
        candidate = raw_line.strip()
        if not candidate:
            return None
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            logger.debug("🔍 [Codex CLI] 忽略非 JSON 流事件: %s", candidate[:200])
            return None

        if not isinstance(payload, dict):
            logger.debug("🔍 [Codex CLI] 忽略非对象流事件: %s", candidate[:200])
            return None
        return payload

    def _mark_stream_activity(
        self,
        stream_state: Dict[str, Any],
        chunk: ChatGenerationChunk,
    ) -> None:
        """记录本轮流式输出是否已经产出文本或工具调用。"""
        message = chunk.message
        if isinstance(message.content, str) and message.content:
            stream_state["saw_text"] = True
        elif isinstance(message.content, list) and message.content:
            stream_state["saw_text"] = True

        if getattr(message, "tool_call_chunks", None):
            stream_state["saw_tool_call"] = True

    def _notify_sync_stream_chunk(
        self,
        run_manager: Optional[CallbackManagerForLLMRun],
        chunk: ChatGenerationChunk,
    ) -> None:
        """把同步流式 chunk 回调给 LangChain callback manager。"""
        if run_manager is None:
            return
        run_manager.on_llm_new_token(self._chunk_text(chunk), chunk=chunk)

    async def _notify_async_stream_chunk(
        self,
        run_manager: Optional[AsyncCallbackManagerForLLMRun],
        chunk: ChatGenerationChunk,
    ) -> None:
        """把异步流式 chunk 回调给 LangChain callback manager。"""
        if run_manager is None:
            return
        await run_manager.on_llm_new_token(self._chunk_text(chunk), chunk=chunk)

    def _chunk_text(self, chunk: ChatGenerationChunk) -> str:
        """提取流式 chunk 的文本部分，供 callback manager 使用。"""
        content = chunk.message.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        text_parts.append(text)
            return "".join(text_parts)
        return ""

    def _build_stream_fallback_chunk(
        self,
        *,
        parsed_response: Dict[str, Any],
        saw_text: bool,
        saw_tool_call: bool,
    ) -> Optional[ChatGenerationChunk]:
        """当 CLI 没有产生足够流式事件时，用最终落盘结果补齐缺失输出。"""
        fallback_content = ""
        if not saw_text and parsed_response["content"]:
            fallback_content = parsed_response["content"]

        fallback_tool_calls: List[Dict[str, Any]] = []
        if not saw_tool_call:
            for index, tool_call in enumerate(parsed_response["tool_calls"]):
                fallback_tool_calls.append(
                    {
                        "type": "tool_call_chunk",
                        "name": tool_call["name"],
                        "args": json.dumps(tool_call["args"], ensure_ascii=False),
                        "id": tool_call["id"],
                        "index": index,
                    }
                )

        if not fallback_content and not fallback_tool_calls:
            return None

        return ChatGenerationChunk(
            message=AIMessageChunk(
                content=fallback_content,
                tool_call_chunks=fallback_tool_calls,
            )
        )

    def _finalize_codex_execution(
        self,
        *,
        returncode: int,
        stdout_text: str,
        stderr_text: str,
        schema_path: str,
        output_path: str,
        resume_session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """读取最终输出、清理临时文件，并统一处理失败场景。"""
        try:
            output_file = Path(output_path)
            raw_output = ""
            if output_file.exists():
                raw_output = output_file.read_text(encoding="utf-8").strip()
        finally:
            self._cleanup_temp_files(schema_path, output_path)

        execution_metadata = self._extract_execution_metadata(stdout_text)
        if resume_session_id and not execution_metadata.get("thread_id"):
            execution_metadata["thread_id"] = resume_session_id

        if returncode != 0:
            logger.error(
                "❌ [Codex CLI] 调用失败: returncode=%s, stderr=%s",
                returncode,
                stderr_text.strip(),
            )
            raise RuntimeError(
                "Codex CLI 调用失败："
                f"returncode={returncode}, stderr={stderr_text.strip() or '<empty>'}, stdout={stdout_text.strip() or '<empty>'}"
            )

        if not raw_output:
            raise RuntimeError("Codex CLI 未返回可解析的输出内容。")

        if resume_session_id:
            raw_output = self._normalize_resume_raw_output(raw_output)

        return {
            "raw_output": raw_output,
            "execution_metadata": execution_metadata,
        }

    def _normalize_resume_raw_output(self, raw_output: str) -> str:
        """将 resume 路径的最后一条消息规整回现有 JSON 响应结构。"""
        try:
            payload = self._load_json_payload(raw_output)
        except ValueError:
            return json.dumps(
                {
                    "content": raw_output,
                    "tool_calls": [],
                },
                ensure_ascii=False,
            )

        if (
            isinstance(payload.get("content"), str)
            and isinstance(payload.get("tool_calls"), list)
        ):
            return raw_output

        return json.dumps(
            {
                "content": raw_output,
                "tool_calls": [],
            },
            ensure_ascii=False,
        )

    def _extract_execution_metadata(self, stdout_text: str) -> Dict[str, Any]:
        """从 Codex CLI `--json` 事件流中提取 session 和 token 使用量。"""
        thread_id = ""
        usage = {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

        for raw_line in stdout_text.splitlines():
            event = self._load_stream_event(raw_line)
            if event is None:
                continue

            event_type = event.get("type")
            if event_type == "thread.started":
                candidate_thread_id = event.get("thread_id")
                if isinstance(candidate_thread_id, str) and candidate_thread_id:
                    thread_id = candidate_thread_id
                continue

            if event_type not in {"turn.completed", "response.completed"}:
                continue

            raw_usage = event.get("usage")
            if not isinstance(raw_usage, dict):
                response_payload = event.get("response")
                if isinstance(response_payload, dict):
                    raw_usage = response_payload.get("usage")
            if not isinstance(raw_usage, dict):
                continue

            input_tokens = int(raw_usage.get("input_tokens", 0) or 0)
            cached_input_tokens = int(raw_usage.get("cached_input_tokens", 0) or 0)
            output_tokens = int(raw_usage.get("output_tokens", 0) or 0)
            usage = {
                "input_tokens": input_tokens,
                "cached_input_tokens": cached_input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }

        return {
            "thread_id": thread_id,
            "usage": usage,
        }

    def _cleanup_temp_files(self, *temp_paths: str) -> None:
        """清理调用过程中产生的临时文件。"""
        for temp_path in temp_paths:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                continue
            except OSError:
                logger.warning("⚠️ [Codex CLI] 临时文件清理失败: %s", temp_path)

    def _parse_codex_response(
        self,
        raw_response: str,
        tools: Optional[Sequence[Dict[str, Any]]],
        tool_choice: Optional[str] = None,
        parallel_tool_calls: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """解析 Codex CLI 输出，并规整为 LangChain 可消费的结构。"""
        payload = self._load_json_payload(raw_response)
        content = payload.get("content") or ""
        raw_tool_calls = payload.get("tool_calls") or []
        if not isinstance(raw_tool_calls, list):
            raise ValueError("Codex CLI 返回的 tool_calls 必须是数组。")
        normalized_tool_calls = self._normalize_tool_calls(
            raw_tool_calls,
            tools or [],
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
        )

        return {
            "content": content,
            "tool_calls": normalized_tool_calls,
        }

    def _load_json_payload(self, raw_response: str) -> Dict[str, Any]:
        """尽量鲁棒地把 CLI 返回文本解析为 JSON 对象。"""
        candidate = raw_response.strip()
        if candidate.startswith("```"):
            candidate = candidate.strip("`")
            if "\n" in candidate:
                candidate = candidate.split("\n", 1)[1]
            if candidate.endswith("```"):
                candidate = candidate[:-3]
            candidate = candidate.strip()

        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError(f"Codex CLI 返回了非 JSON 内容: {raw_response[:300]}")
            payload = json.loads(candidate[start : end + 1])

        if not isinstance(payload, dict):
            raise ValueError("Codex CLI 返回的 JSON 根对象必须是 object。")

        return payload

    def _normalize_tool_calls(
        self,
        raw_tool_calls: List[Dict[str, Any]],
        tools: Sequence[Dict[str, Any]],
        *,
        tool_choice: Optional[str] = None,
        parallel_tool_calls: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """校验工具名和参数，避免非法 tool call 进入 LangGraph。"""
        allowed_tools = set(self._extract_tool_names(tools))
        if tool_choice == "none":
            if raw_tool_calls:
                logger.warning(
                    "⚠️ [Codex CLI] 当前轮次禁止工具调用，已忽略模型返回的 tool_calls。"
                )
            return []

        required_tool_name: Optional[str] = None
        if tool_choice not in {None, "required", "auto", "none"}:
            required_tool_name = tool_choice

        normalized: List[Dict[str, Any]] = []
        for tool_call in raw_tool_calls:
            tool_name = tool_call.get("name")
            if not isinstance(tool_name, str) or not tool_name:
                logger.warning(
                    "⚠️ [Codex CLI] 忽略无效工具调用（缺少 name）: %s", tool_call
                )
                continue

            if allowed_tools and tool_name not in allowed_tools:
                logger.warning("⚠️ [Codex CLI] 忽略未注册工具调用: %s", tool_name)
                continue

            if required_tool_name and tool_name != required_tool_name:
                logger.warning(
                    "⚠️ [Codex CLI] 忽略不符合 tool_choice 的工具调用: expected=%s, actual=%s",
                    required_tool_name,
                    tool_name,
                )
                continue

            tool_args = tool_call.get("args", {})
            if isinstance(tool_args, str):
                try:
                    tool_args = json.loads(tool_args)
                except json.JSONDecodeError:
                    logger.warning(
                        "⚠️ [Codex CLI] 工具参数不是合法 JSON，已降级为空对象: %s",
                        tool_name,
                    )
                    tool_args = {}

            if not isinstance(tool_args, dict):
                logger.warning(
                    "⚠️ [Codex CLI] 工具参数不是对象，已降级为空对象: %s", tool_name
                )
                tool_args = {}

            normalized.append(
                {
                    "name": tool_name,
                    "args": tool_args,
                    "id": tool_call.get("id") or f"call_{uuid.uuid4().hex}",
                    "type": "tool_call",
                }
            )

        if parallel_tool_calls is False and len(normalized) > 1:
            logger.warning(
                "⚠️ [Codex CLI] 当前轮次禁止并行工具调用，已裁剪为首个工具: %s",
                normalized[0]["name"],
            )
            normalized = normalized[:1]

        if tool_choice == "required" and not normalized:
            raise RuntimeError("Codex CLI 未按要求返回工具调用。")

        if required_tool_name and not normalized:
            raise RuntimeError(
                f"Codex CLI 未按要求返回指定工具调用: {required_tool_name}"
            )

        return normalized
