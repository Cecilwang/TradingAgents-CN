import os
import asyncio
import json
import subprocess
from pathlib import Path

os.environ.setdefault("USE_MONGODB_STORAGE", "false")

import pytest
from langchain_core.language_models.chat_models import generate_from_stream
from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGenerationChunk

from app.services.config_service import ConfigService
from tradingagents.graph.trading_graph import create_llm_by_provider
from tradingagents.llm_adapters import codex_cli_adapter
from tradingagents.llm_adapters.codex_cli_adapter import (
    ChatCodexCLI,
    get_codex_cli_profile_name,
    infer_codex_exec_model_name,
)


@pytest.mark.parametrize("model_name", ["auto", "default", ""])
def test_infer_codex_exec_model_name_rejects_implicit_models(model_name: str):
    with pytest.raises(ValueError, match="不能使用 auto/default"):
        infer_codex_exec_model_name(model_name)


def test_build_codex_command_always_passes_explicit_model():
    llm = ChatCodexCLI(model="codex-gpt-5.4-medium")

    command = llm._build_codex_command(
        schema_path="/tmp/schema.json",
        output_path="/tmp/output.json",
        json_output=True,
    )

    assert "-m" in command
    assert command[command.index("-m") + 1] == "gpt-5.4"


def test_build_codex_command_uses_medium_by_default():
    llm = ChatCodexCLI(model="codex-gpt-5.4")

    command = llm._build_codex_command(
        schema_path="/tmp/schema.json",
        output_path="/tmp/output.json",
        json_output=True,
    )

    assert 'model_reasoning_effort="medium"' in command
    assert "service_tier=fast" not in command
    assert "service_tier=flex" not in command


def test_get_codex_cli_profile_name_from_env(monkeypatch):
    monkeypatch.setenv("TA_CODEX_PROFILE", "trading")

    assert get_codex_cli_profile_name() == "trading"


def test_build_codex_command_includes_optional_profile(monkeypatch):
    monkeypatch.setenv("TA_CODEX_PROFILE", "trading")
    llm = ChatCodexCLI(model="codex-gpt-5.4")

    command = llm._build_codex_command(
        schema_path="/tmp/schema.json",
        output_path="/tmp/output.json",
        json_output=True,
    )

    assert "-p" in command
    assert command[command.index("-p") + 1] == "trading"


def test_build_codex_command_resume_omits_schema_and_sandbox():
    llm = ChatCodexCLI(model="codex-gpt-5.4")

    command = llm._build_codex_command(
        schema_path="/tmp/schema.json",
        output_path="/tmp/output.json",
        json_output=True,
        resume_session_id="thread_existing",
    )

    assert "exec" in command
    assert "resume" in command
    assert "thread_existing" in command
    assert "--json" in command
    assert "--output-schema" not in command
    assert "-s" not in command
    assert "--color" not in command
    assert "-o" in command


def test_parse_codex_response_with_tool_calls():
    llm = ChatCodexCLI(model="gpt-5.4")

    payload = json.dumps(
        {
            "content": "",
            "tool_calls": [
                {
                    "name": "lookup_price",
                    "args": {"ticker": "AAPL"},
                }
            ],
        },
        ensure_ascii=False,
    )

    parsed = llm._parse_codex_response(
        payload,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup_price",
                    "description": "查询股价",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ticker": {"type": "string"},
                        },
                    },
                },
            }
        ],
    )

    assert parsed["content"] == ""
    assert len(parsed["tool_calls"]) == 1
    assert parsed["tool_calls"][0]["name"] == "lookup_price"
    assert parsed["tool_calls"][0]["args"] == {"ticker": "AAPL"}
    assert parsed["tool_calls"][0]["type"] == "tool_call"
    assert parsed["tool_calls"][0]["id"].startswith("call_")


def test_generate_returns_ai_message_with_tool_calls(monkeypatch):
    llm = ChatCodexCLI(model="gpt-5.4")

    def mock_run_codex_exec(
        prompt_text: str,
        *,
        tools,
        tool_choice,
        parallel_tool_calls,
    ) -> str:
        assert "会话消息如下" in prompt_text
        assert tool_choice is None
        assert parallel_tool_calls is None
        assert len(tools) == 1
        return json.dumps(
            {
                "content": "需要先调用工具",
                "tool_calls": [
                    {
                        "name": "lookup_price",
                        "args": {"ticker": "MSFT"},
                    }
                ],
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(llm, "_run_codex_exec", mock_run_codex_exec)

    result = llm._generate(
        [HumanMessage(content="请先查询微软股价")],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup_price",
                    "description": "查询股价",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ticker": {"type": "string"},
                        },
                    },
                },
            }
        ],
    )

    message = result.generations[0].message
    assert message.content == "需要先调用工具"
    assert len(message.tool_calls) == 1
    assert message.tool_calls[0]["name"] == "lookup_price"
    assert message.tool_calls[0]["args"] == {"ticker": "MSFT"}


def test_agenerate_returns_ai_message_with_tool_calls(monkeypatch):
    llm = ChatCodexCLI(model="gpt-5.4")

    async def mock_arun_codex_exec(
        prompt_text: str,
        *,
        tools,
        tool_choice,
        parallel_tool_calls,
    ) -> str:
        assert "会话消息如下" in prompt_text
        assert tool_choice is None
        assert parallel_tool_calls is None
        assert len(tools) == 1
        return json.dumps(
            {
                "content": "异步需要先调用工具",
                "tool_calls": [
                    {
                        "name": "lookup_price",
                        "args": {"ticker": "NVDA"},
                    }
                ],
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(llm, "_arun_codex_exec", mock_arun_codex_exec)

    result = asyncio.run(
        llm._agenerate(
            [HumanMessage(content="请异步查询英伟达股价")],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "lookup_price",
                        "description": "查询股价",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "ticker": {"type": "string"},
                            },
                        },
                    },
                }
            ],
        )
    )

    message = result.generations[0].message
    assert message.content == "异步需要先调用工具"
    assert len(message.tool_calls) == 1
    assert message.tool_calls[0]["name"] == "lookup_price"
    assert message.tool_calls[0]["args"] == {"ticker": "NVDA"}


def test_extract_execution_metadata_reads_thread_and_usage():
    llm = ChatCodexCLI(model="gpt-5.4")

    stdout_text = "\n".join(
        [
            json.dumps(
                {
                    "type": "thread.started",
                    "thread_id": "thread_123",
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "type": "turn.completed",
                    "usage": {
                        "input_tokens": 120,
                        "cached_input_tokens": 30,
                        "output_tokens": 45,
                    },
                },
                ensure_ascii=False,
            ),
        ]
    )

    metadata = llm._extract_execution_metadata(stdout_text)

    assert metadata["thread_id"] == "thread_123"
    assert metadata["usage"] == {
        "input_tokens": 120,
        "cached_input_tokens": 30,
        "output_tokens": 45,
        "total_tokens": 165,
    }


def test_generate_attaches_codex_usage_metadata(monkeypatch):
    llm = ChatCodexCLI(model="gpt-5.4")

    monkeypatch.setattr(codex_cli_adapter, "TOKEN_TRACKING_ENABLED", False)

    def mock_run_codex_exec(
        prompt_text: str,
        *,
        tools,
        tool_choice,
        parallel_tool_calls,
    ):
        assert "会话消息如下" in prompt_text
        return {
            "raw_output": json.dumps(
                {
                    "content": "执行完成",
                    "tool_calls": [],
                },
                ensure_ascii=False,
            ),
            "execution_metadata": {
                "thread_id": "thread_456",
                "usage": {
                    "input_tokens": 210,
                    "cached_input_tokens": 80,
                    "output_tokens": 55,
                    "total_tokens": 265,
                },
            },
        }

    monkeypatch.setattr(llm, "_run_codex_exec", mock_run_codex_exec)

    result = llm._generate(
        [HumanMessage(content="给出最终结果")],
        session_id="analysis_789",
        analysis_type="stock_analysis",
    )

    message = result.generations[0].message

    assert message.response_metadata["session_id"] == "thread_456"
    assert message.response_metadata["analysis_session_id"] == "analysis_789"
    assert message.response_metadata["token_usage"] == {
        "prompt_tokens": 210,
        "completion_tokens": 55,
        "total_tokens": 265,
        "cached_input_tokens": 80,
    }
    assert message.usage_metadata == {
        "input_tokens": 210,
        "output_tokens": 55,
        "total_tokens": 265,
        "cached_input_tokens": 80,
    }
    assert result.llm_output == {
        "session_id": "thread_456",
        "analysis_session_id": "analysis_789",
        "token_usage": {
            "prompt_tokens": 210,
            "completion_tokens": 55,
            "total_tokens": 265,
            "cached_input_tokens": 80,
        },
    }


def test_generate_passes_resume_session_id(monkeypatch):
    llm = ChatCodexCLI(model="gpt-5.4")

    def mock_run_codex_exec(
        prompt_text: str,
        *,
        tools,
        tool_choice,
        parallel_tool_calls,
        resume_session_id=None,
    ):
        assert "会话消息如下" in prompt_text
        assert resume_session_id == "thread_prev"
        return {
            "raw_output": json.dumps(
                {
                    "content": "继续回应",
                    "tool_calls": [],
                },
                ensure_ascii=False,
            ),
            "execution_metadata": {
                "thread_id": "thread_prev",
                "usage": {
                    "input_tokens": 11,
                    "cached_input_tokens": 3,
                    "output_tokens": 5,
                    "total_tokens": 16,
                },
            },
        }

    monkeypatch.setattr(llm, "_run_codex_exec", mock_run_codex_exec)

    result = llm._generate(
        [HumanMessage(content="继续讨论")],
        session_id="analysis_1",
        analysis_type="stock_analysis",
        resume_session_id="thread_prev",
    )

    message = result.generations[0].message
    assert message.content == "继续回应"
    assert message.response_metadata["session_id"] == "thread_prev"
    assert message.response_metadata["analysis_session_id"] == "analysis_1"


def test_create_llm_by_provider_returns_codex_cli():
    llm = create_llm_by_provider(
        provider="codex",
        model="gpt-5.4",
        backend_url="local://codex-cli",
        temperature=0.2,
        max_tokens=512,
        timeout=90,
        model_config={
            "reasoning_effort": "high",
            "fast_mode": True,
            "ask_for_approval": "untrusted",
            "sandbox_mode": "workspace-write",
        },
    )

    assert isinstance(llm, ChatCodexCLI)
    assert llm.model_name == "gpt-5.4"
    assert llm.request_timeout == 90
    assert llm.reasoning_effort == "high"
    assert llm.fast_mode is True
    assert llm.ask_for_approval == "untrusted"
    assert llm.sandbox_mode == "workspace-write"


def test_run_codex_exec_includes_cli_overrides(monkeypatch):
    llm = ChatCodexCLI(
        model="gpt-5.4",
        reasoning_effort="high",
        fast_mode=True,
        ask_for_approval="untrusted",
        sandbox_mode="workspace-write",
        request_timeout=45,
    )
    captured: dict[str, list[str]] = {}

    class DummyCompletedProcess:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(
        codex_cli_adapter, "is_codex_cli_available", lambda _command="codex": True
    )

    def mock_run(command, **kwargs):
        captured["command"] = command
        assert kwargs["input"] == "hello"
        assert kwargs["timeout"] == 45
        return DummyCompletedProcess()

    monkeypatch.setattr(subprocess, "run", mock_run)
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda self, encoding="utf-8": '{"content":"ok","tool_calls":[]}',
    )

    execution_result = llm._run_codex_exec(
        "hello",
        tools=[],
        tool_choice=None,
        parallel_tool_calls=None,
    )

    assert execution_result["raw_output"] == '{"content":"ok","tool_calls":[]}'
    assert execution_result["execution_metadata"] == {
        "thread_id": "",
        "usage": {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        },
    }
    assert captured["command"][0:3] == ["codex", "-a", "untrusted"]
    assert 'model_reasoning_effort="high"' in captured["command"]
    assert "service_tier=fast" in captured["command"]
    assert "exec" in captured["command"]
    assert "--json" in captured["command"]
    assert "workspace-write" in captured["command"]


def test_finalize_codex_execution_normalizes_resume_output(tmp_path):
    llm = ChatCodexCLI(model="gpt-5.4")
    schema_path = tmp_path / "schema.json"
    output_path = tmp_path / "output.json"
    schema_path.write_text("{}", encoding="utf-8")
    output_path.write_text("继续辩论", encoding="utf-8")

    execution_result = llm._finalize_codex_execution(
        returncode=0,
        stdout_text=json.dumps(
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 12,
                    "cached_input_tokens": 4,
                    "output_tokens": 8,
                },
            },
            ensure_ascii=False,
        ),
        stderr_text="",
        schema_path=str(schema_path),
        output_path=str(output_path),
        resume_session_id="thread_resume_1",
    )

    assert execution_result["raw_output"] == json.dumps(
        {
            "content": "继续辩论",
            "tool_calls": [],
        },
        ensure_ascii=False,
    )
    assert execution_result["execution_metadata"] == {
        "thread_id": "thread_resume_1",
        "usage": {
            "input_tokens": 12,
            "cached_input_tokens": 4,
            "output_tokens": 8,
            "total_tokens": 20,
        },
    }


def test_bind_tools_normalizes_choice_and_parallel_flag():
    llm = ChatCodexCLI(model="gpt-5.4")
    bound = llm.bind_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "lookup_price",
                    "description": "查询股价",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ticker": {"type": "string"},
                        },
                    },
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "lookup_price"}},
        parallel_tool_calls=False,
    )

    assert bound.kwargs["tool_choice"] == "lookup_price"
    assert bound.kwargs["parallel_tool_calls"] is False
    assert bound.kwargs["tools"][0]["function"]["name"] == "lookup_price"


def test_build_output_schema_makes_tool_args_strict():
    llm = ChatCodexCLI(model="gpt-5.4")

    schema = llm._build_output_schema(
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup_price",
                    "description": "查询股价",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ticker": {"type": "string"},
                            "start_date": {"type": "string", "default": None},
                            "end_date": {"type": "string", "default": None},
                            "curr_date": {"type": "string", "default": None},
                        },
                        "required": ["ticker"],
                    },
                },
            }
        ],
        tool_choice="lookup_price",
        parallel_tool_calls=False,
    )

    tool_calls_schema = schema["properties"]["tool_calls"]
    item_schema = tool_calls_schema["items"]

    assert tool_calls_schema["minItems"] == 1
    assert tool_calls_schema["maxItems"] == 1
    assert item_schema["properties"]["name"]["const"] == "lookup_price"
    assert item_schema["properties"]["args"]["additionalProperties"] is False
    assert item_schema["properties"]["args"]["required"] == [
        "ticker",
        "start_date",
        "end_date",
        "curr_date",
    ]
    assert item_schema["properties"]["args"]["properties"]["start_date"]["type"] == [
        "string",
        "null",
    ]
    assert item_schema["properties"]["args"]["properties"]["end_date"]["type"] == [
        "string",
        "null",
    ]
    assert item_schema["properties"]["args"]["properties"]["curr_date"]["type"] == [
        "string",
        "null",
    ]


def test_parse_codex_response_trims_parallel_tool_calls():
    llm = ChatCodexCLI(model="gpt-5.4")

    payload = json.dumps(
        {
            "content": "",
            "tool_calls": [
                {
                    "name": "lookup_price",
                    "args": {"ticker": "AAPL"},
                },
                {
                    "name": "lookup_price",
                    "args": {"ticker": "MSFT"},
                },
            ],
        },
        ensure_ascii=False,
    )

    parsed = llm._parse_codex_response(
        payload,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup_price",
                    "description": "查询股价",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ticker": {"type": "string"},
                        },
                    },
                },
            }
        ],
        parallel_tool_calls=False,
    )

    assert len(parsed["tool_calls"]) == 1
    assert parsed["tool_calls"][0]["args"] == {"ticker": "AAPL"}


def test_stream_chunk_parser_builds_tool_call_from_codex_events():
    llm = ChatCodexCLI(model="gpt-5.4")

    start_chunk = llm._chunk_from_stream_line(
        json.dumps(
            {
                "type": "response.output_item.added",
                "item": {
                    "type": "function_call",
                    "name": "lookup_price",
                    "arguments": '{"ticker":',
                    "call_id": "call_lookup_1",
                    "id": "item_lookup_1",
                },
                "output_index": 0,
            },
            ensure_ascii=False,
        )
    )
    delta_chunk = llm._chunk_from_stream_line(
        json.dumps(
            {
                "type": "response.function_call_arguments.delta",
                "delta": '"AAPL"}',
                "output_index": 0,
            },
            ensure_ascii=False,
        )
    )

    assert start_chunk is not None
    assert delta_chunk is not None

    result = generate_from_stream(iter([start_chunk, delta_chunk]))
    message = result.generations[0].message

    assert len(message.tool_calls) == 1
    assert message.tool_calls[0]["name"] == "lookup_price"
    assert message.tool_calls[0]["args"] == {"ticker": "AAPL"}


def test_stream_falls_back_to_final_tool_calls(monkeypatch):
    llm = ChatCodexCLI(model="gpt-5.4")

    def fake_stream_codex_exec(
        prompt_text: str,
        *,
        tools,
        tool_choice,
        parallel_tool_calls,
        stream_state,
        run_manager,
    ):
        assert len(tools) == 1
        assert tool_choice is None
        assert parallel_tool_calls is None
        stream_state["saw_text"] = True
        stream_state["raw_response"] = json.dumps(
            {
                "content": "需要先查价格",
                "tool_calls": [
                    {
                        "name": "lookup_price",
                        "args": {"ticker": "TSLA"},
                        "id": "call_tsla",
                    }
                ],
            },
            ensure_ascii=False,
        )
        yield ChatGenerationChunk(message=AIMessageChunk(content="需要先查价格"))

    monkeypatch.setattr(llm, "_stream_codex_exec", fake_stream_codex_exec)

    chunks = list(
        llm._stream(
            [HumanMessage(content="先查一下特斯拉价格")],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "lookup_price",
                        "description": "查询股价",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "ticker": {"type": "string"},
                            },
                        },
                    },
                }
            ],
        )
    )

    result = generate_from_stream(iter(chunks))
    message = result.generations[0].message

    assert message.content == "需要先查价格"
    assert len(message.tool_calls) == 1
    assert message.tool_calls[0]["name"] == "lookup_price"
    assert message.tool_calls[0]["args"] == {"ticker": "TSLA"}


def test_astream_falls_back_to_final_tool_calls(monkeypatch):
    llm = ChatCodexCLI(model="gpt-5.4")

    async def fake_astream_codex_exec(
        prompt_text: str,
        *,
        tools,
        tool_choice,
        parallel_tool_calls,
        stream_state,
        run_manager,
    ):
        assert len(tools) == 1
        assert tool_choice is None
        assert parallel_tool_calls is None
        stream_state["saw_text"] = True
        stream_state["raw_response"] = json.dumps(
            {
                "content": "异步先查价格",
                "tool_calls": [
                    {
                        "name": "lookup_price",
                        "args": {"ticker": "META"},
                        "id": "call_meta",
                    }
                ],
            },
            ensure_ascii=False,
        )
        yield ChatGenerationChunk(message=AIMessageChunk(content="异步先查价格"))

    monkeypatch.setattr(llm, "_astream_codex_exec", fake_astream_codex_exec)

    async def collect_chunks():
        return [
            chunk
            async for chunk in llm._astream(
                [HumanMessage(content="异步先查一下 Meta 价格")],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup_price",
                            "description": "查询股价",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "ticker": {"type": "string"},
                                },
                            },
                        },
                    }
                ],
            )
        ]

    chunks = asyncio.run(collect_chunks())
    result = generate_from_stream(iter(chunks))
    message = result.generations[0].message

    assert message.content == "异步先查价格"
    assert len(message.tool_calls) == 1
    assert message.tool_calls[0]["name"] == "lookup_price"
    assert message.tool_calls[0]["args"] == {"ticker": "META"}


def test_default_model_catalog_contains_only_gpt_5_4_for_codex():
    service = ConfigService()

    catalogs = service._get_default_model_catalog()
    codex_catalog = next(
        catalog for catalog in catalogs if catalog["provider"] == "codex"
    )

    assert codex_catalog["provider_name"] == "Codex CLI"
    assert [model["name"] for model in codex_catalog["models"]] == ["gpt-5.4"]
