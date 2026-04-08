from __future__ import annotations

from typing import Any, Dict, List, Tuple

from langchain_core.messages import AIMessage

from tradingagents.agents.researchers.bull_researcher import create_bull_researcher
from tradingagents.agents.utils.codex_session import invoke_role_with_codex_session


class FakeCodexLLM:
    _llm_type = "codex_cli"

    def __init__(self) -> None:
        self.calls: List[Tuple[str, Dict[str, Any]]] = []
        self._next_session_id = "thread_default"

    def invoke(self, prompt: str, **kwargs: Any) -> AIMessage:
        self.calls.append((prompt, dict(kwargs)))
        return AIMessage(
            content="ok",
            response_metadata={"session_id": self._next_session_id},
        )


def test_invoke_role_with_codex_session_reuses_existing_session():
    llm = FakeCodexLLM()
    llm._next_session_id = "thread_reused"

    response, updated_sessions = invoke_role_with_codex_session(
        llm=llm,
        state={
            "task_id": "task_1",
            "codex_role_sessions": {"Bull Researcher": "thread_old"},
        },
        role_name="Bull Researcher",
        full_prompt="完整提示",
        continuation_prompt="增量提示",
    )

    assert response.content == "ok"
    assert llm.calls == [
        (
            "增量提示",
            {
                "analysis_type": "stock_analysis",
                "session_id": "task_1",
                "resume_session_id": "thread_old",
            },
        )
    ]
    assert updated_sessions == {"Bull Researcher": "thread_reused"}


def test_invoke_role_with_codex_session_rebuilds_after_resume_failure():
    class FlakyCodexLLM(FakeCodexLLM):
        def __init__(self) -> None:
            super().__init__()
            self._next_session_id = "thread_fresh"

        def invoke(self, prompt: str, **kwargs: Any) -> AIMessage:
            self.calls.append((prompt, dict(kwargs)))
            if kwargs.get("resume_session_id"):
                raise RuntimeError("resume failed")
            return AIMessage(
                content="ok",
                response_metadata={"session_id": self._next_session_id},
            )

    llm = FlakyCodexLLM()
    factory_calls = {"count": 0}

    def build_full_prompt() -> str:
        factory_calls["count"] += 1
        return "完整提示"

    response, updated_sessions = invoke_role_with_codex_session(
        llm=llm,
        state={
            "task_id": "task_2",
            "codex_role_sessions": {"Bull Researcher": "thread_broken"},
        },
        role_name="Bull Researcher",
        full_prompt=build_full_prompt,
        continuation_prompt="增量提示",
    )

    assert response.content == "ok"
    assert factory_calls["count"] == 1
    assert llm.calls == [
        (
            "增量提示",
            {
                "analysis_type": "stock_analysis",
                "session_id": "task_2",
                "resume_session_id": "thread_broken",
            },
        ),
        (
            "完整提示",
            {
                "analysis_type": "stock_analysis",
                "session_id": "task_2",
            },
        ),
    ]
    assert updated_sessions == {"Bull Researcher": "thread_fresh"}


def test_bull_researcher_reuses_role_session_between_rounds():
    class SequencedCodexLLM(FakeCodexLLM):
        def __init__(self) -> None:
            super().__init__()
            self._session_ids = ["bull_thread_1", "bull_thread_1"]

        def invoke(self, prompt: str, **kwargs: Any) -> AIMessage:
            self.calls.append((prompt, dict(kwargs)))
            session_id = self._session_ids[len(self.calls) - 1]
            return AIMessage(
                content=f"reply_{len(self.calls)}",
                response_metadata={"session_id": session_id},
            )

    llm = SequencedCodexLLM()
    node = create_bull_researcher(llm, memory=None)

    base_state = {
        "company_of_interest": "AAPL",
        "task_id": "task_bull",
        "codex_role_sessions": {},
        "market_report": "market",
        "sentiment_report": "sentiment",
        "news_report": "news",
        "fundamentals_report": "fundamentals",
        "investment_debate_state": {
            "history": "",
            "bull_history": "",
            "bear_history": "",
            "current_response": "",
            "count": 0,
        },
    }

    first_result = node(base_state)
    assert first_result["codex_role_sessions"] == {"Bull Researcher": "bull_thread_1"}
    assert "市场研究报告" in llm.calls[0][0]
    assert "resume_session_id" not in llm.calls[0][1]

    second_state = dict(base_state)
    second_state["codex_role_sessions"] = first_result["codex_role_sessions"]
    second_state["investment_debate_state"] = {
        "history": "Bull Analyst: reply_1\nBear Analyst: latest rebuttal",
        "bull_history": "Bull Analyst: reply_1",
        "bear_history": "Bear Analyst: latest rebuttal",
        "current_response": "Bear Analyst: latest rebuttal",
        "count": 1,
    }

    second_result = node(second_state)
    assert second_result["codex_role_sessions"] == {"Bull Researcher": "bull_thread_1"}
    assert "最新看跌论点" in llm.calls[1][0]
    assert "市场研究报告" not in llm.calls[1][0]
    assert llm.calls[1][1]["resume_session_id"] == "bull_thread_1"
