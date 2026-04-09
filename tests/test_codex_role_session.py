from __future__ import annotations

from typing import Any, Dict, List, Tuple

from langchain_core.messages import AIMessage

from tradingagents.agents.researchers.bull_researcher import create_bull_researcher
from tradingagents.agents.utils.codex_session import (
    build_codex_session_event,
    build_codex_invoke_kwargs,
    build_invoke_kwargs,
    get_latest_codex_session,
    merge_codex_session_event,
)


class FakeCodexLLM:
    _llm_type = "codex_cli"

    def __init__(self) -> None:
        self.calls: List[Tuple[str, Dict[str, Any]]] = []
        self._next_session_id = "thread_default"

    def invoke(self, prompt: str, **kwargs: Any) -> AIMessage:
        self.calls.append((prompt, dict(kwargs)))
        return AIMessage(
            content="ok",
            response_metadata={"codex_session_id": self._next_session_id},
        )


def test_build_codex_invoke_kwargs_reuses_latest_session():
    invoke_kwargs = build_codex_invoke_kwargs(
        {
            "task_id": "task_1",
            "codex_role_sessions": {"Bull Researcher": ["thread_old", "thread_latest"]},
        },
        "Bull Researcher",
    )

    assert invoke_kwargs == {
        "analysis_type": "stock_analysis",
        "task_id": "task_1",
        "resume_session_id": "thread_latest",
    }
    assert get_latest_codex_session(
        {"codex_role_sessions": {"Bull Researcher": ["thread_old", "thread_latest"]}},
        "Bull Researcher",
    ) == "thread_latest"


def test_build_invoke_kwargs_returns_empty_for_non_codex():
    class FakeNonCodexLLM:
        _llm_type = "other"

    assert build_invoke_kwargs(
        FakeNonCodexLLM(),
        {"task_id": "task_1", "codex_role_sessions": {"Bull Researcher": ["thread_latest"]}},
        "Bull Researcher",
    ) == {}


def test_merge_codex_session_event_appends_and_dedupes_tail():
    response = AIMessage(
        content="ok",
        response_metadata={"codex_session_id": "thread_fresh"},
    )
    event = build_codex_session_event("Bull Researcher", response)

    updated_sessions = merge_codex_session_event(
        {"Bull Researcher": ["thread_old"]},
        event,
    )
    assert updated_sessions == {"Bull Researcher": ["thread_old", "thread_fresh"]}

    deduped_sessions = merge_codex_session_event(
        updated_sessions,
        event,
    )
    assert deduped_sessions == {"Bull Researcher": ["thread_old", "thread_fresh"]}


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
                response_metadata={"codex_session_id": session_id},
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
    assert first_result["codex_session"] == {
        "role": "Bull Researcher",
        "codex_session_id": "bull_thread_1",
    }
    assert "市场研究报告" in llm.calls[0][0]
    assert "resume_session_id" not in llm.calls[0][1]

    second_state = dict(base_state)
    second_state["codex_role_sessions"] = {"Bull Researcher": ["bull_thread_1"]}
    second_state["investment_debate_state"] = {
        "history": "Bull Analyst: reply_1\nBear Analyst: latest rebuttal",
        "bull_history": "Bull Analyst: reply_1",
        "bear_history": "Bear Analyst: latest rebuttal",
        "current_response": "Bear Analyst: latest rebuttal",
        "count": 1,
    }

    second_result = node(second_state)
    assert second_result["codex_session"] == {
        "role": "Bull Researcher",
        "codex_session_id": "bull_thread_1",
    }
    assert "最新看跌论点" in llm.calls[1][0]
    assert "市场研究报告" not in llm.calls[1][0]
    assert llm.calls[1][1]["resume_session_id"] == "bull_thread_1"
