from tradingagents.utils.codex_session_metadata import (
    extract_codex_role_sessions,
    get_related_codex_sessions,
)


def test_extract_codex_role_sessions_prefers_direct_field():
    payload = {
        "codex_role_sessions": {
            "Bull Researcher": "thread_bull",
            "Bear Researcher": "thread_bear",
            "invalid": "",
        },
        "state": {
            "codex_role_sessions": {
                "Bull Researcher": "thread_old",
            }
        },
    }

    assert extract_codex_role_sessions(payload) == {
        "Bull Researcher": "thread_bull",
        "Bear Researcher": "thread_bear",
    }


def test_extract_codex_role_sessions_falls_back_to_state():
    payload = {
        "state": {
            "codex_role_sessions": {
                "Risky Analyst": "thread_risky",
                "Neutral Analyst": "thread_neutral",
            }
        }
    }

    assert extract_codex_role_sessions(payload) == {
        "Risky Analyst": "thread_risky",
        "Neutral Analyst": "thread_neutral",
    }


def test_get_related_codex_sessions_filters_by_report_mapping():
    codex_role_sessions = {
        "Bull Researcher": "thread_bull",
        "Bear Researcher": "thread_bear",
        "Neutral Analyst": "thread_neutral",
    }

    assert get_related_codex_sessions("investment_debate_state", codex_role_sessions) == [
        ("多头研究员", "thread_bull"),
        ("空头研究员", "thread_bear"),
    ]
    assert get_related_codex_sessions("market_report", codex_role_sessions) == []
