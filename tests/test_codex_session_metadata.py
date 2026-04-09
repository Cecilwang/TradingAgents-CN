from tradingagents.utils.codex_session_metadata import (
    extract_codex_role_sessions,
    get_report_codex_sessions,
)


def test_extract_codex_role_sessions_prefers_direct_field():
    payload = {
        "codex_role_sessions": {
            "Bull Researcher": ["thread_bull"],
            "Bear Researcher": ["thread_bear", ""],
            "invalid": [],
        },
        "state": {
            "codex_role_sessions": {
                "Bull Researcher": ["thread_old"],
            }
        },
    }

    assert extract_codex_role_sessions(payload) == {
        "Bull Researcher": ["thread_bull"],
        "Bear Researcher": ["thread_bear"],
    }


def test_extract_codex_role_sessions_falls_back_to_state():
    payload = {
        "state": {
            "codex_role_sessions": {
                "Risky Analyst": ["thread_risky"],
                "Neutral Analyst": ["thread_neutral"],
            }
        }
    }

    assert extract_codex_role_sessions(payload) == {
        "Risky Analyst": ["thread_risky"],
        "Neutral Analyst": ["thread_neutral"],
    }


def test_get_report_codex_sessions_filters_by_single_owner_role():
    codex_role_sessions = {
        "Research Manager": ["thread_manager_1", "thread_manager_2"],
        "Bull Researcher": ["thread_bull"],
        "Bear Researcher": ["thread_bear"],
    }

    assert get_report_codex_sessions(
        "investment_debate_state",
        codex_role_sessions,
    ) == ("研究经理", ["thread_manager_1", "thread_manager_2"])
    assert get_report_codex_sessions("market_report", codex_role_sessions) is None
