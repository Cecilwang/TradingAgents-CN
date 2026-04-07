from app.utils.error_formatter import ErrorFormatter


def test_codex_invalid_json_schema_is_not_reported_as_api_key_error():
    error_message = (
        "Codex CLI 调用失败：returncode=1, stderr=invalid_json_schema "
        "response_format codex_output_schema Missing 'start_date'."
    )

    formatted = ErrorFormatter.format_error(
        error_message,
        {"llm_provider": "codex"},
    )

    assert formatted["category"] == "大模型调用错误"
    assert "调用失败" in formatted["title"]
    assert "API Key 无效" not in formatted["title"]
