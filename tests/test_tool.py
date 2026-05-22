"""Unit tests for ``QuelvioTool``."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from quelvio_langchain import QuelvioClient, QuelvioTool, QuelvioToolInput


def _make_tool(
    api_key: str,
    base_url: str,
    payload: dict[str, Any],
    **kwargs: Any,
) -> tuple[QuelvioTool, dict[str, Any]]:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content) if request.content else None
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport)
    client = QuelvioClient(api_key=api_key, base_url=base_url, http_client=http_client)
    return QuelvioTool(client=client, **kwargs), captured


def test_tool_metadata_defaults() -> None:
    tool = QuelvioTool(api_key="qlv_pat_x")
    assert tool.name == "quelvio_query"
    assert "knowledge brain" in tool.description.lower()
    assert tool.args_schema is QuelvioToolInput


def test_tool_args_schema_handles_missing_optional_fields() -> None:
    parsed = QuelvioToolInput.model_validate({"question": "hi"})
    assert parsed.question == "hi"
    assert parsed.mode is None
    assert parsed.max_sources is None
    assert parsed.domain is None


def test_tool_args_schema_bounds_max_sources() -> None:
    with pytest.raises(ValueError):
        QuelvioToolInput.model_validate({"question": "hi", "max_sources": 0})
    with pytest.raises(ValueError):
        QuelvioToolInput.model_validate({"question": "hi", "max_sources": 51})


def test_tool_returns_formatted_string_with_sources(
    api_key: str, base_url: str, query_response_payload: dict[str, Any]
) -> None:
    tool, captured = _make_tool(api_key, base_url, query_response_payload)
    result = tool.invoke({"question": "what's our refund policy?"})

    assert "Refunds are processed within 14 days" in result
    assert "Sources:" in result
    assert "[1] Refund Policy v3" in result
    assert "https://drive.example/refund-policy-v3" in result

    # And we hit the right path with the right body.
    assert captured["path"] == "/v1/enterprise/query"
    assert captured["body"]["query"] == "what's our refund policy?"


def test_tool_passes_optional_fields_to_request(
    api_key: str, base_url: str, query_response_payload: dict[str, Any]
) -> None:
    tool, captured = _make_tool(api_key, base_url, query_response_payload)
    tool.invoke(
        {
            "question": "who owns finance?",
            "mode": "fast",
            "max_sources": 3,
            "domain": "finance",
        }
    )
    assert captured["body"]["mode"] == "fast"
    assert captured["body"]["limit"] == 3
    assert captured["body"]["domain_filter"] == "finance"


def test_tool_empty_question_raises(
    api_key: str, base_url: str, query_response_payload: dict[str, Any]
) -> None:
    tool, captured = _make_tool(api_key, base_url, query_response_payload)
    with pytest.raises(ValueError):
        tool.invoke({"question": "   "})
    assert "path" not in captured


def test_tool_repr_does_not_leak_api_key(api_key: str) -> None:
    tool = QuelvioTool(api_key=api_key)
    rendered = repr(tool)
    assert api_key not in rendered


def test_tool_no_results_returns_friendly_message(api_key: str, base_url: str) -> None:
    empty_payload = {
        "query": "?",
        "query_id": "q_empty",
        "result_count": 0,
        "risk_flag": {},
        "results": [],
        "synthesis": None,
    }
    tool, _ = _make_tool(api_key, base_url, empty_payload)
    result = tool.invoke({"question": "anything"})
    assert "No matching content" in result
