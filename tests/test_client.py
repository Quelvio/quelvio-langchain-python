"""Unit tests for ``QuelvioClient`` and ``AsyncQuelvioClient``."""

from __future__ import annotations

import logging
from typing import Any

import httpx
import pytest

from quelvio_langchain import (
    AsyncQuelvioClient,
    QuelvioAuthError,
    QuelvioBadRequestError,
    QuelvioClient,
    QuelvioNotFoundError,
    QuelvioRateLimitError,
    QuelvioServerError,
    QuelvioTimeoutError,
)
from quelvio_langchain.client import _build_query_body, _normalize_mode

# ── Configuration & header behavior ────────────────────────────────────────


def test_missing_api_key_raises_auth_error() -> None:
    with pytest.raises(QuelvioAuthError):
        QuelvioClient()


def test_api_key_resolved_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QUELVIO_API_KEY", "qlv_pat_from_env_XYZ")
    client = QuelvioClient()
    headers = client._headers()
    assert headers["Authorization"] == "Bearer qlv_pat_from_env_XYZ"
    client.close()


def test_base_url_resolved_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QUELVIO_API_BASE", "https://api-dev.quelvio.com/")
    client = QuelvioClient(api_key="qlv_pat_x")
    # Trailing slash is stripped.
    assert client.base_url == "https://api-dev.quelvio.com"
    client.close()


def test_repr_does_not_leak_api_key(api_key: str, base_url: str) -> None:
    client = QuelvioClient(api_key=api_key, base_url=base_url)
    rendered = repr(client)
    assert api_key not in rendered
    assert "Bearer" not in rendered
    assert "QuelvioClient(" in rendered
    client.close()


def test_api_key_never_logged(
    api_key: str,
    base_url: str,
    query_response_payload: dict[str, Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify the API key never appears in any log line emitted while
    issuing a request, regardless of log level."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=query_response_payload)

    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport)
    client = QuelvioClient(api_key=api_key, base_url=base_url, http_client=http_client)

    with caplog.at_level(logging.DEBUG):
        client.query(query="what's our refund policy?")

    full_log = "\n".join(record.getMessage() for record in caplog.records)
    assert api_key not in full_log


# ── Request shaping ────────────────────────────────────────────────────────


def test_query_body_includes_normalized_mode_and_limit() -> None:
    body = _build_query_body(
        query="hi",
        limit=999,  # will be clamped to 50
        mode="DEEP",  # will be lowered
        domain_filter="legal",
    )
    assert body == {
        "query": "hi",
        "limit": 50,
        "mode": "deep",
        "domain_filter": "legal",
    }


def test_query_body_omits_domain_filter_when_none() -> None:
    body = _build_query_body(query="hi", limit=None, mode=None, domain_filter=None)
    assert "domain_filter" not in body
    assert body["limit"] == 5
    assert body["mode"] == "standard"


def test_mode_normalization_rejects_invalid_value() -> None:
    with pytest.raises(QuelvioBadRequestError):
        _normalize_mode("turbo")


def test_empty_query_raises_value_error_before_http(api_key: str, base_url: str) -> None:
    requested_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        return httpx.Response(200, json={})

    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport)
    client = QuelvioClient(api_key=api_key, base_url=base_url, http_client=http_client)

    with pytest.raises(ValueError):
        client.query(query="")
    with pytest.raises(ValueError):
        client.query(query="   ")
    assert requested_paths == []


def test_domain_filter_is_passed_in_body(
    api_key: str, base_url: str, query_response_payload: dict[str, Any]
) -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["body"] = json.loads(request.content)
        captured["path"] = request.url.path
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json=query_response_payload)

    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport)
    client = QuelvioClient(api_key=api_key, base_url=base_url, http_client=http_client)

    client.query(query="who owns finance?", mode="fast", domain_filter="finance")

    assert captured["path"] == "/v1/enterprise/query"
    assert captured["body"]["domain_filter"] == "finance"
    assert captured["body"]["mode"] == "fast"
    assert captured["body"]["query"] == "who owns finance?"
    assert captured["auth"] == f"Bearer {api_key}"


# ── Error mapping ──────────────────────────────────────────────────────────


def _client_returning(status: int, body: Any, base_url: str, api_key: str) -> QuelvioClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if isinstance(body, str):
            return httpx.Response(status, text=body)
        return httpx.Response(status, json=body)

    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport)
    return QuelvioClient(api_key=api_key, base_url=base_url, http_client=http_client, max_retries=0)


def test_401_raises_auth_error(api_key: str, base_url: str) -> None:
    client = _client_returning(401, {"detail": "bad token"}, base_url, api_key)
    with pytest.raises(QuelvioAuthError):
        client.query(query="x")


def test_404_raises_not_found(api_key: str, base_url: str) -> None:
    client = _client_returning(404, {"detail": "missing"}, base_url, api_key)
    with pytest.raises(QuelvioNotFoundError):
        client.get_source_detail("q_does_not_exist")


def test_429_raises_rate_limit_error_with_retry_after(api_key: str, base_url: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={"detail": "slow down"},
            headers={"retry-after": "17"},
        )

    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport)
    client = QuelvioClient(
        api_key=api_key, base_url=base_url, http_client=http_client, max_retries=0
    )

    with pytest.raises(QuelvioRateLimitError) as exc_info:
        client.query(query="x")
    assert exc_info.value.retry_after_seconds == 17


def test_500_raises_server_error_after_retries_exhausted(api_key: str, base_url: str) -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(503, text="upstream down")

    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport)
    # Override sleep to make the test instant.
    client = QuelvioClient(
        api_key=api_key, base_url=base_url, http_client=http_client, max_retries=2
    )

    import quelvio_langchain.client as client_module

    original_sleep = client_module.time.sleep
    client_module.time.sleep = lambda _s: None
    try:
        with pytest.raises(QuelvioServerError) as exc_info:
            client.query(query="x")
    finally:
        client_module.time.sleep = original_sleep

    assert exc_info.value.status_code == 503
    assert calls["count"] == 3  # initial + 2 retries


def test_timeout_raises_quelvio_timeout(api_key: str, base_url: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("simulated")

    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport)
    client = QuelvioClient(
        api_key=api_key, base_url=base_url, http_client=http_client, max_retries=0
    )

    with pytest.raises(QuelvioTimeoutError):
        client.query(query="x")


# ── Response parsing ───────────────────────────────────────────────────────


def test_list_domains_parses_response(api_key: str, base_url: str) -> None:
    payload = {
        "total": 1,
        "domains": [
            {
                "taxonomy_domain": "engineering",
                "document_count": 42,
                "chunk_count": 412,
                "expert_count": 5,
                "coverage_level": "complete",
            }
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/enterprise/domains"
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport)
    client = QuelvioClient(api_key=api_key, base_url=base_url, http_client=http_client)

    result = client.list_domains()
    assert result.total == 1
    assert result.domains[0].taxonomy_domain == "engineering"
    assert result.domains[0].document_count == 42


def test_query_returns_typed_response(
    api_key: str, base_url: str, query_response_payload: dict[str, Any]
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=query_response_payload)

    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport)
    client = QuelvioClient(api_key=api_key, base_url=base_url, http_client=http_client)

    response = client.query(query="what's our refund policy?")
    assert response.query_id == "q_01HW9X3J7K8N0V4P2QXYZA"
    assert response.synthesis is not None
    assert len(response.results) == 2
    assert response.results[0].title == "Refund Policy v3"
    assert response.results[0].authority_score == 0.87


def test_extra_response_fields_are_ignored(
    api_key: str, base_url: str, query_response_payload: dict[str, Any]
) -> None:
    """Forward compatibility: the API can add new fields without breaking us."""
    payload = {
        **query_response_payload,
        "future_field": "ignored",
        "results": [
            {**query_response_payload["results"][0], "new_chunk_field": 99},
        ],
        "result_count": 1,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport)
    client = QuelvioClient(api_key=api_key, base_url=base_url, http_client=http_client)

    response = client.query(query="x")
    assert response.results[0].chunk_id == "chunk_001"


# ── Async client ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_async_client_basic_query(
    api_key: str, base_url: str, query_response_payload: dict[str, Any]
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=query_response_payload)

    transport = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=transport)
    client = AsyncQuelvioClient(api_key=api_key, base_url=base_url, http_client=http_client)
    response = await client.query(query="hi")
    assert response.query_id == "q_01HW9X3J7K8N0V4P2QXYZA"
    assert len(response.results) == 2
    await client.aclose()


@pytest.mark.asyncio
async def test_async_401_raises_auth_error(api_key: str, base_url: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "bad token"})

    transport = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=transport)
    client = AsyncQuelvioClient(
        api_key=api_key, base_url=base_url, http_client=http_client, max_retries=0
    )
    with pytest.raises(QuelvioAuthError):
        await client.query(query="hi")
    await client.aclose()
