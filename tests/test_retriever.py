"""Unit tests for ``QuelvioRetriever``."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from langchain_core.documents import Document

from quelvio_langchain import (
    AsyncQuelvioClient,
    QuelvioAuthError,
    QuelvioClient,
    QuelvioRetriever,
)


def _make_retriever(
    api_key: str,
    base_url: str,
    payload: dict[str, Any],
    **kwargs: Any,
) -> tuple[QuelvioRetriever, dict[str, Any]]:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content) if request.content else None
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport)
    client = QuelvioClient(api_key=api_key, base_url=base_url, http_client=http_client)
    return QuelvioRetriever(client=client, **kwargs), captured


def test_retriever_returns_documents_with_metadata(
    api_key: str, base_url: str, query_response_payload: dict[str, Any]
) -> None:
    retriever, _ = _make_retriever(api_key, base_url, query_response_payload)
    docs = retriever.invoke("what's our refund policy?")

    assert len(docs) == 2
    assert all(isinstance(d, Document) for d in docs)

    first = docs[0]
    assert first.page_content.startswith("All paid customers")
    assert first.metadata["chunk_id"] == "chunk_001"
    assert first.metadata["title"] == "Refund Policy v3"
    assert first.metadata["authority_score"] == 0.87
    assert first.metadata["taxonomy_domain"] == "finance"
    assert first.metadata["source_url"] == "https://drive.example/refund-policy-v3"
    assert first.metadata["source"] == "https://drive.example/refund-policy-v3"
    # query_id stashed on the first document for downstream provenance lookups
    assert first.metadata["query_id"] == "q_01HW9X3J7K8N0V4P2QXYZA"


def test_retriever_passes_mode_and_domain_filter(
    api_key: str, base_url: str, query_response_payload: dict[str, Any]
) -> None:
    retriever, captured = _make_retriever(
        api_key,
        base_url,
        query_response_payload,
        mode="deep",
        domain_filter="engineering",
        limit=10,
    )
    retriever.invoke("what's our deploy process?")
    assert captured["body"]["mode"] == "deep"
    assert captured["body"]["domain_filter"] == "engineering"
    assert captured["body"]["limit"] == 10


def test_retriever_empty_query_raises_before_http(
    api_key: str, base_url: str, query_response_payload: dict[str, Any]
) -> None:
    captured: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request.url.path)
        return httpx.Response(200, json=query_response_payload)

    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport)
    client = QuelvioClient(api_key=api_key, base_url=base_url, http_client=http_client)
    retriever = QuelvioRetriever(client=client)

    with pytest.raises(ValueError):
        retriever.invoke("")
    assert captured == []


def test_retriever_repr_does_not_leak_api_key(api_key: str, base_url: str) -> None:
    retriever = QuelvioRetriever(api_key=api_key, base_url=base_url)
    rendered = repr(retriever)
    assert api_key not in rendered


def test_retriever_propagates_auth_error(api_key: str, base_url: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "no"})

    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport)
    client = QuelvioClient(
        api_key=api_key, base_url=base_url, http_client=http_client, max_retries=0
    )
    retriever = QuelvioRetriever(client=client)
    with pytest.raises(QuelvioAuthError):
        retriever.invoke("hi")


@pytest.mark.asyncio
async def test_retriever_async_invoke(
    api_key: str, base_url: str, query_response_payload: dict[str, Any]
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=query_response_payload)

    sync_transport = httpx.MockTransport(handler)
    async_transport = httpx.MockTransport(handler)
    sync_http = httpx.Client(transport=sync_transport)
    async_http = httpx.AsyncClient(transport=async_transport)

    sync_client = QuelvioClient(api_key=api_key, base_url=base_url, http_client=sync_http)
    async_client = AsyncQuelvioClient(api_key=api_key, base_url=base_url, http_client=async_http)

    retriever = QuelvioRetriever(client=sync_client, async_client=async_client)
    docs = await retriever.ainvoke("what's our refund policy?")
    assert len(docs) == 2
    assert docs[0].metadata["chunk_id"] == "chunk_001"
