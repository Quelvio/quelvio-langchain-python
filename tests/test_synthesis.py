"""Unit tests for ``synthesize_answer`` and ``asynthesize_answer``."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from quelvio_langchain import (
    AsyncQuelvioClient,
    QuelvioClient,
    SynthesizedAnswer,
    asynthesize_answer,
    synthesize_answer,
)


def test_synthesize_answer_returns_dataclass(
    api_key: str, base_url: str, query_response_payload: dict[str, Any]
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=query_response_payload)

    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport)
    client = QuelvioClient(api_key=api_key, base_url=base_url, http_client=http_client)

    result = synthesize_answer("what's our refund policy?", client=client)
    assert isinstance(result, SynthesizedAnswer)
    assert result.answer is not None
    assert "Refunds are processed" in result.answer
    assert len(result.sources) == 2
    assert result.query_id == "q_01HW9X3J7K8N0V4P2QXYZA"


@pytest.mark.asyncio
async def test_asynthesize_answer(
    api_key: str, base_url: str, query_response_payload: dict[str, Any]
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=query_response_payload)

    transport = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=transport)
    client = AsyncQuelvioClient(api_key=api_key, base_url=base_url, http_client=http_client)
    result = await asynthesize_answer("hi", client=client)
    assert result.query_id == "q_01HW9X3J7K8N0V4P2QXYZA"
    assert len(result.sources) == 2
    await client.aclose()
