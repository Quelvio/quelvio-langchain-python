"""One-shot helpers for getting a synthesized answer from Quelvio.

These are the lowest-ceremony entry points: useful when you want a single
function call that returns a final answer + citations, without wiring up
a retriever or an agent. Both a sync and an async variant are provided.
"""

from __future__ import annotations

from dataclasses import dataclass

from .client import AsyncQuelvioClient, QuelvioClient
from .types import ChunkResult, QueryResponse


@dataclass(frozen=True)
class SynthesizedAnswer:
    """The return value of :func:`synthesize_answer`.

    Attributes:
        answer: The synthesized natural-language answer, or ``None`` if
            the retrieval mode did not produce one (e.g. ``mode="fast"``).
        sources: The chunks that informed the answer, in rank order.
        query_id: Server-side identifier — pass to
            :meth:`QuelvioClient.get_source_detail` to resolve per-chunk
            provenance later.
    """

    answer: str | None
    sources: list[ChunkResult]
    query_id: str

    @classmethod
    def from_response(cls, response: QueryResponse) -> SynthesizedAnswer:
        return cls(
            answer=response.synthesis,
            sources=list(response.results),
            query_id=response.query_id,
        )


def synthesize_answer(
    question: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    mode: str = "standard",
    max_sources: int = 5,
    domain_filter: str | None = None,
    timeout: float = 30.0,
    client: QuelvioClient | None = None,
) -> SynthesizedAnswer:
    """Synchronously ask Quelvio a question and return a synthesized answer.

    Args:
        question: The natural-language question to ask.
        api_key: Bearer token; falls back to ``QUELVIO_API_KEY``.
        base_url: Override the API base URL.
        mode: ``fast`` | ``standard`` | ``deep``.
        max_sources: Maximum number of chunks to retrieve (1 to 50).
        domain_filter: Restrict retrieval to a taxonomy domain.
        timeout: Per-request timeout in seconds.
        client: Reuse an existing :class:`QuelvioClient`.

    Returns:
        A :class:`SynthesizedAnswer` holding the answer + cited sources.
    """
    own_client = client is None
    qclient = client or QuelvioClient(api_key=api_key, base_url=base_url, timeout=timeout)
    try:
        response = qclient.query(
            query=question,
            limit=max_sources,
            mode=mode,
            domain_filter=domain_filter,
        )
        return SynthesizedAnswer.from_response(response)
    finally:
        if own_client:
            qclient.close()


async def asynthesize_answer(
    question: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    mode: str = "standard",
    max_sources: int = 5,
    domain_filter: str | None = None,
    timeout: float = 30.0,
    client: AsyncQuelvioClient | None = None,
) -> SynthesizedAnswer:
    """Asynchronously ask Quelvio a question. See :func:`synthesize_answer`."""
    own_client = client is None
    qclient = client or AsyncQuelvioClient(api_key=api_key, base_url=base_url, timeout=timeout)
    try:
        response = await qclient.query(
            query=question,
            limit=max_sources,
            mode=mode,
            domain_filter=domain_filter,
        )
        return SynthesizedAnswer.from_response(response)
    finally:
        if own_client:
            await qclient.aclose()
