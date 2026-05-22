"""``QuelvioRetriever`` — a LangChain ``BaseRetriever`` backed by Quelvio.

Drop this into any LangChain RAG chain. Each call to ``invoke()`` makes
exactly one HTTP request to ``POST /v1/enterprise/query`` and converts the
returned chunks to LangChain ``Document`` objects, preserving per-chunk
provenance (source URL, authority score, taxonomy domain, chunk id) on
the ``metadata`` dict.
"""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field, PrivateAttr

from .client import AsyncQuelvioClient, QuelvioClient
from .types import ChunkResult, QueryResponse


def _chunk_to_document(chunk: ChunkResult) -> Document:
    """Convert a Quelvio chunk to a LangChain ``Document``."""
    metadata: dict[str, Any] = {
        "chunk_id": chunk.chunk_id,
        "content_piece_id": chunk.content_piece_id,
        "title": chunk.title,
        "score": chunk.score,
        "rank": chunk.rank,
    }
    if chunk.authority_score is not None:
        metadata["authority_score"] = chunk.authority_score
    if chunk.taxonomy_domain is not None:
        metadata["taxonomy_domain"] = chunk.taxonomy_domain
    if chunk.source_url is not None:
        metadata["source_url"] = chunk.source_url
        metadata["source"] = chunk.source_url
    if chunk.author_name is not None:
        metadata["author_name"] = chunk.author_name
    if chunk.author_email is not None:
        metadata["author_email"] = chunk.author_email
    if chunk.department is not None:
        metadata["department"] = chunk.department
    return Document(page_content=chunk.excerpt, metadata=metadata)


def _response_to_documents(response: QueryResponse) -> list[Document]:
    docs = [_chunk_to_document(c) for c in response.results]
    if docs:
        # Stash the query_id on the first document so callers can re-resolve
        # provenance later via ``QuelvioClient.get_source_detail``.
        docs[0].metadata["query_id"] = response.query_id
    return docs


class QuelvioRetriever(BaseRetriever):
    """LangChain retriever backed by the Quelvio enterprise knowledge API.

    Example:
        >>> from quelvio_langchain import QuelvioRetriever
        >>> retriever = QuelvioRetriever(api_key="qlv_pat_...")
        >>> docs = retriever.invoke("what's our refund policy?")

    Args:
        api_key: Quelvio Personal Access Token, OAuth access token, or
            Service Account key. Falls back to the ``QUELVIO_API_KEY``
            environment variable when omitted.
        base_url: Override the API base URL. Defaults to
            ``QUELVIO_API_BASE`` env var or ``https://api.quelvio.com``.
        limit: Maximum number of chunks to retrieve (1 to 50, default 5).
        mode: Synthesis mode - ``fast``, ``standard`` (default), or ``deep``.
        domain_filter: Restrict retrieval to a single taxonomy domain.
        timeout: Per-request timeout in seconds (default 30).
        client: Optionally inject a pre-built :class:`QuelvioClient` for
            connection reuse. When provided, ``api_key`` / ``base_url`` /
            ``timeout`` arguments are ignored.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    api_key: str | None = Field(default=None, exclude=True, repr=False)
    base_url: str | None = None
    limit: int = 5
    mode: str = "standard"
    domain_filter: str | None = None
    timeout: float = 30.0

    _client: QuelvioClient = PrivateAttr()
    _async_client: AsyncQuelvioClient | None = PrivateAttr(default=None)

    def __init__(
        self,
        *,
        client: QuelvioClient | None = None,
        async_client: AsyncQuelvioClient | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if client is not None:
            self._client = client
        else:
            self._client = QuelvioClient(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
        self._async_client = async_client
        # Defensive: scrub the constructor-provided key off the model so it
        # never appears in serialization / repr output.
        object.__setattr__(self, "api_key", None)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")
        response = self._client.query(
            query=query,
            limit=self.limit,
            mode=self.mode,
            domain_filter=self.domain_filter,
        )
        return _response_to_documents(response)

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> list[Document]:
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")
        if self._async_client is None:
            # Reuse the already-resolved key from the sync client so callers
            # who passed ``api_key=...`` to the constructor don't also need
            # QUELVIO_API_KEY set in the environment.
            self._async_client = AsyncQuelvioClient(
                api_key=self._client._api_key,
                base_url=self._client.base_url,
                timeout=self._client.timeout,
            )
        response = await self._async_client.query(
            query=query,
            limit=self.limit,
            mode=self.mode,
            domain_filter=self.domain_filter,
        )
        return _response_to_documents(response)
