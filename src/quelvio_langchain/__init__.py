"""Quelvio for LangChain — your company's brain as a retriever and a tool."""

from __future__ import annotations

from ._version import __version__
from .client import AsyncQuelvioClient, QuelvioClient
from .exceptions import (
    QuelvioAuthError,
    QuelvioBadRequestError,
    QuelvioError,
    QuelvioNetworkError,
    QuelvioNotFoundError,
    QuelvioRateLimitError,
    QuelvioServerError,
    QuelvioTimeoutError,
)
from .retriever import QuelvioRetriever
from .synthesis import SynthesizedAnswer, asynthesize_answer, synthesize_answer
from .tool import QuelvioTool, QuelvioToolInput
from .types import (
    ChunkResult,
    DomainCoverage,
    DomainsListResponse,
    QueryMode,
    QueryRequest,
    QueryResponse,
    SourceChunk,
    SourceDetailResponse,
)

__all__ = [
    "AsyncQuelvioClient",
    "ChunkResult",
    "DomainCoverage",
    "DomainsListResponse",
    "QuelvioAuthError",
    "QuelvioBadRequestError",
    "QuelvioClient",
    "QuelvioError",
    "QuelvioNetworkError",
    "QuelvioNotFoundError",
    "QuelvioRateLimitError",
    "QuelvioRetriever",
    "QuelvioServerError",
    "QuelvioTimeoutError",
    "QuelvioTool",
    "QuelvioToolInput",
    "QueryMode",
    "QueryRequest",
    "QueryResponse",
    "SourceChunk",
    "SourceDetailResponse",
    "SynthesizedAnswer",
    "__version__",
    "asynthesize_answer",
    "synthesize_answer",
]
