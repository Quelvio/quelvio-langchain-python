"""Pydantic models for Quelvio API request and response payloads.

These mirror the wire format documented at
``https://api.quelvio.com/openapi.json``. ``model_config`` is set to
``extra="ignore"`` everywhere so that additions to the API (new fields
on a chunk, a new top-level response key) do not break older client
versions.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

QueryMode = Literal["fast", "standard", "deep"]
"""Synthesis mode. ``standard`` is the default and includes a synthesized
answer; ``fast`` skips synthesis for lowest latency; ``deep`` enables
multi-pass reasoning over a wider retrieval window."""


class QueryRequest(BaseModel):
    """Body for ``POST /v1/enterprise/query``."""

    model_config = ConfigDict(extra="forbid")

    query: str
    limit: int = Field(default=5, ge=1, le=50)
    mode: QueryMode = "standard"
    domain_filter: str | None = None


class ChunkResult(BaseModel):
    """A single retrieved chunk in a :class:`QueryResponse`."""

    model_config = ConfigDict(extra="ignore")

    chunk_id: str
    content_piece_id: str
    title: str
    excerpt: str
    score: float
    rank: int
    authority_score: float | None = None
    taxonomy_domain: str | None = None
    source_url: str | None = None
    creator_id: str | None = None
    author_name: str | None = None
    author_email: str | None = None
    department: str | None = None


class QueryResponse(BaseModel):
    """Response from ``POST /v1/enterprise/query``."""

    model_config = ConfigDict(extra="ignore")

    query: str
    query_id: str
    results: list[ChunkResult] = Field(default_factory=list)
    result_count: int = 0
    coverage: str | None = None
    risk_flag: dict[str, bool] = Field(default_factory=dict)
    retrieval_mode: str | None = None
    synthesis: str | None = None
    synthesis_model: str | None = None
    latency_ms: int | None = None
    tokens_consumed: int | None = None


class DomainCoverage(BaseModel):
    """A single entry in :class:`DomainsListResponse`."""

    model_config = ConfigDict(extra="ignore")

    taxonomy_domain: str
    document_count: int = 0
    chunk_count: int = 0
    expert_count: int = 0
    coverage_level: str | None = None


class DomainsListResponse(BaseModel):
    """Response from ``GET /v1/enterprise/domains``."""

    model_config = ConfigDict(extra="ignore")

    domains: list[DomainCoverage] = Field(default_factory=list)
    total: int = 0


class SourceChunk(BaseModel):
    """A chunk's provenance record in :class:`SourceDetailResponse`."""

    model_config = ConfigDict(extra="ignore")

    chunk_id: str
    content_piece_id: str
    title: str
    excerpt: str
    source_url: str | None = None
    source_type: str | None = None
    lifecycle_state: str | None = None
    embedded_at: str | None = None
    last_source_updated_at: str | None = None
    authority_score: float | None = None
    taxonomy_domain: str | None = None
    author_name: str | None = None
    author_email: str | None = None


class SourceDetailResponse(BaseModel):
    """Response from ``GET /v1/enterprise/sources/{query_id}``."""

    model_config = ConfigDict(extra="ignore")

    query_id: str
    tenant_id: str | None = None
    chunks: list[SourceChunk] = Field(default_factory=list)
    chunk_count: int = 0
