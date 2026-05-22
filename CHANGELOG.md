# Changelog

All notable changes to `quelvio-langchain` will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-05-22

Initial release.

### Added
- `QuelvioRetriever` — `langchain_core.retrievers.BaseRetriever` subclass that
  wraps `POST /v1/enterprise/query`. Implements both sync (`_get_relevant_documents`)
  and async (`_aget_relevant_documents`) and returns `Document` objects with
  `source_url`, `authority_score`, `chunk_id`, `taxonomy_domain`, and
  `lifecycle_state` metadata.
- `QuelvioTool` — `langchain_core.tools.BaseTool` subclass for use inside
  LangChain agents. Returns a synthesized answer plus cited sources.
- `QuelvioClient` — thin httpx-based HTTP client (sync + async) for the
  Quelvio enterprise REST API. Bearer-token auth, `User-Agent` includes the
  package version, retries 502/503/504 with jittered exponential backoff.
- `quelvio_langchain.synthesis.synthesize_answer` helper — one-shot answer
  with citations for non-agent / non-retriever use cases.
- Typed exception hierarchy: `QuelvioError`, `QuelvioAuthError`,
  `QuelvioRateLimitError`, `QuelvioBadRequestError`, `QuelvioServerError`,
  `QuelvioNotFoundError`, `QuelvioTimeoutError`, `QuelvioNetworkError`.
- API key redaction in `repr()`, exception messages, and User-Agent — keys
  never appear in any log line.
- Configuration via `QUELVIO_API_KEY` and `QUELVIO_API_BASE` env vars.
- Pydantic v2 request / response models in `quelvio_langchain.types`.

[0.1.0]: https://github.com/Quelvio/quelvio-langchain-python/releases/tag/v0.1.0
