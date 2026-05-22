"""HTTP client for the Quelvio enterprise REST API.

Thin wrapper around ``httpx`` that exposes sync and async variants of the
three endpoints we care about for LangChain integrations:

* ``POST /v1/enterprise/query``           — retrieval + optional synthesis
* ``GET  /v1/enterprise/domains``         — taxonomy discovery
* ``GET  /v1/enterprise/sources/{id}``    — provenance for a previous query

The bearer token is held privately and is never written to logs, exception
messages, or ``repr()``. Pass it via the ``api_key`` argument or set the
``QUELVIO_API_KEY`` environment variable.
"""

from __future__ import annotations

import os
import platform as _platform
import random
import time
from typing import Any

import httpx

from ._version import __version__
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
from .types import (
    DomainsListResponse,
    QueryMode,
    QueryRequest,
    QueryResponse,
    SourceDetailResponse,
)

DEFAULT_BASE_URL = "https://api.quelvio.com"
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_RETRIES = 3
_BASE_BACKOFF_SECONDS = 1.0
_RETRYABLE_STATUSES = frozenset({502, 503, 504})

VALID_MODES: frozenset[str] = frozenset({"fast", "standard", "deep"})


def _resolve_api_key(explicit: str | None) -> str:
    if explicit:
        return explicit
    env = os.environ.get("QUELVIO_API_KEY")
    if env:
        return env
    raise QuelvioAuthError(
        "No Quelvio API key was provided. Pass `api_key=...` to the constructor "
        "or set the QUELVIO_API_KEY environment variable. Generate a Personal "
        "Access Token at https://enterprise.quelvio.com/account."
    )


def _resolve_base_url(explicit: str | None) -> str:
    candidate = explicit or os.environ.get("QUELVIO_API_BASE") or DEFAULT_BASE_URL
    return candidate.rstrip("/")


def _user_agent() -> str:
    return (
        f"quelvio-langchain/{__version__} "
        f"python/{_platform.python_version()} "
        f"{_platform.system().lower()}-{_platform.machine().lower()}"
    )


def _normalize_mode(mode: str | None) -> QueryMode:
    if mode is None:
        return "standard"
    lowered = mode.strip().lower()
    if lowered not in VALID_MODES:
        raise QuelvioBadRequestError(
            f"Invalid mode {mode!r}. Expected one of: fast, standard, deep."
        )
    return lowered  # type: ignore[return-value]


def _bound_limit(value: int | None) -> int:
    if value is None:
        return 5
    if value < 1:
        return 1
    if value > 50:
        return 50
    return value


def _backoff_seconds(attempt: int, rand: float) -> float:
    """Exponential backoff with +/-20% jitter. ``attempt`` is 0-indexed."""
    base: float = _BASE_BACKOFF_SECONDS * (2**attempt)
    jitter: float = 1.0 + (rand * 0.4 - 0.2)
    return float(base * jitter)


def _map_error(response: httpx.Response) -> QuelvioError:
    """Translate a non-2xx response into the appropriate typed exception."""
    detail = _extract_detail(response)
    status = response.status_code

    if status == 400:
        return QuelvioBadRequestError(detail or f"Bad request: {response.reason_phrase}")
    if status in (401, 403):
        return QuelvioAuthError(
            "Quelvio authentication failed. Your token may be invalid, expired, "
            "or revoked. Generate a new Personal Access Token at "
            "https://enterprise.quelvio.com/account."
        )
    if status == 404:
        return QuelvioNotFoundError(detail or f"Not found: {response.reason_phrase}")
    if status == 429:
        retry_after = _parse_retry_after(response.headers.get("retry-after"))
        suffix = f" (retry after {retry_after}s)" if retry_after is not None else ""
        return QuelvioRateLimitError(
            f"Quelvio rate limit exceeded.{suffix}",
            retry_after_seconds=retry_after,
        )
    if 500 <= status < 600:
        return QuelvioServerError(
            f"Quelvio server error: {status} {response.reason_phrase}",
            status_code=status,
        )
    return QuelvioError(f"Unexpected response: {status} {response.reason_phrase}")


def _parse_retry_after(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_detail(response: httpx.Response) -> str | None:
    try:
        payload = response.json()
    except ValueError:
        text = response.text.strip()
        return text[:500] if text else None
    if isinstance(payload, dict):
        detail = payload.get("detail") or payload.get("message")
        if isinstance(detail, str):
            return detail
    return None


def _build_query_body(
    *,
    query: str,
    limit: int | None,
    mode: str | None,
    domain_filter: str | None,
) -> dict[str, Any]:
    req = QueryRequest(
        query=query,
        limit=_bound_limit(limit),
        mode=_normalize_mode(mode),
        domain_filter=domain_filter,
    )
    return req.model_dump(exclude_none=True)


class _BaseClient:
    """Shared helpers between :class:`QuelvioClient` and :class:`AsyncQuelvioClient`.

    The bearer token is stored in a private attribute and never appears in
    ``repr()``. Subclasses must implement the actual request methods.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        # Resolve and stash. The leading underscore communicates "do not touch";
        # we further hide the value from ``repr`` and exception messages so a
        # caller never has to think about leakage.
        self._api_key = _resolve_api_key(api_key)
        self.base_url = _resolve_base_url(base_url)
        self.timeout = timeout
        self.max_retries = max_retries

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(base_url={self.base_url!r}, "
            f"timeout={self.timeout!r}, max_retries={self.max_retries!r})"
        )

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": _user_agent(),
            "X-Quelvio-Source": "langchain-python",
        }

    def _url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return f"{self.base_url}{path}"


class QuelvioClient(_BaseClient):
    """Synchronous HTTP client for the Quelvio enterprise API."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        http_client: httpx.Client | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._client = http_client or httpx.Client(timeout=timeout)
        self._owns_client = http_client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> QuelvioClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def query(
        self,
        *,
        query: str,
        limit: int | None = 5,
        mode: str | None = "standard",
        domain_filter: str | None = None,
    ) -> QueryResponse:
        """Send a one-shot query and return the parsed response."""
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")
        body = _build_query_body(query=query, limit=limit, mode=mode, domain_filter=domain_filter)
        raw = self._request("POST", "/v1/enterprise/query", json=body)
        return QueryResponse.model_validate(raw)

    def list_domains(self, *, coverage: str | None = None) -> DomainsListResponse:
        params = {"coverage": coverage} if coverage else None
        raw = self._request("GET", "/v1/enterprise/domains", params=params)
        return DomainsListResponse.model_validate(raw)

    def get_source_detail(self, query_id: str) -> SourceDetailResponse:
        if not query_id:
            raise ValueError("query_id must be a non-empty string")
        raw = self._request("GET", f"/v1/enterprise/sources/{query_id}")
        return SourceDetailResponse.model_validate(raw)

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.request(
                    method,
                    self._url(path),
                    headers=self._headers(),
                    json=json,
                    params=params,
                )
            except httpx.TimeoutException as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(_backoff_seconds(attempt, random.random()))
                    continue
                raise QuelvioTimeoutError(
                    f"Request to {path} timed out after {self.timeout}s"
                ) from exc
            except httpx.HTTPError as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(_backoff_seconds(attempt, random.random()))
                    continue
                raise QuelvioNetworkError(
                    f"Network error contacting Quelvio: {type(exc).__name__}"
                ) from exc

            if response.is_success:
                if not response.content:
                    return None
                return response.json()

            if response.status_code in _RETRYABLE_STATUSES and attempt < self.max_retries:
                time.sleep(_backoff_seconds(attempt, random.random()))
                continue

            raise _map_error(response)

        # Loop fell through — should be unreachable, but raise the last error
        # if it ever happens (defensive: keeps mypy happy and the contract clear).
        raise QuelvioError(f"Request failed after {self.max_retries + 1} attempts") from last_exc


class AsyncQuelvioClient(_BaseClient):
    """Asynchronous HTTP client for the Quelvio enterprise API."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._client = http_client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = http_client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> AsyncQuelvioClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def query(
        self,
        *,
        query: str,
        limit: int | None = 5,
        mode: str | None = "standard",
        domain_filter: str | None = None,
    ) -> QueryResponse:
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")
        body = _build_query_body(query=query, limit=limit, mode=mode, domain_filter=domain_filter)
        raw = await self._request("POST", "/v1/enterprise/query", json=body)
        return QueryResponse.model_validate(raw)

    async def list_domains(self, *, coverage: str | None = None) -> DomainsListResponse:
        params = {"coverage": coverage} if coverage else None
        raw = await self._request("GET", "/v1/enterprise/domains", params=params)
        return DomainsListResponse.model_validate(raw)

    async def get_source_detail(self, query_id: str) -> SourceDetailResponse:
        if not query_id:
            raise ValueError("query_id must be a non-empty string")
        raw = await self._request("GET", f"/v1/enterprise/sources/{query_id}")
        return SourceDetailResponse.model_validate(raw)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        import asyncio

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = await self._client.request(
                    method,
                    self._url(path),
                    headers=self._headers(),
                    json=json,
                    params=params,
                )
            except httpx.TimeoutException as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    await asyncio.sleep(_backoff_seconds(attempt, random.random()))
                    continue
                raise QuelvioTimeoutError(
                    f"Request to {path} timed out after {self.timeout}s"
                ) from exc
            except httpx.HTTPError as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    await asyncio.sleep(_backoff_seconds(attempt, random.random()))
                    continue
                raise QuelvioNetworkError(
                    f"Network error contacting Quelvio: {type(exc).__name__}"
                ) from exc

            if response.is_success:
                if not response.content:
                    return None
                return response.json()

            if response.status_code in _RETRYABLE_STATUSES and attempt < self.max_retries:
                await asyncio.sleep(_backoff_seconds(attempt, random.random()))
                continue

            raise _map_error(response)

        raise QuelvioError(f"Request failed after {self.max_retries + 1} attempts") from last_exc
