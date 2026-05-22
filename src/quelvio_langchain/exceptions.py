"""Typed exceptions for quelvio-langchain.

Every exception inherits from :class:`QuelvioError`. Catch that for a
broad net; catch a subclass to handle a specific failure mode (auth,
rate limit, transient server error, etc.).
"""

from __future__ import annotations


class QuelvioError(Exception):
    """Base exception for all Quelvio client errors."""


class QuelvioAuthError(QuelvioError):
    """Authentication failed (HTTP 401 or 403).

    The bearer token is invalid, expired, revoked, or lacks the scope
    required for the requested operation.
    """


class QuelvioBadRequestError(QuelvioError):
    """The request was rejected as malformed (HTTP 400)."""


class QuelvioNotFoundError(QuelvioError):
    """The requested resource was not found (HTTP 404)."""


class QuelvioRateLimitError(QuelvioError):
    """Rate limited by the Quelvio API (HTTP 429).

    Attributes:
        retry_after_seconds: Value of the ``Retry-After`` response header
            if the server provided one; otherwise ``None``.
    """

    def __init__(self, message: str, retry_after_seconds: int | None = None) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class QuelvioServerError(QuelvioError):
    """Quelvio returned a 5xx server error after retries were exhausted.

    Attributes:
        status_code: The HTTP status code from the final response.
    """

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class QuelvioTimeoutError(QuelvioError):
    """The HTTP request timed out before a response was received."""


class QuelvioNetworkError(QuelvioError):
    """A non-timeout transport-level error (DNS, TLS, connection refused, etc.)."""
