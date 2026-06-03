# Strict-Mode Sentinel Handoff (FE-13)

> **Docs PR.** The full reference implementation lives in
> [`Quelvio/quelvio-mcp-server`](https://github.com/Quelvio/quelvio-mcp-server/blob/main/STRICT_MODE_HANDOFF.md).
> This file copies the contract so an engineer touching
> `quelvio-langchain-python` can mirror the pattern locally without
> context-switching across repos.

## What backend PR #643 ships

Backend PR #643 emits two response headers globally on the search /
retrieval endpoints:

| Header | Value | Meaning |
| --- | --- | --- |
| `X-Quelvio-API-Version` | `2.0` | API contract version. Informational. |
| `X-Quelvio-Sentinel-Set` | `closed-v1` | Tenant is on the strict (closed) permission model. Some results may be filtered. |

When the sentinel header is present, SDK consumers may see fewer search
results than expected — the strict model only returns chunks for which
the calling employee has explicit access.

## Contract

When this retriever observes `X-Quelvio-Sentinel-Set` on any response:

1. Log a warning **once per process** (idempotent — repeated observations
   stay silent). Warning text:
   ```
   Quelvio v2 strict permission mode is active for your tenant.
   Some search results may be filtered to enforce explicit permissions.
   Learn more: https://docs.quelvio.com/permission-model
   ```
2. Surface via `logging.getLogger("quelvio_langchain").warning(...)` —
   LangChain Python uses stdlib logging. Never raise.
3. Prefix with the structured event token
   `quelvio_sentinel_set_detected sentinel=<value>`.

## Where to wire it in `quelvio-langchain-python`

This SDK uses `httpx`. The cleanest hook is the `event_hooks` parameter
on `httpx.Client`:

```python
# src/quelvio_langchain/sentinel.py
import logging
import threading

import httpx

_log = logging.getLogger(__name__)
_seen: set[str] = set()
_lock = threading.Lock()

SENTINEL_HEADER = "X-Quelvio-Sentinel-Set"
DOCS_URL = "https://docs.quelvio.com/permission-model"


def note_sentinel(response: httpx.Response) -> None:
    value = response.headers.get(SENTINEL_HEADER)
    if not value:
        return
    with _lock:
        if value in _seen:
            return
        _seen.add(value)
    _log.warning(
        "quelvio_sentinel_set_detected sentinel=%s docs=%s", value, DOCS_URL
    )
    _log.warning("Quelvio v2 strict permission mode is active for your tenant.")
    _log.warning(
        "Some search results may be filtered to enforce explicit permissions."
    )
    _log.warning("Learn more: %s", DOCS_URL)
```

Wire into the client at `src/quelvio_langchain/client.py:237`:

```python
self._client = http_client or httpx.Client(
    timeout=timeout,
    event_hooks={"response": [note_sentinel]},
)
```

For the async variant repeat the same hook on the `httpx.AsyncClient`.

Add a `pytest` case under `tests/` using `respx` or `httpx.MockTransport`
to assert the warning fires once on `closed-v1` and stays silent on
subsequent calls. Use `caplog` to capture log records.

## Implementation crib

See [`Quelvio/quelvio-mcp-server` `src/sentinel.ts`](https://github.com/Quelvio/quelvio-mcp-server/blob/main/src/sentinel.ts)
for the dedupe-set pattern. The Python translation above mirrors the
contract exactly.

## Owner

FE-13 / antonis@rolle.io. Backend counterpart: PR #643 on
`Quelvio/quelvio-platform`.
