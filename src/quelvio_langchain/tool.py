"""``QuelvioTool`` — a LangChain ``BaseTool`` for agents.

Use this when you want an LLM agent to *decide* whether to query the
company's knowledge brain. The tool returns a synthesized natural-language
answer plus a list of cited sources (titles + URLs), which the agent can
quote back to the user.
"""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from .client import AsyncQuelvioClient, QuelvioClient
from .types import QueryResponse

_DEFAULT_DESCRIPTION = (
    "Search the organization's connected knowledge brain (Google Drive, "
    "SharePoint, Confluence, Slack, Notion, and other internal sources) "
    "for an authoritative, cited answer. Use this whenever the user asks "
    "about internal company information — policies, processes, decisions, "
    "people, products, projects, or anything else that lives in the "
    "company's systems rather than on the public internet. The answer is "
    "scoped to the running user's individual access permissions, so "
    "results never include documents they cannot already see. Returns a "
    "synthesized answer plus a list of cited sources (titles + URLs)."
)


class QuelvioToolInput(BaseModel):
    """Arguments schema for :class:`QuelvioTool`.

    Modeled with Pydantic so LangChain agents get a well-typed JSON
    Schema for function-calling.
    """

    model_config = ConfigDict(extra="ignore")

    question: str = Field(
        ...,
        description=(
            "The natural-language question to ask the company's knowledge "
            "brain. Phrase it as the user would ask it — do not pre-process "
            "or keyword-extract."
        ),
    )
    mode: str | None = Field(
        default=None,
        description=(
            "Synthesis depth: 'fast' for low-latency retrieval-only, "
            "'standard' (default) for retrieval + synthesis, 'deep' for "
            "multi-pass reasoning over a wider window."
        ),
    )
    max_sources: int | None = Field(
        default=None,
        ge=1,
        le=50,
        description="Maximum number of source chunks to retrieve (1 to 50, default 5).",
    )
    domain: str | None = Field(
        default=None,
        description=(
            "Optional taxonomy domain to restrict retrieval to "
            "(e.g. 'engineering', 'legal', 'people-ops')."
        ),
    )


def _format_response(response: QueryResponse) -> str:
    """Render a :class:`QueryResponse` as a single string for the agent."""
    lines: list[str] = []
    if response.synthesis:
        lines.append(response.synthesis.strip())
        lines.append("")
    if response.results:
        lines.append("Sources:")
        for idx, chunk in enumerate(response.results, start=1):
            label = chunk.title or chunk.chunk_id
            if chunk.source_url:
                lines.append(f"  [{idx}] {label} — {chunk.source_url}")
            else:
                lines.append(f"  [{idx}] {label}")
    if not lines:
        return "No matching content was found in the company's knowledge brain."
    return "\n".join(lines).rstrip()


class QuelvioTool(BaseTool):
    """LangChain agent tool backed by the Quelvio enterprise knowledge API.

    Example:
        >>> from quelvio_langchain import QuelvioTool
        >>> tool = QuelvioTool(api_key="qlv_pat_...")
        >>> from langchain.agents import AgentExecutor, create_tool_calling_agent
        >>> # ... pass [tool] to your agent constructor ...

    Args:
        api_key: Quelvio Personal Access Token, OAuth access token, or
            Service Account key. Falls back to ``QUELVIO_API_KEY``.
        base_url: Override the API base URL.
        timeout: Per-request timeout in seconds.
        default_mode: Default synthesis mode if the agent omits ``mode``.
        default_max_sources: Default chunk limit if the agent omits ``max_sources``.
        client / async_client: Optionally inject pre-built clients.
    """

    name: str = "quelvio_query"
    description: str = _DEFAULT_DESCRIPTION
    args_schema: type[BaseModel] = QuelvioToolInput

    api_key: str | None = Field(default=None, exclude=True, repr=False)
    base_url: str | None = None
    timeout: float = 30.0
    default_mode: str = "standard"
    default_max_sources: int = 5

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
        object.__setattr__(self, "api_key", None)

    def _run(
        self,
        question: str,
        mode: str | None = None,
        max_sources: int | None = None,
        domain: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        if not question or not question.strip():
            raise ValueError("question must be a non-empty string")
        response = self._client.query(
            query=question,
            limit=max_sources if max_sources is not None else self.default_max_sources,
            mode=mode or self.default_mode,
            domain_filter=domain,
        )
        return _format_response(response)

    async def _arun(
        self,
        question: str,
        mode: str | None = None,
        max_sources: int | None = None,
        domain: str | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        if not question or not question.strip():
            raise ValueError("question must be a non-empty string")
        if self._async_client is None:
            self._async_client = AsyncQuelvioClient(
                api_key=self._client._api_key,
                base_url=self._client.base_url,
                timeout=self._client.timeout,
            )
        response = await self._async_client.query(
            query=question,
            limit=max_sources if max_sources is not None else self.default_max_sources,
            mode=mode or self.default_mode,
            domain_filter=domain,
        )
        return _format_response(response)
