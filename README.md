# quelvio-langchain

> Quelvio for LangChain — your company's brain as a LangChain tool and retriever.

`quelvio-langchain` is the official Python integration that plugs Quelvio's
enterprise knowledge API into [LangChain](https://python.langchain.com). It
ships two first-class building blocks — a `Retriever` for RAG chains and a
`Tool` for agents — both wired to your organization's connected sources
(Google Drive, SharePoint, Confluence, Slack, Notion, and the rest of your
content fabric) and scoped to the running user's individual permissions.

[![PyPI version](https://img.shields.io/pypi/v/quelvio-langchain.svg)](https://pypi.org/project/quelvio-langchain/)
[![Python versions](https://img.shields.io/pypi/pyversions/quelvio-langchain.svg)](https://pypi.org/project/quelvio-langchain/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

## Why Quelvio (and not vanilla RAG)?

A naive RAG pipeline embeds every chunk it can find and ranks by cosine
similarity. That's why most internal copilots confidently quote a
three-year-old draft. Quelvio is a managed company-brain that does the
work a generic vector store can't:

- **Authority scoring.** Every chunk is ranked by *who authored it*, *how
  fresh it is*, and *how many downstream documents reference it* — not just
  semantic similarity to the question.
- **Lifecycle awareness.** Drafts, deprecated docs, and superseded
  decisions are demoted automatically; chunks return a `lifecycle_state`
  the LLM can quote when hedging.
- **Per-employee permissioning.** Every query is scoped to the running
  user's identity. Results never include documents the user can't already
  read in the source system (Drive ACLs, Confluence space restrictions,
  SharePoint groups).
- **Synthesized answers with citations.** The API returns a final answer
  *plus* the chunks that informed it, so your agent can hand the user a
  link to the source of truth, not a hallucination.

## Install

```bash
pip install quelvio-langchain
```

Requires Python 3.10+ and `langchain-core>=0.1`.

## Quickstart

### As a retriever (RAG chain)

```python
from quelvio_langchain import QuelvioRetriever

retriever = QuelvioRetriever(api_key="qlv_pat_...")  # or set QUELVIO_API_KEY
docs = retriever.invoke("what's our refund policy?")

for d in docs:
    print(f"{d.metadata['title']} (authority={d.metadata.get('authority_score')})")
    print(d.page_content[:200])
    print("---")
```

Each returned `Document` carries the chunk's `source_url`, `authority_score`,
`taxonomy_domain`, `chunk_id`, and (when present) the author's name,
email, and department in `metadata`.

### As an agent tool

```python
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from quelvio_langchain import QuelvioTool

llm = ChatAnthropic(model="claude-sonnet-4-6")
tools = [QuelvioTool(api_key="qlv_pat_...")]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the quelvio_query tool "
               "whenever the user asks about internal company information."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)

print(executor.invoke({"input": "What's our parental leave policy?"})["output"])
```

The tool's name is `quelvio_query` and its schema accepts `question`
(required) plus optional `mode` (`fast` | `standard` | `deep`),
`max_sources` (1–50), and `domain` (taxonomy domain filter).

### One-shot synthesis

For the simplest case — single answer, no chain, no agent:

```python
from quelvio_langchain import synthesize_answer

result = synthesize_answer("what is our deployment process?")
print(result.answer)
for source in result.sources:
    print(f"  • {source.title} → {source.source_url}")
```

## Authentication

`quelvio-langchain` resolves a bearer token from the first non-empty
source, in order:

| Precedence | Source                          | Notes                                              |
| ---------- | ------------------------------- | -------------------------------------------------- |
| 1          | `api_key=...` constructor arg   | Highest priority; never persisted, never logged.   |
| 2          | `QUELVIO_API_KEY` env var       | Best for CI, notebooks, and one-off scripts.       |

Three token types are accepted — the wire format is identical, so the
library does not need to know which kind you provided:

- **Personal Access Token (PAT).** Long-lived bearer tied to a human
  user. Generate at <https://enterprise.quelvio.com/account> → *Personal
  API Keys* → *Create token*. Best for ad-hoc use and CI.
- **OAuth access token.** Short-lived token from the device-code flow
  (`quelvio login` in the [CLI](https://github.com/Quelvio/quelvio-cli)).
- **Service Account key.** Long-lived, machine-scoped. Generate at
  *Settings* → *Service Accounts*. Best for production agents.

The token is held privately on the client; it never appears in
`repr()`, exception messages, or any log line emitted by this library.

## Configuration

| Constructor arg / env var       | Default                       | Purpose                                            |
| ------------------------------- | ----------------------------- | -------------------------------------------------- |
| `api_key` / `QUELVIO_API_KEY`   | *(required)*                  | Bearer token (PAT, OAuth, or Service Account).     |
| `base_url` / `QUELVIO_API_BASE` | `https://api.quelvio.com`     | API base — point at `api-dev` for staging.         |
| `timeout`                       | `30.0` seconds                | Per-request HTTP timeout.                          |
| `limit` (retriever) / `max_sources` (tool) | `5`                | Max chunks returned per query (1–50).              |
| `mode`                          | `"standard"`                  | `fast` / `standard` / `deep`.                      |
| `domain_filter` (retriever) / `domain` (tool) | `None`          | Restrict to one taxonomy domain.                   |

## Examples

### 1. Simple Q&A with citations

```python
from quelvio_langchain import QuelvioRetriever

retriever = QuelvioRetriever()  # reads QUELVIO_API_KEY

docs = retriever.invoke("how do we handle on-call escalations?")
for d in docs:
    title = d.metadata["title"]
    url = d.metadata.get("source_url", "(no link)")
    authority = d.metadata.get("authority_score", "—")
    print(f"[authority {authority}] {title}\n  {url}\n  {d.page_content[:160]}\n")
```

### 2. Agent that combines Quelvio + web search + a calculator

```python
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from quelvio_langchain import QuelvioTool


@tool
def calculator(expression: str) -> str:
    """Evaluate a simple arithmetic expression. Supports + - * / ( )."""
    import ast
    import operator as op

    ops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv}

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.BinOp):
            return ops[type(node.op)](_eval(node.left), _eval(node.right))
        raise ValueError(f"Unsupported: {ast.dump(node)}")

    return str(_eval(ast.parse(expression, mode="eval").body))


tools = [QuelvioTool(), DuckDuckGoSearchRun(), calculator]

llm = ChatAnthropic(model="claude-sonnet-4-6")
prompt = ChatPromptTemplate.from_messages([
    ("system", "Use quelvio_query for anything about THIS company. Use the "
               "web search for external/public information. Use the "
               "calculator for math. Always cite Quelvio sources by URL."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
print(executor.invoke({"input": "How does our refund window compare to "
                                "the industry standard, and how many "
                                "refunds did we process last quarter?"})["output"])
```

### 3. RAG chain with Quelvio as the retriever

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from quelvio_langchain import QuelvioRetriever

retriever = QuelvioRetriever(mode="deep", limit=8)

prompt = ChatPromptTemplate.from_template(
    "Answer the user's question using ONLY the context below. After your "
    "answer, list the source URLs you used.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)


def format_docs(docs):
    blocks = []
    for d in docs:
        url = d.metadata.get("source_url", "(no url)")
        blocks.append(f"[{d.metadata['title']} — {url}]\n{d.page_content}")
    return "\n\n".join(blocks)


llm = ChatAnthropic(model="claude-sonnet-4-6")
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print(chain.invoke("Summarize our Q4 OKR review decisions."))
```

## Related packages

- **[`@quelvio/cli`](https://github.com/Quelvio/quelvio-cli)** — query the
  brain from your terminal, scriptable in CI, JSON output.
- **[`@quelvio/mcp-server`](https://github.com/Quelvio/quelvio-mcp-server)** —
  use Quelvio from any Model Context Protocol client (Claude Desktop,
  Cursor, VS Code, etc.).
- **[Quelvio docs](https://docs.quelvio.com)** — concepts, API reference,
  source connectors.

## Development

```bash
git clone https://github.com/Quelvio/quelvio-langchain-python
cd quelvio-langchain-python
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Linting and type-checking:

```bash
ruff check src tests
ruff format --check src tests
mypy src
```

## Contributing

Issues and pull requests welcome at
<https://github.com/Quelvio/quelvio-langchain-python>. Please run `ruff
check`, `ruff format`, `mypy`, and `pytest` before opening a PR.

## License

MIT — see [LICENSE](./LICENSE).
