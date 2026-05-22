"""Shared pytest fixtures."""

from __future__ import annotations

from typing import Any

import pytest

TEST_API_KEY = "qlv_pat_test_ABCDEFGHIJKLMNOP_NEVER_LOG_THIS"
TEST_BASE_URL = "https://api-test.quelvio.example"


@pytest.fixture(autouse=True)
def _clear_quelvio_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make sure no real env vars leak into unit tests."""
    monkeypatch.delenv("QUELVIO_API_KEY", raising=False)
    monkeypatch.delenv("QUELVIO_API_BASE", raising=False)


@pytest.fixture
def api_key() -> str:
    return TEST_API_KEY


@pytest.fixture
def base_url() -> str:
    return TEST_BASE_URL


@pytest.fixture
def query_response_payload() -> dict[str, Any]:
    """Realistic response payload from POST /v1/enterprise/query."""
    return {
        "query": "what's our refund policy?",
        "query_id": "q_01HW9X3J7K8N0V4P2QXYZA",
        "result_count": 2,
        "coverage": "complete",
        "risk_flag": {"single_source": False, "low_authority": False},
        "retrieval_mode": "standard",
        "synthesis": "Refunds are processed within 14 days of request, per policy v3.",
        "synthesis_model": "claude-sonnet-4-6",
        "latency_ms": 432,
        "tokens_consumed": 12500,
        "results": [
            {
                "chunk_id": "chunk_001",
                "content_piece_id": "cp_001",
                "title": "Refund Policy v3",
                "excerpt": "All paid customers may request a refund within 30 days.",
                "score": 0.92,
                "rank": 1,
                "authority_score": 0.87,
                "taxonomy_domain": "finance",
                "source_url": "https://drive.example/refund-policy-v3",
                "author_name": "Alex CFO",
                "author_email": "alex@example.com",
                "department": "Finance",
            },
            {
                "chunk_id": "chunk_002",
                "content_piece_id": "cp_002",
                "title": "Customer Success Playbook",
                "excerpt": "Refund requests should be acknowledged within 24h.",
                "score": 0.81,
                "rank": 2,
                "authority_score": 0.74,
                "taxonomy_domain": "customer-success",
                "source_url": "https://confluence.example/cs-playbook",
            },
        ],
    }
