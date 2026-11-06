"""Deterministic retrieval tests; no model download or GPU is required."""

import json
from unittest.mock import Mock

import numpy as np
import pytest

from supportops.retrieval import HybridPolicySearcher, reciprocal_rank_fusion, tokenize
from supportops.schemas import PolicyChunk
from supportops.tools.policies import search_policy


@pytest.fixture
def chunks():
    return [
        PolicyChunk(
            source_id="refunds.processing", document="refunds.md",
            content="Refunds go to the original payment method after inspection.",
        ),
        PolicyChunk(
            source_id="warranty.coverage", document="warranty.md",
            content="Manufacturing defects have warranty coverage for electronics.",
        ),
        PolicyChunk(
            source_id="shipping.delivery", document="shipping.md",
            content="Standard delivery takes five business days after shipment.",
        ),
    ]


@pytest.fixture
def embedder():
    model = Mock()
    model.encode_document.return_value = np.eye(3) * 2
    model.encode_query.return_value = np.array([3.0, 0.0, 0.0])
    return model


@pytest.fixture
def searcher(chunks, embedder):
    return HybridPolicySearcher(chunks, embedder=embedder, rerank=False)


def test_tokenizer_keeps_term_frequency_and_amounts():
    assert tokenize("Refund refund USD 50.00!") == ["refund", "refund", "usd", "50.00"]


def test_rrf_uses_ranks_and_rewards_shared_hits():
    scores = reciprocal_rank_fusion([["a", "b"], ["b", "c"]])
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["b"] > scores["a"]
    assert scores["b"] > scores["c"]


def test_rrf_does_not_count_duplicate_ids_twice():
    assert reciprocal_rank_fusion([["a", "a", "b"]]) == {
        "a": 1 / 61, "b": 1 / 62,
    }


def test_semantic_only_hit_can_be_returned(searcher):
    hits = searcher.search("money back", top_k=1)
    assert hits[0]["source_id"] == "refunds.processing"
    assert hits[0]["bm25_score"] == 0
    assert hits[0]["semantic_score"] == pytest.approx(1)
    assert hits[0]["rrf_score"] == pytest.approx(1 / 61)


def test_bm25_can_supply_a_hit_missing_from_positive_semantic_candidates(searcher):
    hits = searcher.search("manufacturing defects", top_k=3)
    by_id = {hit["source_id"]: hit for hit in hits}
    assert by_id["warranty.coverage"]["bm25_score"] > 0
    assert by_id["warranty.coverage"]["semantic_score"] == 0
    assert "shipping.delivery" not in by_id


def test_matching_both_retrievers_adds_both_rank_contributions(searcher):
    hit = searcher.search("original payment method", top_k=1)[0]
    assert hit["source_id"] == "refunds.processing"
    assert hit["rrf_score"] == pytest.approx(2 / 61)


def test_embeddings_are_built_once_and_queries_are_encoded_per_search(searcher, embedder):
    searcher.search("refund")
    searcher.search("delivery")
    embedder.encode_document.assert_called_once()
    assert embedder.encode_query.call_count == 2


def test_blank_query_does_not_call_embedder(searcher, embedder):
    assert searcher.search("  !!! ") == []
    embedder.encode_query.assert_not_called()


def test_optional_threshold_can_reject_a_question(searcher, embedder):
    embedder.encode_query.return_value = np.ones(3)
    assert searcher.search("unrelated topic", min_semantic_score=0.9) == []
    assert searcher.search("unrelated topic")


def test_top_k_and_candidate_limits(searcher, embedder):
    embedder.encode_query.return_value = np.array([3.0, 2.0, 1.0])
    hits = searcher.search("unmatched", top_k=2, candidate_k=2)
    assert [hit["source_id"] for hit in hits] == [
        "refunds.processing", "warranty.coverage",
    ]


def test_tool_returns_serializable_sources(searcher):
    result = search_policy(searcher, "original payment", top_k=1)
    assert result["ok"] is True
    hit = result["data"]["matches"][0]
    assert hit["document"] == "refunds.md"
    assert "original payment" in hit["content"]
    json.dumps(result)


def test_tool_rejects_bad_query_and_options(searcher):
    assert search_policy(searcher, " ")["error"] == "empty_query"
    assert search_policy(searcher, "refund", top_k=0)["error"] == "invalid_search_options"
    assert search_policy(searcher, "refund", min_semantic_score=2)["ok"] is False


@pytest.mark.parametrize("options", [
    {"top_k": 0}, {"top_k": 4, "candidate_k": 2}, {"min_semantic_score": 2},
])
def test_search_rejects_bad_options(searcher, options):
    with pytest.raises(ValueError):
        searcher.search("refund", **options)


def test_empty_and_duplicate_corpora_are_rejected(chunks, embedder):
    with pytest.raises(ValueError, match="At least one"):
        HybridPolicySearcher([], embedder=embedder)
    with pytest.raises(ValueError, match="unique"):
        HybridPolicySearcher([chunks[0], chunks[0]], embedder=embedder)


@pytest.mark.parametrize("vectors", [
    np.zeros((3, 3)), np.ones((2, 3)), np.full((3, 3), np.nan),
])
def test_bad_document_vectors_are_rejected(chunks, embedder, vectors):
    embedder.encode_document.return_value = vectors
    with pytest.raises(ValueError):
        HybridPolicySearcher(chunks, embedder=embedder)


@pytest.mark.parametrize("vector", [np.ones(2), np.zeros(3), np.full(3, np.nan)])
def test_bad_query_vectors_are_rejected(searcher, embedder, vector):
    embedder.encode_query.return_value = vector
    with pytest.raises(ValueError):
        searcher.search("refund")
