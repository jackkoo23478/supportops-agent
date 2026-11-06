"""Reranking tests with fake models; no downloads are needed."""

from unittest.mock import Mock

import numpy as np
import pytest

from supportops.reranking import CrossEncoderReranker
from supportops.retrieval import HybridPolicySearcher
from supportops.schemas import PolicyChunk
from supportops.tools.policies import search_policy


@pytest.fixture
def setup_search():
    chunks = [
        PolicyChunk(source_id=source_id, document="policies.md", content=content)
        for source_id, content in [
            ("a", "First passage about returns."),
            ("b", "Second passage about warranty."),
            ("c", "Third passage about shipping."),
        ]
    ]
    embedder = Mock()
    embedder.encode_document.return_value = np.eye(3)
    embedder.encode_query.return_value = np.array([3.0, 2.0, 1.0])
    reranker = Mock()
    reranker.score.side_effect = lambda question, passages: [
        {chunk.content: value for chunk, value in zip(chunks, [-4.0, 2.0, 8.0])}[passage]
        for passage in passages
    ]
    searcher = HybridPolicySearcher(chunks, embedder=embedder, reranker=reranker)
    return searcher, reranker


def test_default_top_three_are_selected_after_reranking(setup_search):
    searcher, reranker = setup_search
    hits = searcher.search("unmatched")
    assert [hit["source_id"] for hit in hits] == ["c", "b", "a"]
    assert hits[0]["source_id"] == "c"
    assert hits[0]["rerank_score"] == 8.0
    assert len(reranker.score.call_args.args[1]) == 3


def test_rrf_candidate_pool_is_truncated_before_reranking(setup_search):
    searcher, reranker = setup_search
    hits = searcher.search("unmatched", top_k=1, candidate_k=2)
    assert hits[0]["source_id"] == "b"
    assert reranker.score.call_args.args == (
        "unmatched", ["First passage about returns.", "Second passage about warranty."]
    )


def test_reranking_preserves_source_and_retrieval_scores(setup_search):
    searcher, _ = setup_search
    result = search_policy(searcher, "unmatched")
    hit = result["data"]["matches"][0]
    assert hit["document"] == "policies.md"
    assert hit["content"] == "Third passage about shipping."
    assert hit["bm25_score"] == 0
    assert hit["semantic_score"] > 0
    assert hit["rrf_score"] == pytest.approx(1 / 63)


def test_equal_rerank_scores_preserve_rrf_order(setup_search):
    searcher, reranker = setup_search
    reranker.score.side_effect = lambda question, passages: [0.0] * len(passages)
    assert [hit["source_id"] for hit in searcher.search("unmatched", top_k=3)] == [
        "a", "b", "c",
    ]


def test_rejected_or_empty_queries_do_not_call_reranker(setup_search):
    searcher, reranker = setup_search
    assert searcher.search(" ") == []
    assert searcher.search("unmatched", min_semantic_score=0.99) == []
    reranker.score.assert_not_called()


@pytest.mark.parametrize("scores", [[1.0], [1.0, np.nan, 2.0]])
def test_search_rejects_malformed_reranker_results(setup_search, scores):
    searcher, reranker = setup_search
    reranker.score.side_effect = None
    reranker.score.return_value = scores
    with pytest.raises(ValueError, match="one finite score"):
        searcher.search("unmatched")


def test_disabled_reranking_does_not_call_the_model(setup_search):
    searcher, reranker = setup_search
    searcher.reranker = None
    hits = searcher.search("unmatched")
    assert hits[0]["source_id"] == "a"
    assert hits[0]["rerank_score"] is None
    reranker.score.assert_not_called()


def test_cross_encoder_wrapper_reuses_model_and_accepts_negative_logits():
    reranker = CrossEncoderReranker(local_files_only=True)
    model = Mock()
    model.predict.return_value = np.array([-3.0, 2.0])
    reranker._model = model
    assert reranker.score("question", ["first", "second"]) == [-3.0, 2.0]
    assert reranker.score("question", ["first", "second"]) == [-3.0, 2.0]
    assert model.predict.call_count == 2
    assert model.predict.call_args.args[0] == [
        ("question", "first"), ("question", "second"),
    ]


def test_empty_passages_do_not_load_a_model():
    reranker = CrossEncoderReranker()
    assert reranker.score("question", []) == []
    assert reranker._model is None
