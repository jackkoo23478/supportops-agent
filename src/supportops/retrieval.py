"""Hybrid policy retrieval using BM25, normalized embeddings, and RRF."""

import re
from typing import Any, Protocol

import numpy as np
from rank_bm25 import BM25Okapi

from supportops.reranking import CrossEncoderReranker, Reranker
from supportops.schemas import PolicyChunk
from supportops.tracing import span, traced


class Embedder(Protocol):
    def encode_document(self, texts: list[str], **kwargs: Any) -> Any: ...

    def encode_query(self, text: str, **kwargs: Any) -> Any: ...


def tokenize(text: str) -> list[str]:
    """Tokenize English policies, retaining term frequency and decimal amounts."""
    return re.findall(r"[a-z]+|\d+(?:\.\d+)?", text.lower())


def reciprocal_rank_fusion(
    rankings: list[list[str]], *, rank_constant: int = 60
) -> dict[str, float]:
    """Sum reciprocal ranks; each source contributes once per ranking."""
    if rank_constant < 1:
        raise ValueError("rank_constant must be positive.")
    scores: dict[str, float] = {}
    for ranking in rankings:
        unique_ids = dict.fromkeys(ranking)
        for rank, source_id in enumerate(unique_ids, start=1):
            scores[source_id] = scores.get(source_id, 0.0) + 1.0 / (rank_constant + rank)
    return scores


class HybridPolicySearcher:
    """Build one in-memory index and reuse it across questions.

    RRF scores are ranking scores, not confidence probabilities. Without an
    explicitly calibrated semantic threshold, unrelated questions can return hits.
    Rebuild the searcher after changing policies or the embedding model.
    """

    def __init__(
        self,
        chunks: list[PolicyChunk],
        *,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedder: Embedder | None = None,
        local_files_only: bool = False,
        rerank: bool = True,
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
        reranker: Reranker | None = None,
    ) -> None:
        if not chunks:
            raise ValueError("At least one policy chunk is required.")
        if len({chunk.source_id for chunk in chunks}) != len(chunks):
            raise ValueError("Policy source IDs must be unique.")

        self.chunks = [chunk.model_copy(deep=True) for chunk in chunks]
        texts = [f"{chunk.source_id}\n{chunk.content}" for chunk in self.chunks]
        tokens = [tokenize(text) for text in texts]
        if not any(tokens):
            raise ValueError("Policies must contain searchable English words or numbers.")
        self.bm25 = BM25Okapi(tokens)

        if embedder is None:
            from sentence_transformers import SentenceTransformer

            with span("embedding.model.load", inputs={"model": model_name}):
                embedder = SentenceTransformer(model_name, local_files_only=local_files_only)
        self.embedder = embedder
        with span("embedding.documents", inputs={"chunk_count": len(texts)}) as embedding:
            vectors = np.asarray(
                embedder.encode_document(
                    texts, normalize_embeddings=True, convert_to_numpy=True
                ),
                dtype=float,
            )
            embedding.output = {"shape": list(vectors.shape)}
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks) or vectors.shape[1] == 0:
            raise ValueError("Document embeddings must have shape (chunk_count, dimensions).")
        if not np.isfinite(vectors).all():
            raise ValueError("Document embeddings must be finite.")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if (norms == 0).any():
            raise ValueError("Document embeddings must not be zero vectors.")
        self.document_vectors = vectors / norms
        self.reranker = (
            reranker if reranker is not None else CrossEncoderReranker(
                reranker_model, local_files_only=local_files_only
            )
        ) if rerank else None

    @traced("retrieval.hybrid", inputs=("query", "top_k", "candidate_k", "min_semantic_score"))
    def search(
        self,
        query: str,
        *,
        top_k: int = 3,
        candidate_k: int = 5,
        min_semantic_score: float | None = None,
    ) -> list[dict]:
        """Fuse candidates, rerank the best candidate_k, then return top_k.

        Optional min_semantic_score gates the whole question using its best cosine
        score before fusion. Choose it using evaluation data for the actual model.
        """
        if top_k < 1 or candidate_k < top_k:
            raise ValueError("Require top_k >= 1 and candidate_k >= top_k.")
        if min_semantic_score is not None and not -1 <= min_semantic_score <= 1:
            raise ValueError("min_semantic_score must be between -1 and 1.")
        query = query.strip()
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        with span("retrieval.bm25", inputs={"query_tokens": query_tokens}) as keyword:
            bm25_scores = np.asarray(self.bm25.get_scores(query_tokens), dtype=float)
            keyword.output = dict(zip(
                [chunk.source_id for chunk in self.chunks], bm25_scores.tolist(),
            ))
        with span("embedding.query", inputs={"query": query}) as embedding:
            query_vector = np.asarray(
                self.embedder.encode_query(
                    query, normalize_embeddings=True, convert_to_numpy=True
                ),
                dtype=float,
            )
            embedding.output = {"shape": list(query_vector.shape)}
        if query_vector.shape != (self.document_vectors.shape[1],):
            raise ValueError("Query embedding dimensions must match document embeddings.")
        query_norm = np.linalg.norm(query_vector)
        if not np.isfinite(query_vector).all() or not np.isfinite(query_norm) or query_norm == 0:
            raise ValueError("Query embedding must be a finite, nonzero vector.")
        with span("retrieval.semantic", inputs={"query": query}) as semantic:
            semantic_scores = np.clip(self.document_vectors @ (query_vector / query_norm), -1, 1)
            semantic.output = dict(zip(
                [chunk.source_id for chunk in self.chunks], semantic_scores.tolist(),
            ))
        if min_semantic_score is not None and semantic_scores.max() < min_semantic_score:
            return []

        def ranked_ids(scores: np.ndarray) -> list[str]:
            indices = sorted(
                (index for index, score in enumerate(scores) if score > 0),
                key=lambda index: (-float(scores[index]), self.chunks[index].source_id),
            )[:candidate_k]
            return [self.chunks[index].source_id for index in indices]

        rankings = [ranked_ids(bm25_scores), ranked_ids(semantic_scores)]
        with span("retrieval.rrf", inputs={"rankings": rankings}) as fusion:
            fused = reciprocal_rank_fusion(rankings)
            fusion.output = fused
        indices_by_id = {chunk.source_id: index for index, chunk in enumerate(self.chunks)}
        candidate_ids = sorted(
            fused, key=lambda source_id: (-fused[source_id], source_id)
        )[:candidate_k]
        rerank_scores: dict[str, float] = {}
        if self.reranker is not None and candidate_ids:
            passages = [
                self.chunks[indices_by_id[source_id]].content for source_id in candidate_ids
            ]
            with span("retrieval.rerank", inputs={"source_ids": candidate_ids}) as reranking:
                scores = np.asarray(self.reranker.score(query, passages), dtype=float)
                reranking.output = scores.tolist()
            if scores.shape != (len(candidate_ids),) or not np.isfinite(scores).all():
                raise ValueError("Reranker must return one finite score per passage.")
            rerank_scores = dict(zip(candidate_ids, scores.tolist()))
            # Stable ties preserve the existing RRF ranking.
            candidate_ids.sort(key=lambda source_id: -rerank_scores[source_id])
        best_ids = candidate_ids[:top_k]
        return [
            {
                **self.chunks[indices_by_id[source_id]].model_dump(),
                "bm25_score": float(bm25_scores[indices_by_id[source_id]]),
                "semantic_score": float(semantic_scores[indices_by_id[source_id]]),
                "rrf_score": fused[source_id],
                "rerank_score": rerank_scores.get(source_id),
            }
            for source_id in best_ids
        ]
