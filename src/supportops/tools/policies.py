"""Policy search tool backed by a reusable hybrid index."""

from supportops.retrieval import HybridPolicySearcher, tokenize
from supportops.tools.common import error, ok
from supportops.tracing import traced


@traced("tool.search_policy", inputs=("query", "top_k", "candidate_k", "min_semantic_score"))
def search_policy(
    searcher: HybridPolicySearcher,
    query: str,
    *,
    top_k: int = 3,
    candidate_k: int = 5,
    min_semantic_score: float | None = None,
) -> dict:
    """Return policy passages and their sources, without generating an answer."""
    if top_k < 1 or candidate_k < top_k:
        return error("invalid_search_options", "Require top_k >= 1 and candidate_k >= top_k.")
    if min_semantic_score is not None and not -1 <= min_semantic_score <= 1:
        return error("invalid_search_options", "min_semantic_score must be between -1 and 1.")
    if not tokenize(query):
        return error("empty_query", "Please provide policy search keywords.")
    matches = searcher.search(
        query,
        top_k=top_k,
        candidate_k=candidate_k,
        min_semantic_score=min_semantic_score,
    )
    return ok({"query": query.strip(), "matches": matches})
