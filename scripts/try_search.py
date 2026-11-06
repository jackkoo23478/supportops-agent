"""Inspect hybrid policy search without calling the chat model."""

import argparse
import json

from supportops.policies import load_policy_chunks
from supportops.retrieval import HybridPolicySearcher
from supportops.tools.policies import search_policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", help="Omit to start interactive search.")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--candidate-k", type=int, default=5)
    parser.add_argument("--min-semantic-score", type=float)
    args = parser.parse_args()
    if args.top_k < 1 or args.candidate_k < args.top_k:
        parser.error("Require top-k >= 1 and candidate-k >= top-k.")
    if args.min_semantic_score is not None and not -1 <= args.min_semantic_score <= 1:
        parser.error("min-semantic-score must be between -1 and 1.")

    chunks = load_policy_chunks()
    print(f"Loaded {len(chunks)} policy chunks. Building the search index...")
    searcher = HybridPolicySearcher(
        chunks, model_name=args.model, local_files_only=args.local_files_only,
        rerank=not args.no_rerank, reranker_model=args.reranker_model,
    )

    def show(query: str) -> None:
        result = search_policy(
            searcher, query, top_k=args.top_k, candidate_k=args.candidate_k,
            min_semantic_score=args.min_semantic_score,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.query is not None:
        show(args.query)
        return

    print("Search is ready. Type 'exit' or 'quit' to leave.")
    try:
        while True:
            query = input("\nQuery: ").strip()
            if query.lower() in {"exit", "quit"}:
                break
            if query:
                show(query)
    except (KeyboardInterrupt, EOFError):
        pass
    print("Goodbye.")


if __name__ == "__main__":
    main()
