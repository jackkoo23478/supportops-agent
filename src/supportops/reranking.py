"""Lazy cross-encoder reranking of question/policy pairs."""

from typing import Protocol

import numpy as np

from supportops.tracing import span, traced


class Reranker(Protocol):
    def score(self, question: str, passages: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    """Load once on first use and return raw relevance logits, not probabilities."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        *,
        local_files_only: bool = False,
    ) -> None:
        self.model_name = model_name
        self.local_files_only = local_files_only
        self._model = None

    @traced("rerank.score", inputs=("question", "passages"))
    def score(self, question: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        if self._model is None:
            import torch
            from sentence_transformers import CrossEncoder

            on_gpu = torch.cuda.is_available()
            with span("rerank.model.load", inputs={"model": self.model_name}):
                self._model = CrossEncoder(
                    self.model_name,
                    local_files_only=self.local_files_only,
                    device="cuda" if on_gpu else "cpu",
                    max_length=512,
                    activation_fn=torch.nn.Identity(),
                    model_kwargs={"torch_dtype": torch.float16} if on_gpu else {},
                )
        scores = self._model.predict(
            [(question, passage) for passage in passages],
            batch_size=min(8, len(passages)),
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        values = np.asarray(scores, dtype=float).reshape(-1)
        if values.shape != (len(passages),) or not np.isfinite(values).all():
            raise ValueError("Reranker must return one finite score per passage.")
        return values.tolist()
