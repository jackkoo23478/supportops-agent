"""The only module that talks to a language model."""

from dataclasses import dataclass
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from supportops.budget import (
    BudgetLimits, current_budget, output_limit, requested_output_limit,
)
from supportops.config import Settings
from supportops.tracing import span, traced

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """The model could not be reached or kept returning invalid output."""


@dataclass
class ChatResult:
    content: str
    prompt_tokens: int
    completion_tokens: int
    duration_ms: float


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 120,
        transport: httpx.BaseTransport | None = None,
        budget_limits: BudgetLimits | None = None,
    ) -> None:
        self.budget_limits = budget_limits or BudgetLimits()
        self.model = model
        self._http = httpx.Client(base_url=base_url, timeout=timeout, transport=transport)

    @traced("llm.chat", inputs=("messages", "schema"))
    def chat(self, messages: list[dict], *, schema: dict | None = None) -> ChatResult:
        budget = current_budget()
        limits = budget.limits if budget is not None else self.budget_limits
        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0, "num_predict": requested_output_limit(
                limits.answer_max_output_tokens)},
        }
        if schema is not None:
            body["format"] = schema
        if budget is not None:
            budget.consume("llm")
        try:
            response = self._http.post("/api/chat", json=body)
            response.raise_for_status()
        except httpx.TimeoutException as e:
            raise LLMError(f"Model {self.model} timed out. Is another program using the GPU?") from e
        except httpx.HTTPError as e:
            raise LLMError(f"Ollama request failed: {e}") from e

        data = response.json()
        if budget is not None:
            budget.record_tokens(data.get("prompt_eval_count", 0), data.get("eval_count", 0))
        return ChatResult(
            content=data["message"]["content"],
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            duration_ms=data.get("total_duration", 0) / 1_000_000,
        )

    @traced("llm.structured", inputs=("messages", "max_retries"))
    def chat_json(self, messages: list[dict], output: type[T], max_retries: int = 2) -> T:
        """Ask for JSON matching `output`. Invalid replies are sent back to the model to fix."""
        messages = list(messages)
        schema = output.model_json_schema()
        last_error: ValidationError | None = None
        for attempt in range(max_retries + 1):
            with output_limit(
                output.__name__,
                current_budget().limits if current_budget() is not None else self.budget_limits,
            ):
                result = self.chat(messages, schema=schema)
            try:
                return output.model_validate_json(result.content)
            except ValidationError as e:
                with span("llm.validation_retry", inputs={
                    "attempt": attempt + 1, "error": str(e),
                }) as step:
                    step.status = "error"
                    step.error = {"type": "ValidationError", "message": str(e)}
                last_error = e
                messages += [
                    {"role": "assistant", "content": result.content},
                    {
                        "role": "user",
                        "content": f"That reply was invalid: {e}. Reply again with JSON that matches the schema.",
                    },
                ]
        raise LLMError(
            f"Model output did not match {output.__name__} after {max_retries + 1} attempts: {last_error}"
        )


def make_llm(settings: Settings) -> OllamaClient:
    if settings.llm_provider == "ollama":
        return OllamaClient(
            settings.ollama_base_url, settings.llm_model, settings.llm_timeout_seconds,
            budget_limits=settings.budget_limits
        )
    raise LLMError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")
