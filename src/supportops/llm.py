"""The only module that talks to a language model."""

from dataclasses import dataclass
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from supportops.config import Settings

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
    ) -> None:
        self.model = model
        self._http = httpx.Client(base_url=base_url, timeout=timeout, transport=transport)

    def chat(self, messages: list[dict], *, schema: dict | None = None) -> ChatResult:
        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0},
        }
        if schema is not None:
            body["format"] = schema
        try:
            response = self._http.post("/api/chat", json=body)
            response.raise_for_status()
        except httpx.TimeoutException as e:
            raise LLMError(f"Model {self.model} timed out. Is another program using the GPU?") from e
        except httpx.HTTPError as e:
            raise LLMError(f"Ollama request failed: {e}") from e

        data = response.json()
        return ChatResult(
            content=data["message"]["content"],
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            duration_ms=data.get("total_duration", 0) / 1_000_000,
        )

    def chat_json(self, messages: list[dict], output: type[T], max_retries: int = 2) -> T:
        """Ask for JSON matching `output`. Invalid replies are sent back to the model to fix."""
        messages = list(messages)
        schema = output.model_json_schema()
        last_error: ValidationError | None = None
        for _ in range(max_retries + 1):
            result = self.chat(messages, schema=schema)
            try:
                return output.model_validate_json(result.content)
            except ValidationError as e:
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
            settings.ollama_base_url, settings.llm_model, settings.llm_timeout_seconds
        )
    raise LLMError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")