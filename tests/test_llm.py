import json

import httpx
import pytest
from pydantic import BaseModel

from supportops.config import Settings
from supportops.llm import LLMError, OllamaClient, make_llm


class Answer(BaseModel):
    intent: str


def ollama_reply(content: str) -> httpx.Response:
    """A response shaped like Ollama's /api/chat reply."""
    return httpx.Response(
        200,
        json={
            "message": {"role": "assistant", "content": content},
            "prompt_eval_count": 12,
            "eval_count": 5,
            "total_duration": 2_000_000,  # nanoseconds
        },
    )


def make_client(handler) -> OllamaClient:
    return OllamaClient(
        "http://ollama.test", "test-model", transport=httpx.MockTransport(handler)
    )


def test_chat_parses_reply_and_usage():
    client = make_client(lambda request: ollama_reply("hello"))

    result = client.chat([{"role": "user", "content": "hi"}])

    assert result.content == "hello"
    assert result.prompt_tokens == 12
    assert result.completion_tokens == 5
    assert result.duration_ms == 2.0


def test_chat_sends_model_temperature_and_schema():
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return ollama_reply("{}")

    schema = {"type": "object"}
    make_client(handler).chat([{"role": "user", "content": "hi"}], schema=schema)

    assert seen["path"] == "/api/chat"
    assert seen["body"]["model"] == "test-model"
    assert seen["body"]["stream"] is False
    assert seen["body"]["options"]["temperature"] == 0
    assert seen["body"]["format"] == schema


def test_chat_json_returns_model_object():
    client = make_client(lambda request: ollama_reply('{"intent": "refund"}'))

    result = client.chat_json([{"role": "user", "content": "money back"}], Answer)

    assert result == Answer(intent="refund")


def test_chat_json_retries_and_shows_the_error_to_the_model():
    replies = iter(["not json at all", '{"intent": "refund"}'])
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return ollama_reply(next(replies))

    result = make_client(handler).chat_json([{"role": "user", "content": "x"}], Answer)

    assert result.intent == "refund"
    assert len(requests) == 2
    second_messages = requests[1]["messages"]
    assert second_messages[-2] == {"role": "assistant", "content": "not json at all"}
    assert "invalid" in second_messages[-1]["content"]


def test_chat_json_gives_up_after_max_retries():
    calls = []

    def handler(request):
        calls.append(1)
        return ollama_reply("still not json")

    with pytest.raises(LLMError, match="did not match Answer"):
        make_client(handler).chat_json([{"role": "user", "content": "x"}], Answer, max_retries=2)

    assert len(calls) == 3


def test_timeout_becomes_llm_error():
    def handler(request):
        raise httpx.ReadTimeout("too slow", request=request)

    with pytest.raises(LLMError, match="timed out"):
        make_client(handler).chat([{"role": "user", "content": "hi"}])


def test_server_error_becomes_llm_error():
    client = make_client(lambda request: httpx.Response(500, text="boom"))

    with pytest.raises(LLMError, match="request failed"):
        client.chat([{"role": "user", "content": "hi"}])


def test_make_llm_rejects_unknown_provider():
    settings = Settings(
        llm_provider="unknown",
        llm_model="m",
        ollama_base_url="http://ollama.test",
        llm_timeout_seconds=1,
        anthropic_api_key=None,
        max_steps=8,
        refund_auto_approve_limit=50,
    )

    with pytest.raises(LLMError, match="Unsupported"):
        make_llm(settings)
