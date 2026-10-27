"""Settings read from environment variables and the .env file."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    llm_model: str
    ollama_base_url: str
    llm_timeout_seconds: float
    anthropic_api_key: str | None
    max_steps: int
    refund_auto_approve_limit: float


def load_settings() -> Settings:
    load_dotenv()
    return Settings(
        llm_provider=os.environ.get("LLM_PROVIDER", "ollama"),
        llm_model=os.environ.get("LLM_MODEL", "qwen2.5:7b"),
        ollama_base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        llm_timeout_seconds=float(os.environ.get("LLM_TIMEOUT_SECONDS", "120")),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY") or None,
        max_steps=int(os.environ.get("MAX_STEPS", "8")),
        refund_auto_approve_limit=float(os.environ.get("REFUND_AUTO_APPROVE_LIMIT", "50")),
    )