"""Settings read from environment variables and the .env file."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from supportops.budget import BudgetLimits


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    llm_model: str
    ollama_base_url: str
    llm_timeout_seconds: float
    anthropic_api_key: str | None
    max_steps: int
    refund_auto_approve_limit: float
    budget_limits: BudgetLimits = BudgetLimits()


def load_settings() -> Settings:
    load_dotenv()
    return Settings(
        llm_provider=os.environ.get("LLM_PROVIDER", "ollama"),
        llm_model=os.environ.get("LLM_MODEL", "qwen2.5:7b"),
        ollama_base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        llm_timeout_seconds=float(os.environ.get("LLM_TIMEOUT_SECONDS", "120")),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY") or None,
        max_steps=8,  # Deprecated compatibility field; use separate call limits.
        budget_limits=BudgetLimits(**{
            key: int(os.environ.get(key.upper(), str(value)))
            for key, value in vars(BudgetLimits()).items()
        }),
        refund_auto_approve_limit=float(os.environ.get("REFUND_AUTO_APPROVE_LIMIT", "50")),
    )