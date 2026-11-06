"""Strict evaluation fixtures with explicit expected outcomes."""

import re
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ReferenceFact(BaseModel):
    """Reviewed reference claims with transparent, bounded regex checks."""

    model_config = ConfigDict(extra="forbid")

    id: str
    description: str
    any_patterns: list[str] = Field(min_length=1)
    reject_patterns: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_patterns(self):
        for pattern in self.any_patterns + self.reject_patterns:
            if not pattern.strip():
                raise ValueError("Reference fact patterns must not be empty.")
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"Invalid reference fact pattern: {exc}") from exc
        return self


class Expected(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: Literal["order_status", "policy_question", "refund", "complaint", "other", "none"]
    contains: list[str] = Field(default_factory=list)
    excludes: list[str] = Field(default_factory=list)
    required_sources: list[str] = Field(default_factory=list)
    facts: list[ReferenceFact] = Field(default_factory=list)
    state: dict[str, Any] = Field(default_factory=dict)
    tickets: int = Field(ge=0)
    refunds: int = Field(ge=0)
    transactions: int = Field(ge=0)
    refund_statuses: list[str] | None = None
    refund_orders: list[str] | None = None
    refund_amounts: list[str] | None = None
    ticket_orders: list[str] | None = None
    tool_calls: list[str] | None = None


class EvalStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["message", "review"] = "message"
    message: str | None = None
    decision: Literal["approve", "reject"] | None = None
    expect: Expected

    @model_validator(mode="after")
    def check_action(self):
        if self.kind == "message" and (not self.message or self.decision is not None):
            raise ValueError("Message steps require a message and no review decision.")
        if self.kind == "review" and (self.decision is None or self.message is not None):
            raise ValueError("Review steps require a decision and no customer message.")
        if self.kind == "review" and self.expect.intent != "none":
            raise ValueError("Administrator reviews do not call intent classification.")
        return self


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9_]+$")
    description: str
    customer_id: str = "C001"
    requested_on: date | None = None
    steps: list[EvalStep] = Field(min_length=1)


class EvalSuite(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal[1]
    requested_on: date
    cases: list[EvalCase] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_ids(self):
        if len({case.id for case in self.cases}) != len(self.cases):
            raise ValueError("Evaluation case IDs must be unique.")
        return self
