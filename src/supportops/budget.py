"""Per-message execution limits, independent of optional tracing."""

import sys
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from functools import wraps

from supportops.tracing import span


class BudgetExceeded(Exception):
    """An operation was blocked before execution."""


@dataclass(frozen=True)
class BudgetLimits:
    max_llm_calls: int = 10
    max_tool_calls: int = 5
    classify_max_output_tokens: int = 256
    answer_max_output_tokens: int = 768
    verify_max_output_tokens: int = 1536
    max_total_tokens: int = 20000

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in asdict(self).values()):
            raise ValueError("Budget limits must be positive integers.")


@dataclass
class MessageBudget:
    limits: BudgetLimits = field(default_factory=BudgetLimits)
    llm_calls: int = 0
    tool_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens_warning: bool = False
    blocked_reason: str | None = None
    completed_writes: list[dict] = field(default_factory=list)

    def consume(self, kind: str):
        attr = "llm_calls" if kind == "llm" else "tool_calls"
        if getattr(self, attr) >= getattr(self.limits, "max_" + attr):
            self.blocked_reason = attr
            raise BudgetExceeded(f"The per-message {attr} limit was reached.")
        setattr(self, attr, getattr(self, attr) + 1)

    def record_tokens(self, prompt: int, completion: int):
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        if (self.prompt_tokens + self.completion_tokens > self.limits.max_total_tokens
                and not self.total_tokens_warning):
            self.total_tokens_warning = True
            print("Message token usage exceeded the configured advisory threshold.", file=sys.stderr)

    def snapshot(self):
        return {key: value for key, value in asdict(self).items() if key != "completed_writes"}


_current: ContextVar[MessageBudget | None] = ContextVar("message_budget", default=None)
_output_limit: ContextVar[int | None] = ContextVar("output_limit", default=None)


def current_budget():
    return _current.get()


@contextmanager
def message_budget(budget: MessageBudget):
    token = _current.set(budget)
    try:
        with span("budget.message", inputs=asdict(budget.limits)) as active:
            try:
                yield budget
            finally:
                active.output = budget.snapshot()
                if budget.blocked_reason:
                    active.status = "error"
                    active.error = {"code": "budget_exceeded", "limit": budget.blocked_reason}
    finally:
        _current.reset(token)


@contextmanager
def output_limit(schema_name: str, limits: BudgetLimits):
    amount = (limits.classify_max_output_tokens if schema_name == "IntentResult" else
              limits.verify_max_output_tokens if schema_name == "PolicyVerification" else
              limits.answer_max_output_tokens)
    token = _output_limit.set(amount)
    try:
        yield
    finally:
        _output_limit.reset(token)


def requested_output_limit(default: int):
    return _output_limit.get() or default


def budgeted_tool(function):
    """Count public tool invocations before any read or mutation occurs."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        budget = current_budget()
        if budget is not None:
            budget.consume("tool")
        result = function(*args, **kwargs)
        if (budget is not None and function.__name__ in
                {"create_ticket", "request_refund", "execute_refund"} and result.get("ok")):
            budget.completed_writes.append({"tool": function.__name__, "data": result["data"]})
        return result
    return wrapped


def limit_reply(budget: MessageBudget):
    lines = ["I could not complete this request within the processing limit."]
    for action in budget.completed_writes:
        data = action["data"]
        if action["tool"] == "create_ticket":
            lines.append(f"Your ticket was saved. Ticket ID: {data['ticket_id']}.")
        elif action["tool"] == "request_refund":
            lines.append(f"Your refund request was saved. Refund ID: {data['refund_id']}. "
                         "Use 'refund status' with this ID to check its status.")
        else:
            lines.append(f"Simulated refund recorded. Refund ID: {data['refund_id']}. "
                         "No real money was transferred.")
    if not budget.completed_writes:
        lines.append("Please try a more specific question or retry the same request.")
    return "\n".join(lines)
