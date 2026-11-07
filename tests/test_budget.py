"""Limits must stop actual work without losing saved actions."""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from unittest.mock import Mock

import httpx
import pytest
from pydantic import BaseModel

from supportops.budget import (
    BudgetExceeded, BudgetLimits, MessageBudget, budgeted_tool, current_budget,
    message_budget,
)
from supportops.config import load_settings
from supportops.llm import OllamaClient
from supportops.loop import run_message
from supportops.refund_dialogue import RefundService
from supportops.refund_store import RefundContextStore, RefundStore
from supportops.schemas import ConversationState, IntentResult, PolicyAnswer
from supportops.store import default_data_dir
from supportops.tools.orders import get_order
from supportops.tools.tickets import create_ticket
from supportops.tracing import JsonlTracer
from supportops.verification import PolicyVerification
from supportops.tickets import TicketStore


class Answer(BaseModel):
    answer: str


def client(handler, **kwargs):
    return OllamaClient("http://test", "test", transport=httpx.MockTransport(handler), **kwargs)


def response(content="{}", prompt=12, completion=5):
    return httpx.Response(200, json={"message": {"content": content},
                                    "prompt_eval_count": prompt, "eval_count": completion})


def test_invalid_json_retries_share_the_http_limit():
    requests = []
    llm = client(lambda request: (requests.append(request), response("invalid"))[1])
    budget = MessageBudget(BudgetLimits(max_llm_calls=2))
    with message_budget(budget), pytest.raises(BudgetExceeded):
        llm.chat_json([], Answer)
    assert len(requests) == 2
    assert budget.llm_calls == 2
    assert budget.prompt_tokens + budget.completion_tokens == 34
    assert current_budget() is None


def test_failed_http_attempt_also_consumes_allowance():
    requests = []
    llm = client(lambda request: (requests.append(request), httpx.Response(500))[1])
    from supportops.llm import LLMError
    with message_budget(MessageBudget(BudgetLimits(max_llm_calls=1))):
        with pytest.raises(LLMError):
            llm.chat([])
        with pytest.raises(BudgetExceeded):
            llm.chat([])
    assert len(requests) == 1


@pytest.mark.parametrize("schema,expected", [
    (IntentResult, 256), (PolicyAnswer, 768), (PolicyVerification, 1536),
])
def test_output_cap_sent_for_each_structured_role(schema, expected):
    requests = []
    def handler(request):
        requests.append(json.loads(request.content))
        return response("{}")
    from supportops.llm import LLMError
    with pytest.raises(LLMError):
        client(handler).chat_json([], schema, max_retries=0)
    assert requests[0]["options"]["num_predict"] == expected


def test_explicit_message_limits_override_client_defaults():
    requests = []
    llm = client(lambda request: (requests.append(json.loads(request.content)),
                                 response('{"answer":"ok"}'))[1])
    with message_budget(MessageBudget(BudgetLimits(answer_max_output_tokens=90))):
        llm.chat_json([], Answer)
    llm.chat([])
    assert [r["options"]["num_predict"] for r in requests] == [90, 768]


def test_total_token_threshold_warns_once_and_does_not_block(capsys):
    llm = client(lambda request: response())
    budget = MessageBudget(BudgetLimits(max_total_tokens=10))
    with message_budget(budget):
        llm.chat([])
        llm.chat([])
    assert budget.prompt_tokens == 24 and budget.completion_tokens == 10
    assert budget.total_tokens_warning
    assert capsys.readouterr().err.count("advisory threshold") == 1


def test_tool_limit_blocks_before_reading_store():
    store = Mock()
    store.get_order.return_value = None
    with message_budget(MessageBudget(BudgetLimits(max_tool_calls=1))):
        get_order(store, "O1001")
        with pytest.raises(BudgetExceeded):
            get_order(store, "O1002")
    store.get_order.assert_called_once_with("O1001")


def test_ticket_limit_prevents_database_write(store, tmp_path):
    tickets = TicketStore(tmp_path / "tickets.db")
    with message_budget(MessageBudget(BudgetLimits(max_tool_calls=1))):
        get_order(store, "O1001")
        with pytest.raises(BudgetExceeded):
            create_ticket(store, tickets, request_id="request1", customer_id="C001",
                          order_id="O1001", description="Broken package")
    with sqlite3.connect(tickets.path) as db:
        assert db.execute("SELECT COUNT(*) FROM tickets").fetchone()[0] == 0


def test_message_reset_and_trace_capture(store, tmp_path):
    requests = []
    llm = client(lambda request: (requests.append(request), response("invalid"))[1],
                 budget_limits=BudgetLimits(max_llm_calls=1))
    tracer = JsonlTracer(tmp_path / "trace.jsonl")
    for _ in range(2):
        assert "processing limit" in run_message("Hi", llm=llm, store=store, tracer=tracer)
    assert len(requests) == 2
    events = [json.loads(line) for line in tracer.path.read_text().splitlines()]
    summaries = [e for e in events if e["name"] == "budget.message" and e["phase"] == "end"]
    assert len(summaries) == 2
    assert all(e["output"]["llm_calls"] == 1 for e in summaries)
    assert all(e["output"]["blocked_reason"] == "llm_calls" for e in summaries)
    assert current_budget() is None


def test_verifier_budget_exhaustion_reaches_message_boundary(store):
    from supportops.policy_answers import PolicyAnswerService
    llm = Mock(spec=OllamaClient)
    llm.chat_json.side_effect = [IntentResult(intent="policy_question"),
                                PolicyAnswer(answer="Returns require packaging.",
                                             source_ids=["returns.eligibility"], insufficient_evidence=False)]
    searcher = Mock()
    searcher.search.return_value = [{"source_id": "returns.eligibility",
                                   "document": "returns.md", "content": "Packaging required."}]
    verifier = Mock()
    verifier.verify.side_effect = BudgetExceeded("test limit")
    reply = run_message("Packaging?", llm=llm, store=store,
                        policy_answers=PolicyAnswerService(searcher, verifier=verifier))
    assert "processing limit" in reply


def test_saved_refund_survives_blocked_execution_and_retries_once(store, tmp_path):
    refunds = RefundStore(tmp_path / "refund.db")
    contexts = RefundContextStore(default_data_dir() / "refund_contexts.json")
    service = RefundService(refunds, contexts, requested_on=date(2026, 11, 6))
    state = ConversationState()
    llm = Mock(spec=OllamaClient)
    llm.chat_json.return_value = IntentResult(intent="refund")
    options = dict(llm=llm, store=store, state=state, refunds=service,
                   budget_limits=BudgetLimits(max_tool_calls=1))
    run_message("Refund O1001 USD 30", **options)
    reply = run_message("confirm refund", **options)
    assert "processing limit" in reply and state.last_refund_id in reply
    assert "refund request was saved" in reply
    with sqlite3.connect(refunds.path) as db:
        assert db.execute("SELECT COUNT(*) FROM refund_requests").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM refund_transactions").fetchone()[0] == 0
    assert "Status: executed" in run_message("retry refund", **options)
    run_message("retry refund", **options)
    with sqlite3.connect(refunds.path) as db:
        assert db.execute("SELECT COUNT(*) FROM refund_transactions").fetchone()[0] == 1


def test_completed_ticket_is_reported_if_later_work_is_blocked(store):
    @budgeted_tool
    def create_ticket():
        return {"ok": True, "data": {"ticket_id": "T-saved"}}
    class Service:
        def handle(self, *args, **kwargs):
            create_ticket()
            get_order(store, "O1001")
    llm = Mock(spec=OllamaClient)
    llm.chat_json.return_value = IntentResult(intent="complaint")
    reply = run_message("Complaint", llm=llm, store=store, complaints=Service(),
                        budget_limits=BudgetLimits(max_tool_calls=1))
    assert "Your ticket was saved" in reply and "T-saved" in reply


def test_parallel_budgets_do_not_share_counters():
    def work(_):
        budget = MessageBudget()
        with message_budget(budget):
            budget.consume("llm")
            return current_budget().llm_calls
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(work, range(4))) == [1, 1, 1, 1]
    assert current_budget() is None


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_limits_are_rejected(value):
    with pytest.raises(ValueError, match="positive integers"):
        BudgetLimits(max_llm_calls=value)


def test_settings_load_new_limits_and_ignore_legacy_steps(monkeypatch):
    monkeypatch.setenv("MAX_LLM_CALLS", "4")
    monkeypatch.setenv("MAX_TOOL_CALLS", "2")
    monkeypatch.setenv("MAX_STEPS", "legacy-unused")
    monkeypatch.setenv("VERIFY_MAX_OUTPUT_TOKENS", "900")
    limits = load_settings().budget_limits
    assert limits.max_llm_calls == 4 and limits.max_tool_calls == 2
    assert limits.verify_max_output_tokens == 900
