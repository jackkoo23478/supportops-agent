"""Refund dialogue uses real SQLite and mocked classification, never model approvals."""

import sqlite3
from datetime import date
from decimal import Decimal
from unittest.mock import Mock

import pytest

from supportops import cli, refund_dialogue
from supportops.llm import OllamaClient
from supportops.loop import run_message
from supportops.refund_dialogue import RefundService, extract_amount
from supportops.refund_store import RefundContextStore, RefundStore
from supportops.schemas import ConversationState, IntentResult
from supportops.store import default_data_dir
from supportops.tools.common import error


@pytest.fixture
def setup_chat(tmp_path):
    refunds = RefundStore(tmp_path / "chat.db")
    contexts = RefundContextStore(default_data_dir() / "refund_contexts.json")
    service = RefundService(refunds, contexts, requested_on=date(2026, 11, 6))
    state = ConversationState()
    llm = Mock(spec=OllamaClient)
    llm.chat_json.return_value = IntentResult(intent="refund")
    return refunds, contexts, service, state, llm


def send(message, store, setup_chat, **kwargs):
    _, _, service, state, llm = setup_chat
    return run_message(message, store=store, llm=llm, state=state, refunds=service, **kwargs)


def counts(refunds):
    with sqlite3.connect(refunds.path) as db:
        return tuple(db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                     for table in ("refund_requests", "refund_transactions"))


@pytest.mark.parametrize("text", ["USD 30", "$30", "30.00 USD", "30", "Refund USD 30."])
def test_explicit_amount_formats(text):
    assert extract_amount(text) == Decimal("30")


@pytest.mark.parametrize("text", [
    "USD 0", "USD -5", "USD 1.001", "USD 20 or USD 30", "EUR30", "HK$30", "USD 1,000",
])
def test_invalid_or_ambiguous_amounts_are_not_silently_accepted(text):
    with pytest.raises(ValueError):
        extract_amount(text)


def test_collects_details_confirms_then_executes_once(store, setup_chat):
    refunds, _, _, state, llm = setup_chat
    assert "order ID" in send("I want a refund.", store, setup_chat)
    request_id = state.pending_refund.request_id
    assert "How much" in send("O1001", store, setup_chat)
    assert "confirm refund" in send("30", store, setup_chat)
    assert counts(refunds) == (0, 0)
    reply = send("confirm refund", store, setup_chat)
    assert "Status: executed" in reply
    assert "simulated refund" in reply
    assert state.pending_refund is None
    assert refunds.get(state.last_refund_id).request_id == request_id
    assert counts(refunds) == (1, 1)
    assert send("retry refund", store, setup_chat) == reply
    assert counts(refunds) == (1, 1)
    llm.chat_json.assert_called_once()


def test_complete_request_still_requires_confirmation(store, setup_chat):
    refunds, _, _, _, _ = setup_chat
    assert "confirm refund" in send("Refund O1001 USD 30.", store, setup_chat)
    assert counts(refunds) == (0, 0)


def test_human_approval_resume_and_restart_by_id(store, setup_chat):
    refunds, _, service, state, llm = setup_chat
    send("Refund O1001 USD 59.90", store, setup_chat)
    assert "pending_approval" in send("confirm refund", store, setup_chat)
    refund_id = state.last_refund_id
    assert counts(refunds) == (1, 0)
    assert "administrator" in send("approve refund", store, setup_chat)
    assert "pending_approval" in send("execute refund", store, setup_chat)
    refunds.review(refund_id, action="approve", reviewer="admin", reason="Checked.")
    assert "Status: approved" in send("refund status", store, setup_chat)
    assert counts(refunds) == (1, 0)
    restarted = ConversationState()
    reply = run_message(
        f"execute refund {refund_id}", store=store, llm=llm, state=restarted, refunds=service,
    )
    assert "Status: executed" in reply
    assert counts(refunds) == (1, 1)


def test_wrong_customer_cannot_read_or_execute_saved_refund(store, setup_chat):
    refunds, _, service, state, llm = setup_chat
    send("Refund O1001 USD 59.90", store, setup_chat)
    send("confirm refund", store, setup_chat)
    for command in ("refund status", "execute refund"):
        reply = run_message(
            f"{command} {state.last_refund_id}", llm=llm, store=store,
            state=ConversationState(customer_id="C002"), refunds=service,
        )
        assert "not available" in reply
        assert "59.90" not in reply
    assert counts(refunds) == (1, 0)


def test_policy_question_preserves_pending_draft(store, setup_chat):
    _, _, _, state, llm = setup_chat
    send("Refund O1001 USD 30", store, setup_chat)
    original = state.pending_refund.model_dump()
    llm.chat_json.return_value = IntentResult(intent="policy_question")
    policy = Mock()
    policy.answer.return_value = "Policy answer."
    assert send("What is your refund policy?", store, setup_chat, policy_answers=policy) == "Policy answer."
    assert state.pending_refund.model_dump() == original
    assert "executed" in send("confirm refund", store, setup_chat)


def test_model_invented_order_and_amount_are_not_used(store, setup_chat):
    refunds, _, _, state, llm = setup_chat
    llm.chat_json.return_value = IntentResult(intent="refund", order_id="O1001")
    assert "order ID in your message" in send("I want a refund", store, setup_chat)
    assert state.pending_refund.order_id is None
    assert state.pending_refund.amount is None
    assert counts(refunds) == (0, 0)


def test_previous_order_requires_clear_model_reference(store, setup_chat):
    _, _, _, state, llm = setup_chat
    state.last_order_id = "O1001"
    assert "order ID" in send("I want a refund", store, setup_chat)
    send("cancel refund", store, setup_chat)
    llm.chat_json.return_value = IntentResult(intent="refund", use_previous_order=True)
    assert "Refund draft: O1001" in send("Refund that order USD 30", store, setup_chat)


def test_wrong_order_clears_amount_before_correction(store, setup_chat):
    refunds, _, _, state, _ = setup_chat
    assert "correct order ID" in send("Refund O1003 USD 30", store, setup_chat)
    assert state.pending_refund.amount is None
    assert "How much" in send("O1001", store, setup_chat)
    assert counts(refunds) == (0, 0)


def test_cancellation_and_multiple_orders_never_submit(store, setup_chat):
    refunds, _, _, state, _ = setup_chat
    assert "single order ID" in send("Refund O1001 and O1002 USD 30", store, setup_chat)
    assert "cancelled" in send("cancel refund", store, setup_chat)
    assert state.pending_refund is None
    assert "no pending" in send("confirm refund", store, setup_chat)
    assert counts(refunds) == (0, 0)


def test_uncertain_submission_locks_details_and_reuses_id(store, setup_chat, monkeypatch):
    refunds, _, _, state, _ = setup_chat
    actual = refund_dialogue.request_refund
    calls = []

    def uncertain(*args, **kwargs):
        calls.append(kwargs["request_id"])
        saved = actual(*args, **kwargs)
        if len(calls) == 1:
            return error("refund_storage_failed", "Uncertain save.")
        return saved

    monkeypatch.setattr(refund_dialogue, "request_refund", uncertain)
    send("Refund O1001 USD 30", store, setup_chat)
    assert "retry refund" in send("confirm refund", store, setup_chat)
    assert counts(refunds) == (1, 0)
    assert "unresolved" in send("USD 20", store, setup_chat)
    assert "unresolved" in send("cancel refund", store, setup_chat)
    assert state.pending_refund.amount == Decimal("30")
    assert "Status: executed" in send("retry refund", store, setup_chat)
    assert calls[0] == calls[1]
    assert counts(refunds) == (1, 1)


def test_failed_execution_preserves_saved_id_for_retry(store, setup_chat, monkeypatch):
    refunds, _, _, state, _ = setup_chat
    actual = refund_dialogue.execute_refund
    monkeypatch.setattr(refund_dialogue, "execute_refund", lambda *a, **k: error(
        "refund_storage_failed", "Uncertain execution.",
    ))
    send("Refund O1001 USD 30", store, setup_chat)
    assert "execute refund R-" in send("confirm refund", store, setup_chat)
    assert state.last_refund_id is not None
    assert state.pending_refund is None
    monkeypatch.setattr(refund_dialogue, "execute_refund", actual)
    assert "executed" in send("retry", store, setup_chat)
    assert counts(refunds) == (1, 1)


def test_cannot_mix_pending_complaint_and_refund(store, setup_chat):
    _, _, _, _, llm = setup_chat
    send("Refund O1001", store, setup_chat)
    llm.chat_json.return_value = IntentResult(intent="complaint")
    assert "cancel the pending refund" in send("My item is damaged", store, setup_chat)


def test_uncertain_execution_does_not_falsely_claim_nothing_happened(
    store, setup_chat, monkeypatch,
):
    refunds, _, _, _, _ = setup_chat
    actual = refund_dialogue.execute_refund

    def committed_then_failed(*args, **kwargs):
        saved = actual(*args, **kwargs)
        assert saved["ok"]
        return error("refund_storage_failed", "Response lost after commit.")

    monkeypatch.setattr(refund_dialogue, "execute_refund", committed_then_failed)
    send("Refund O1001 USD 30", store, setup_chat)
    reply = send("confirm refund", store, setup_chat)
    assert "could not be confirmed" in reply
    assert "No refund has been executed" not in reply
    assert counts(refunds) == (1, 1)
    monkeypatch.setattr(refund_dialogue, "execute_refund", actual)
    assert "Status: executed" in send("retry refund", store, setup_chat)
    assert counts(refunds) == (1, 1)


def test_cli_passes_refund_database_and_preserves_draft(monkeypatch, capsys, store, setup_chat):
    refunds, _, service, _, llm = setup_chat
    monkeypatch.setattr(cli, "load_settings", lambda: object())
    monkeypatch.setattr(cli, "Store", lambda: store)
    monkeypatch.setattr(cli, "make_llm", lambda settings: llm)
    factory = Mock(return_value=service)
    monkeypatch.setattr(cli, "RefundService", factory)
    inputs = iter(["Refund O1001", "USD 30", "confirm refund", "retry refund", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
    assert cli.main(["--refund-db", str(refunds.path)]) == 0
    output = capsys.readouterr().out
    assert "simulated refund" in output
    assert output.count("Status: executed") == 2
    factory.assert_called_once_with(db_path=refunds.path)
    assert counts(refunds) == (1, 1)
