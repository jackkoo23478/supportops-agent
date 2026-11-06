"""End-to-end complaint collection with a real temporary SQLite ticket store."""

import sqlite3
from contextlib import closing
from unittest.mock import Mock

import pytest

from supportops import cli, complaints as complaint_module
from supportops.complaints import ComplaintService
from supportops.llm import OllamaClient
from supportops.loop import run_message
from supportops.schemas import ConversationState, IntentResult
from supportops.tickets import TicketStore
from supportops.tools.common import error


@pytest.fixture
def setup_flow(tmp_path):
    tickets = TicketStore(tmp_path / "tickets.db")
    service = ComplaintService(tickets)
    state = ConversationState(customer_id="C001")
    llm = Mock(spec=OllamaClient)
    llm.chat_json.return_value = IntentResult(intent="complaint")
    return tickets, service, state, llm


def rows(tickets):
    with closing(sqlite3.connect(tickets.path)) as db:
        db.row_factory = sqlite3.Row
        return [dict(row) for row in db.execute("SELECT * FROM tickets")]


def send(message, store, setup_flow):
    _, service, state, llm = setup_flow
    return run_message(message, store=store, llm=llm, state=state, complaints=service)


def test_complaint_then_bare_order_preserves_description_and_creates_ticket(store, setup_flow):
    tickets, _, state, llm = setup_flow
    assert "provide the order ID" in send("My item arrived damaged.", store, setup_flow)
    request_id = state.pending_complaint.request_id
    assert rows(tickets) == []
    reply = send("o1001", store, setup_flow)
    assert "Your complaint has been recorded." in reply
    assert state.last_ticket_id in reply
    assert state.pending_complaint is None
    saved = rows(tickets)[0]
    assert saved["description"] == "My item arrived damaged."
    assert saved["request_id"] == request_id
    assert saved["customer_id"] == "C001"
    assert saved["order_id"] == "O1001"
    llm.chat_json.assert_called_once()


def test_explicit_order_can_create_ticket_in_one_turn(store, setup_flow):
    tickets, _, _, _ = setup_flow
    reply = send("The item in O1001 arrived damaged.", store, setup_flow)
    assert "Ticket ID:" in reply
    assert len(rows(tickets)) == 1


@pytest.mark.parametrize("order_id", ["O9999", "O1003"])
def test_missing_or_other_customer_order_keeps_complaint_pending(store, setup_flow, order_id):
    tickets, _, state, _ = setup_flow
    send("My item arrived damaged.", store, setup_flow)
    request_id = state.pending_complaint.request_id
    reply = send(order_id, store, setup_flow)
    assert "could not find that order for this customer" in reply
    assert rows(tickets) == []
    assert state.customer_id == "C001"
    assert state.pending_complaint.request_id == request_id
    assert state.pending_complaint.order_id is None
    assert "Ticket ID:" in send("O1001", store, setup_flow)


def test_clear_previous_order_reference_is_supported(store, setup_flow):
    tickets, _, state, llm = setup_flow
    state.last_order_id = "O1001"
    llm.chat_json.return_value = IntentResult(intent="complaint", use_previous_order=True)
    assert "Ticket ID:" in send("That order arrived damaged.", store, setup_flow)
    assert rows(tickets)[0]["order_id"] == "O1001"


def test_new_complaint_does_not_implicitly_use_last_order(store, setup_flow):
    tickets, _, state, _ = setup_flow
    state.last_order_id = "O1001"
    assert "provide the order ID" in send("My item arrived damaged.", store, setup_flow)
    assert rows(tickets) == []


def test_model_invented_id_is_not_used_for_a_write(store, setup_flow):
    tickets, _, _, llm = setup_flow
    llm.chat_json.return_value = IntentResult(intent="complaint", order_id="O1001")
    assert "provide the order ID" in send("My item arrived damaged.", store, setup_flow)
    assert rows(tickets) == []


def test_multiple_explicit_ids_require_clarification(store, setup_flow):
    tickets, _, state, _ = setup_flow
    reply = send("O1001 and O1002 arrived damaged.", store, setup_flow)
    assert "single order ID" in reply
    assert rows(tickets) == []
    assert state.pending_complaint is not None


def test_policy_question_does_not_discard_pending_complaint(store, setup_flow):
    _, service, state, llm = setup_flow
    send("My item arrived damaged.", store, setup_flow)
    pending = state.pending_complaint.model_dump()
    llm.chat_json.return_value = IntentResult(intent="policy_question")
    policy_answers = Mock()
    policy_answers.answer.return_value = "A policy answer."
    reply = run_message(
        "What is the return policy?", llm=llm, store=store, state=state,
        complaints=service, policy_answers=policy_answers,
    )
    assert reply == "A policy answer."
    assert state.pending_complaint.model_dump() == pending
    assert "Ticket ID:" in send("O1001", store, setup_flow)


def test_retry_after_uncertain_save_uses_same_id_and_does_not_duplicate(
    store, setup_flow, monkeypatch,
):
    tickets, _, state, _ = setup_flow
    actual_create = complaint_module.create_ticket
    seen_ids = []

    def save_then_report_failure(*args, **kwargs):
        seen_ids.append(kwargs["request_id"])
        saved = actual_create(*args, **kwargs)
        assert saved["ok"]
        if len(seen_ids) == 1:
            return error("ticket_storage_failed", "Uncertain result")
        return saved

    monkeypatch.setattr(complaint_module, "create_ticket", save_then_report_failure)
    reply = send("Order O1001 arrived damaged.", store, setup_flow)
    assert "could not confirm" in reply
    assert "Ticket ID:" not in reply
    assert state.pending_complaint is not None
    assert len(rows(tickets)) == 1
    assert "Ticket ID:" in send("retry", store, setup_flow)
    assert seen_ids[0] == seen_ids[1]
    assert len(rows(tickets)) == 1
    send("retry", store, setup_flow)
    assert len(seen_ids) == 2


def test_initial_database_failure_preserves_pending_action(store, tmp_path):
    blocked_parent = tmp_path / "not-a-directory"
    blocked_parent.write_text("file", encoding="utf-8")
    service = ComplaintService(db_path=blocked_parent / "tickets.db")
    state = ConversationState()
    llm = Mock(spec=OllamaClient)
    llm.chat_json.return_value = IntentResult(intent="complaint")
    reply = run_message(
        "Order O1001 arrived damaged.", llm=llm, store=store, state=state, complaints=service
    )
    assert "could not confirm" in reply
    assert state.pending_complaint.order_id == "O1001"


def test_cancellation_discards_pending_action_without_calling_model(store, setup_flow):
    tickets, _, state, llm = setup_flow
    send("My item arrived damaged.", store, setup_flow)
    reply = send("cancel complaint", store, setup_flow)
    assert "cancelled" in reply
    assert state.pending_complaint is None
    assert rows(tickets) == []
    llm.chat_json.assert_called_once()


def test_cli_collects_and_saves_complaint(monkeypatch, capsys, tmp_path, store):
    path = tmp_path / "cli.db"
    llm = Mock(spec=OllamaClient)
    llm.chat_json.return_value = IntentResult(intent="complaint")
    monkeypatch.setattr(cli, "load_settings", lambda: object())
    monkeypatch.setattr(cli, "Store", lambda: store)
    monkeypatch.setattr(cli, "make_llm", lambda settings: llm)
    inputs = iter(["My item arrived damaged.", "O1001", "retry", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
    assert cli.main(["--ticket-db", str(path)]) == 0
    output = capsys.readouterr().out
    assert "Demo customer: C001" in output
    assert "provide the order ID" in output
    assert "Your complaint has been recorded." in output
    assert len(rows(TicketStore(path))) == 1


def test_single_message_cli_uses_selected_demo_identity(monkeypatch, capsys, tmp_path, store):
    path = tmp_path / "single.db"
    llm = Mock(spec=OllamaClient)
    llm.chat_json.return_value = IntentResult(intent="complaint")
    monkeypatch.setattr(cli, "load_settings", lambda: object())
    monkeypatch.setattr(cli, "Store", lambda: store)
    monkeypatch.setattr(cli, "make_llm", lambda settings: llm)
    assert cli.main([
        "Order O1003 arrived damaged.", "--customer-id", "C002", "--ticket-db", str(path),
    ]) == 0
    assert "Ticket ID:" in capsys.readouterr().out
    assert rows(TicketStore(path))[0]["customer_id"] == "C002"
