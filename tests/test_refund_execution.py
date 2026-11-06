"""Simulated refunds must be authorized, atomic, and exactly once locally."""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from decimal import Decimal

import pytest

from scripts.execute_refund import main
from supportops.refund_store import RefundContextStore, RefundStore
from supportops.store import default_data_dir
from supportops.tools.refunds import execute_refund, request_refund


@pytest.fixture
def setup_execution(tmp_path, store):
    refunds = RefundStore(tmp_path / "execution.db")
    contexts = RefundContextStore(default_data_dir() / "refund_contexts.json")

    def submit(request_id="one", amount="30.00", **changes):
        args = dict(
            request_id=request_id, customer_id="C001", order_id="O1001", amount=amount,
            requested_on=date(2026, 11, 6),
        )
        args.update(changes)
        return request_refund(store, refunds, contexts, **args)["data"]

    def execute(refund_id, customer_id="C001"):
        return execute_refund(
            store, refunds, contexts, refund_id=refund_id, customer_id=customer_id,
        )

    return refunds, contexts, submit, execute


def transaction_count(refunds):
    with sqlite3.connect(refunds.path) as db:
        return db.execute("SELECT COUNT(*) FROM refund_transactions").fetchone()[0]


def test_auto_approved_execution_is_persistent_and_explicitly_simulated(setup_execution):
    refunds, _, submit, execute = setup_execution
    refund_id = submit()["refund_id"]
    result = execute(refund_id)
    assert result["ok"]
    data = result["data"]
    assert data["decision"]["status"] == "executed"
    assert data["authorization_decision"]["status"] == "auto_approved"
    transaction = data["execution"]
    assert transaction["simulated"] is True
    assert transaction["transaction_id"].startswith("SIM-")
    assert transaction["amount"] == "30.00"
    assert transaction["payment_method"] == "original_payment_method"
    saved = RefundStore(refunds.path).get(refund_id)
    assert saved.model_dump(mode="json") == data
    assert transaction_count(refunds) == 1


def test_human_approval_is_required_above_limit(setup_execution):
    refunds, _, submit, execute = setup_execution
    refund_id = submit(amount="59.90")["refund_id"]
    assert execute(refund_id)["error"] == "refund_execution_blocked"
    assert transaction_count(refunds) == 0
    refunds.review(refund_id, action="approve", reviewer="admin", reason="Verified.")
    result = execute(refund_id)
    assert result["data"]["decision"]["status"] == "executed"
    assert result["data"]["review"]["action"] == "approve"
    assert result["data"]["authorization_decision"]["status"] == "approved"


@pytest.mark.parametrize("changes", [
    {"order_id": "O1002", "amount": "10"},
    {"customer_id": "C002", "order_id": "O1003", "amount": "10"},
    {"customer_id": "C003", "order_id": "O1005", "amount": "10"},
])
def test_nonapproved_states_never_execute(setup_execution, changes):
    refunds, _, submit, execute = setup_execution
    record = submit(**changes)
    result = execute(record["refund_id"], record["customer_id"])
    assert result["error"] == "refund_execution_blocked"
    assert transaction_count(refunds) == 0


def test_human_rejection_cannot_execute(setup_execution):
    refunds, _, submit, execute = setup_execution
    refund_id = submit(amount="59.90")["refund_id"]
    refunds.review(refund_id, action="reject", reviewer="admin", reason="Not approved.")
    assert execute(refund_id)["error"] == "refund_execution_blocked"
    assert transaction_count(refunds) == 0


def test_wrong_customer_and_missing_refund_do_not_write(setup_execution):
    refunds, _, submit, execute = setup_execution
    refund_id = submit()["refund_id"]
    assert execute(refund_id, "C002")["error"] == "refund_customer_mismatch"
    assert execute("R-MISSING")["error"] == "refund_not_found"
    assert execute(" ")["error"] == "invalid_refund_execution"
    assert transaction_count(refunds) == 0


@pytest.mark.parametrize("change", ["balance", "facts"])
def test_current_verified_context_is_rechecked_before_execution(setup_execution, change):
    refunds, contexts, submit, execute = setup_execution
    refund_id = submit()["refund_id"]
    context = contexts._entries["O1001"]
    if change == "balance":
        context.refundable_balance = Decimal("20")
    else:
        context.facts.inspection_passed = False
    assert execute(refund_id)["error"] == "refund_execution_blocked"
    assert refunds.get(refund_id).decision.status == "auto_approved"
    assert transaction_count(refunds) == 0


def test_retry_returns_same_transaction_after_context_changes(setup_execution):
    refunds, contexts, submit, execute = setup_execution
    refund_id = submit()["refund_id"]
    first = execute(refund_id)
    contexts._entries["O1001"].facts.inspection_passed = False
    assert execute(refund_id) == first
    assert submit() == first["data"]
    assert transaction_count(refunds) == 1


def test_executed_amount_stays_consumed_and_other_reservations_remain(setup_execution):
    refunds, _, submit, execute = setup_execution
    first = submit()
    second = submit(request_id="two", amount="29.90")
    assert execute(first["refund_id"])["ok"]
    assert submit(request_id="three", amount="0.01")["decision"]["status"] == "rejected"
    assert execute(second["refund_id"])["ok"]
    assert transaction_count(refunds) == 2


def test_concurrent_execution_produces_one_transaction(setup_execution):
    refunds, _, submit, execute = setup_execution
    refund_id = submit()["refund_id"]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: execute(refund_id), range(4)))
    assert all(result == results[0] for result in results)
    assert results[0]["ok"]
    assert transaction_count(refunds) == 1


def test_failed_state_update_rolls_back_ledger_insert(setup_execution):
    refunds, _, submit, execute = setup_execution
    refund_id = submit()["refund_id"]
    with sqlite3.connect(refunds.path) as db:
        db.execute("""
            CREATE TRIGGER fail_execution BEFORE UPDATE ON refund_requests
            BEGIN SELECT RAISE(ABORT, 'simulated failure'); END
        """)
    assert execute(refund_id)["error"] == "refund_storage_failed"
    assert transaction_count(refunds) == 0
    assert refunds.get(refund_id).decision.status == "auto_approved"
    with sqlite3.connect(refunds.path) as db:
        db.execute("DROP TRIGGER fail_execution")
    assert execute(refund_id)["ok"]
    assert transaction_count(refunds) == 1


@pytest.mark.parametrize("status,amount", [("approved", "30.00"), ("auto_approved", "59.90")])
def test_status_alone_cannot_bypass_approval_evidence(setup_execution, status, amount):
    refunds, _, submit, execute = setup_execution
    record = submit(amount=amount)
    record["decision"]["status"] = status
    with sqlite3.connect(refunds.path) as db:
        db.execute("UPDATE refund_requests SET record_json = ?", (json.dumps(record),))
    assert execute(record["refund_id"])["error"] == "refund_execution_blocked"
    assert transaction_count(refunds) == 0


def test_cli_executes_and_retries_same_simulated_transaction(setup_execution, capsys):
    refunds, _, submit, _ = setup_execution
    args = ["--db", str(refunds.path), "--refund-id", submit()["refund_id"],
            "--customer-id", "C001"]
    assert main(args) == 0
    first = json.loads(capsys.readouterr().out)
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out) == first
    assert transaction_count(refunds) == 1
