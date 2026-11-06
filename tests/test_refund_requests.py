"""Persistence, ownership, idempotency, and balance reservation boundaries."""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from decimal import Decimal

import pytest

from supportops.refund_store import RefundContextStore, RefundStore
from supportops.store import default_data_dir
from supportops.tools.refunds import request_refund


@pytest.fixture
def contexts():
    return RefundContextStore(default_data_dir() / "refund_contexts.json")


@pytest.fixture
def refunds(tmp_path):
    return RefundStore(tmp_path / "refunds.db")


def submit(store, refunds, contexts, **changes):
    args = dict(
        request_id="test-001", customer_id="C001", order_id="O1001",
        amount="30.00", requested_on=date(2026, 11, 6),
    )
    args.update(changes)
    return request_refund(store, refunds, contexts, **args)


def count(refunds):
    with sqlite3.connect(refunds.path) as db:
        return db.execute("SELECT COUNT(*) FROM refund_requests").fetchone()[0]


def test_result_is_persistent_and_not_a_payment(store, refunds, contexts):
    result = submit(store, refunds, contexts)
    assert result["ok"]
    data = result["data"]
    assert data["decision"]["status"] == "auto_approved"
    assert data["facts"]["delivered_date"] == "2026-10-18"
    assert data["amount"] == "30.00"
    saved = RefundStore(refunds.path).get(data["refund_id"])
    assert saved.model_dump(mode="json") == data
    assert count(refunds) == 1
    assert refunds.get("missing") is None


def test_above_limit_waits_for_approval(store, refunds, contexts):
    result = submit(store, refunds, contexts, amount="59.90")
    assert result["data"]["decision"]["status"] == "pending_approval"


def test_retry_returns_original_snapshot_even_on_later_date(store, refunds, contexts):
    first = submit(store, refunds, contexts)
    contexts._entries["O1001"].facts.unused = False
    second = submit(store, refunds, contexts, requested_on=date(2027, 1, 1))
    assert first == second
    assert count(refunds) == 1


@pytest.mark.parametrize("change", [
    {"amount": "20.00"}, {"order_id": "O1002"},
    {"customer_id": "C003", "order_id": "O1005"},
])
def test_changed_payload_cannot_reuse_id(store, refunds, contexts, change):
    original = submit(store, refunds, contexts)
    assert submit(store, refunds, contexts, **change)["error"] == "request_conflict"
    assert refunds.get(original["data"]["refund_id"]).amount == Decimal("30.00")
    assert count(refunds) == 1


@pytest.mark.parametrize("changes,code", [
    ({"customer_id": "C999"}, "customer_not_found"),
    ({"order_id": "O999"}, "order_not_found"),
    ({"order_id": "O1003"}, "order_customer_mismatch"),
    ({"request_id": " "}, "invalid_refund_request"),
    ({"amount": "0"}, "invalid_refund_request"),
    ({"amount": "1.001"}, "invalid_refund_request"),
    ({"amount": "NaN"}, "invalid_refund_request"),
])
def test_invalid_or_unauthorized_requests_do_not_write(store, refunds, contexts, changes, code):
    assert submit(store, refunds, contexts, **changes)["error"] == code
    assert count(refunds) == 0


def test_normalizes_customer_and_order_ids(store, refunds, contexts):
    result = submit(store, refunds, contexts, customer_id=" c001 ", order_id=" o1001 ")
    assert result["data"]["customer_id"] == "C001"
    assert result["data"]["order_id"] == "O1001"


def test_context_cannot_override_authoritative_delivery_date(store, refunds, contexts):
    contexts._entries["O1002"].facts.delivered_date = date(2026, 11, 5)
    result = submit(store, refunds, contexts, order_id="O1002", amount="10")
    assert result["data"]["decision"]["status"] == "rejected"
    assert result["data"]["facts"]["delivered_date"] == "2026-08-25"


def test_missing_and_invalid_context_do_not_write(store, refunds, contexts):
    entry = contexts._entries.pop("O1001")
    assert submit(store, refunds, contexts)["error"] == "refund_context_missing"
    entry.refundable_balance = Decimal("100")
    contexts._entries["O1001"] = entry
    assert submit(store, refunds, contexts)["error"] == "invalid_refund_context"
    assert count(refunds) == 0


@pytest.mark.parametrize("customer,order,status", [
    ("C001", "O1002", "rejected"),
    ("C002", "O1003", "needs_information"),
    ("C003", "O1005", "manual_review"),
])
def test_nonapproved_decisions_are_saved(store, refunds, contexts, customer, order, status):
    result = submit(store, refunds, contexts, customer_id=customer, order_id=order, amount="10")
    assert result["ok"]
    assert result["data"]["decision"]["status"] == status
    assert count(refunds) == 1


def test_distinct_requests_reserve_remaining_balance(store, refunds, contexts):
    first = submit(store, refunds, contexts)
    second = submit(store, refunds, contexts, request_id="second")
    assert first["data"]["decision"]["status"] == "auto_approved"
    assert second["data"]["decision"]["status"] == "rejected"
    assert Decimal(second["data"]["available_balance"]) == Decimal("29.90")
    third = submit(store, refunds, contexts, request_id="third", amount="29.90")
    assert third["data"]["decision"]["status"] == "auto_approved"


def test_pending_approval_also_reserves_balance(store, refunds, contexts):
    submit(store, refunds, contexts, amount="59.90")
    second = submit(store, refunds, contexts, request_id="second", amount="0.01")
    assert second["data"]["decision"]["status"] == "rejected"


def test_concurrent_identical_retries_produce_one_row(store, refunds, contexts):
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: submit(store, refunds, contexts), range(4)))
    assert all(result == results[0] for result in results)
    assert count(refunds) == 1


def test_concurrent_distinct_requests_cannot_overreserve(store, refunds, contexts):
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(
            lambda key: submit(store, refunds, contexts, request_id=key), ["one", "two"]
        ))
    statuses = [result["data"]["decision"]["status"] for result in results]
    assert sorted(statuses) == ["auto_approved", "rejected"]


def test_storage_failure_is_not_reported_as_success(store, refunds, contexts, monkeypatch):
    def fail(*args, **kwargs):
        raise sqlite3.OperationalError("disk full")
    monkeypatch.setattr(refunds, "create", fail)
    assert submit(store, refunds, contexts)["error"] == "refund_storage_failed"
    assert count(refunds) == 0


def test_duplicate_context_ids_are_rejected(tmp_path):
    path = tmp_path / "facts.json"
    entry = {"order_id": "O1001", "facts": {}, "refundable_balance": "10"}
    path.write_text(json.dumps([entry, entry]), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        RefundContextStore(path)
