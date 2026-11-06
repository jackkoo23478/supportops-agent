"""Human-review transitions, audit records, retries, and reservation safety."""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from scripts.review_refund import main
from supportops.refund_store import (
    RefundContextStore, RefundNotFoundError, RefundReviewConflictError, RefundStore,
)
from supportops.store import default_data_dir
from supportops.tools.refunds import request_refund


@pytest.fixture
def setup_review(tmp_path, store):
    refunds = RefundStore(tmp_path / "reviews.db")
    contexts = RefundContextStore(default_data_dir() / "refund_contexts.json")

    def submit(request_id="pending", amount="59.90", **changes):
        args = dict(
            request_id=request_id, customer_id="C001", order_id="O1001", amount=amount,
            requested_on=date(2026, 11, 6),
        )
        args.update(changes)
        return request_refund(store, refunds, contexts, **args)["data"]

    return refunds, submit


def review(refunds, refund_id, action="approve", **changes):
    args = dict(action=action, reviewer="admin-jack", reason="Verified the return documents.")
    args.update(changes)
    return refunds.review(refund_id, **args)


@pytest.mark.parametrize("action,status", [("approve", "approved"), ("reject", "rejected")])
def test_persists_review_and_original_decision(setup_review, action, status):
    refunds, submit = setup_review
    pending = submit()
    record = review(refunds, pending["refund_id"], action)
    assert record.decision.status == status
    assert record.eligibility_decision.status == "pending_approval"
    assert record.review.action == action
    assert record.review.reviewer == "admin-jack"
    assert record.review.reviewed_at.utcoffset().total_seconds() == 0
    assert record.facts.model_dump(mode="json") == pending["facts"]
    assert RefundStore(refunds.path).get(record.refund_id) == record


def test_identical_retry_preserves_original_timestamp(setup_review):
    refunds, submit = setup_review
    refund_id = submit()["refund_id"]
    first = review(refunds, refund_id)
    assert review(refunds, refund_id) == first
    assert submit()["review"] == first.review.model_dump(mode="json")
    assert submit()["decision"]["status"] == "approved"


@pytest.mark.parametrize("changes", [
    {"action": "reject"}, {"reviewer": "another-admin"}, {"reason": "Changed my mind."},
])
def test_changed_review_cannot_overwrite_audit(setup_review, changes):
    refunds, submit = setup_review
    refund_id = submit()["refund_id"]
    first = review(refunds, refund_id)
    with pytest.raises(RefundReviewConflictError):
        review(refunds, refund_id, **changes)
    assert refunds.get(refund_id) == first


def test_rejected_review_cannot_later_be_approved(setup_review):
    refunds, submit = setup_review
    refund_id = submit()["refund_id"]
    review(refunds, refund_id, "reject")
    with pytest.raises(RefundReviewConflictError):
        review(refunds, refund_id, "approve")


@pytest.mark.parametrize("changes", [
    {"amount": "30"},
    {"order_id": "O1002", "amount": "10"},
    {"customer_id": "C003", "order_id": "O1005", "amount": "10"},
    {"customer_id": "C002", "order_id": "O1003", "amount": "10"},
])
def test_only_pending_approvals_can_be_reviewed(setup_review, changes):
    refunds, submit = setup_review
    refund_id = submit(**changes)["refund_id"]
    with pytest.raises(RefundReviewConflictError):
        review(refunds, refund_id)
    assert refunds.get(refund_id).review is None


def test_unknown_refund_is_not_created(setup_review):
    refunds, _ = setup_review
    with pytest.raises(RefundNotFoundError):
        review(refunds, "R-MISSING")
    assert refunds.get("R-MISSING") is None


@pytest.mark.parametrize("changes", [
    {"reviewer": " "}, {"reason": " "}, {"action": "execute"},
])
def test_invalid_reviews_do_not_change_pending_state(setup_review, changes):
    refunds, submit = setup_review
    refund_id = submit()["refund_id"]
    with pytest.raises(ValidationError):
        review(refunds, refund_id, **changes)
    assert refunds.get(refund_id).decision.status == "pending_approval"


def test_rejection_releases_reservation_but_same_request_stays_rejected(setup_review):
    refunds, submit = setup_review
    refund_id = submit()["refund_id"]
    review(refunds, refund_id, "reject")
    assert submit()["decision"]["status"] == "rejected"
    replacement = submit(request_id="new-request")
    assert replacement["decision"]["status"] == "pending_approval"
    assert Decimal(replacement["available_balance"]) == Decimal("59.90")


def test_approval_keeps_reservation(setup_review):
    refunds, submit = setup_review
    review(refunds, submit()["refund_id"])
    result = submit(request_id="new-request", amount="0.01")
    assert result["decision"]["status"] == "rejected"


def test_concurrent_identical_reviews_are_idempotent(setup_review):
    refunds, submit = setup_review
    refund_id = submit()["refund_id"]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: review(refunds, refund_id), range(4)))
    assert all(record == results[0] for record in results)


def test_concurrent_opposing_reviews_have_one_winner(setup_review):
    refunds, submit = setup_review
    refund_id = submit()["refund_id"]

    def attempt(action):
        try:
            return review(refunds, refund_id, action).review.action
        except RefundReviewConflictError:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, ["approve", "reject"]))
    assert results.count("conflict") == 1
    assert refunds.get(refund_id).review.action in results


def test_old_records_without_review_fields_remain_compatible(setup_review):
    refunds, submit = setup_review
    data = submit()
    data.pop("review")
    data.pop("eligibility_decision")
    with sqlite3.connect(refunds.path) as db:
        db.execute("UPDATE refund_requests SET record_json = ?", (json.dumps(data),))
    record = review(refunds, data["refund_id"])
    assert record.eligibility_decision.status == "pending_approval"
    assert record.decision.status == "approved"


def test_cli_reviews_and_reports_conflicts(setup_review, capsys):
    refunds, submit = setup_review
    refund_id = submit()["refund_id"]
    args = [
        "--db", str(refunds.path), "--refund-id", refund_id,
        "--decision", "approve", "--reviewer", "admin-jack", "--reason", "Return checked.",
    ]
    assert main(args) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["data"]["decision"]["status"] == "approved"
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out) == first
    args[args.index("approve")] = "reject"
    assert main(args) == 1
    assert json.loads(capsys.readouterr().out)["error"] == "review_conflict"
