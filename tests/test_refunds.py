"""Policy boundaries and missing facts must never authorize unsafe refunds."""

from datetime import date, timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from supportops.refunds import RefundFacts, evaluate_refund


@pytest.fixture
def facts():
    return RefundFacts(
        delivered_date=date(2026, 10, 20), item_type="standard", unused=True,
        original_packaging=True, proof_of_purchase=True, damaged_or_defective=False,
        return_received=True, inspection_passed=True,
    )


def decide(facts, amount="30.00", **kwargs):
    return evaluate_refund(
        facts, amount=Decimal(amount), refundable_balance=Decimal("100.00"),
        requested_on=kwargs.get("requested_on", date(2026, 11, 5)),
    )


@pytest.mark.parametrize("amount,status", [
    ("0.01", "auto_approved"), ("50.00", "auto_approved"),
    ("50.01", "pending_approval"), ("100.00", "pending_approval"),
])
def test_approval_amount_boundary(facts, amount, status):
    assert decide(facts, amount).status == status


@pytest.mark.parametrize("amount", ["0", "-1", "100.01", "1.001", "NaN", "Infinity"])
def test_invalid_amounts_never_approve(facts, amount):
    assert decide(facts, amount).status == "rejected"


@pytest.mark.parametrize("balance", ["-1", "NaN", "Infinity", "10.001"])
def test_invalid_balances_never_approve(facts, balance):
    decision = evaluate_refund(
        facts, amount=Decimal("1"), refundable_balance=Decimal(balance),
        requested_on=date(2026, 11, 5),
    )
    assert decision.status == "rejected"


def test_float_money_is_not_accepted(facts):
    with pytest.raises(TypeError, match="Decimal"):
        evaluate_refund(
            facts, amount=30.0, refundable_balance=Decimal("100"),
            requested_on=date(2026, 11, 5),
        )


@pytest.mark.parametrize("days,status", [
    (-1, "manual_review"), (0, "auto_approved"),
    (30, "auto_approved"), (31, "rejected"),
])
def test_return_window_uses_request_date(facts, days, status):
    requested_on = facts.delivered_date + timedelta(days=days)
    assert decide(facts, requested_on=requested_on).status == status


@pytest.mark.parametrize("field", [
    "delivered_date", "item_type", "unused", "original_packaging",
    "proof_of_purchase", "damaged_or_defective", "return_received",
])
def test_missing_verified_facts_are_reported(facts, field):
    setattr(facts, field, None)
    decision = decide(facts)
    assert decision.status == "needs_information"
    assert field in decision.missing_fields


@pytest.mark.parametrize("changes", [
    {"item_type": "personalized"}, {"item_type": "digital"},
    {"unused": False}, {"original_packaging": False}, {"proof_of_purchase": False},
])
def test_standard_return_exclusions(facts, changes):
    assert decide(facts.model_copy(update=changes)).status == "rejected"


def test_damage_requires_review_even_with_incomplete_facts():
    facts = RefundFacts(damaged_or_defective=True)
    assert decide(facts).status == "manual_review"


@pytest.mark.parametrize("changes,status", [
    ({"return_received": False}, "awaiting_return"),
    ({"inspection_passed": None}, "awaiting_inspection"),
    ({"inspection_passed": False}, "manual_review"),
])
def test_no_execution_approval_before_successful_inspection(facts, changes, status):
    assert decide(facts.model_copy(update=changes)).status == status


def test_customer_strings_are_not_coerced_into_verified_booleans():
    with pytest.raises(ValidationError):
        RefundFacts(unused="yes")
