"""Evaluate verified refund facts without executing a payment."""

from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool

from supportops.tracing import traced


class RefundFacts(BaseModel):
    """Application-verified facts, not unverified customer or model claims."""

    model_config = ConfigDict(extra="forbid")

    delivered_date: date | None = None
    item_type: Literal["standard", "personalized", "digital"] | None = None
    unused: StrictBool | None = None
    original_packaging: StrictBool | None = None
    proof_of_purchase: StrictBool | None = None
    damaged_or_defective: StrictBool | None = None
    return_received: StrictBool | None = None
    inspection_passed: StrictBool | None = None


class RefundDecision(BaseModel):
    status: Literal[
        "needs_information", "rejected", "manual_review", "awaiting_return",
        "awaiting_inspection", "pending_approval", "auto_approved", "approved", "executed",
    ]
    reason: str
    missing_fields: list[str] = Field(default_factory=list)


def _valid_money(value: Decimal) -> bool:
    if not isinstance(value, Decimal):
        raise TypeError("Money must be supplied as Decimal, not float.")
    if not value.is_finite():
        return False
    try:
        return value == value.quantize(Decimal("0.01"))
    except InvalidOperation:
        return False


@traced("refund.evaluate", inputs=("facts", "amount", "refundable_balance", "requested_on"))
def evaluate_refund(
    facts: RefundFacts,
    *,
    amount: Decimal,
    refundable_balance: Decimal,
    requested_on: date,
) -> RefundDecision:
    """Apply sample USD merchandise policies using the original request date.

    The caller must verify customer/order ownership and load trusted facts and
    the remaining merchandise balance. Approval here does not execute a refund;
    an executor must recheck authorization and balance in its transaction.
    """
    if (
        not _valid_money(amount)
        or not _valid_money(refundable_balance)
        or amount <= 0
        or refundable_balance < 0
        or amount > refundable_balance
    ):
        return RefundDecision(
            status="rejected",
            reason="The amount is invalid or exceeds the refundable balance.",
        )

    # Damage exceptions always require review, even if other facts are missing.
    if facts.damaged_or_defective is True:
        return RefundDecision(
            status="manual_review",
            reason=(
                "Damaged or defective items require manual review. "
                "The policy asks customers to report them within 7 days of delivery."
            ),
        )

    required = (
        "delivered_date", "item_type", "unused", "original_packaging",
        "proof_of_purchase", "damaged_or_defective",
    )
    missing = [name for name in required if getattr(facts, name) is None]
    if missing:
        return RefundDecision(
            status="needs_information",
            reason="More verified information is required.",
            missing_fields=missing,
        )

    delivered_date = facts.delivered_date
    assert delivered_date is not None
    days_since_delivery = (requested_on - delivered_date).days
    if days_since_delivery < 0:
        return RefundDecision(
            status="manual_review",
            reason="The delivery date is later than the request date.",
        )
    if days_since_delivery > 30:
        return RefundDecision(
            status="rejected", reason="The request is outside the 30-day return window."
        )
    if (
        facts.item_type != "standard"
        or not facts.unused
        or not facts.original_packaging
        or not facts.proof_of_purchase
    ):
        return RefundDecision(
            status="rejected",
            reason="The request does not satisfy the standard return conditions.",
        )

    if facts.return_received is None:
        return RefundDecision(
            status="needs_information",
            reason="Confirm whether the returned item has been received.",
            missing_fields=["return_received"],
        )
    if not facts.return_received:
        return RefundDecision(
            status="awaiting_return", reason="The returned item has not been received."
        )
    if facts.inspection_passed is False:
        return RefundDecision(
            status="manual_review", reason="The returned item failed inspection."
        )
    if facts.inspection_passed is None:
        return RefundDecision(
            status="awaiting_inspection",
            reason="A successful inspection is required before refund execution.",
        )
    if amount > Decimal("50.00"):
        return RefundDecision(
            status="pending_approval",
            reason="Refunds above USD 50.00 require human approval.",
        )
    return RefundDecision(
        status="auto_approved",
        reason="The request satisfies the automatic approval conditions.",
    )
