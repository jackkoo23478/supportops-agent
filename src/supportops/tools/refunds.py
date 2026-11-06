"""Validate order ownership and persist a refund request without an LLM."""

import sqlite3
from datetime import date
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from supportops.refund_store import (
    RefundConflictError, RefundContextStore, RefundExecutionError,
    RefundNotFoundError, RefundRequest, RefundStore,
)
from supportops.store import Store
from supportops.tools.common import error, ok
from supportops.tracing import traced


@traced("tool.request_refund", inputs=(
    "request_id", "customer_id", "order_id", "amount", "requested_on",
))
def request_refund(
    store: Store, refunds: RefundStore, contexts: RefundContextStore, *,
    request_id: str, customer_id: str, order_id: str, amount: str | Decimal,
    requested_on: date | None = None,
) -> dict:
    """Bind identity/date/context from the application, never from model claims.

    requested_on is a trusted application/test clock override. This tool handles
    USD merchandise only, creates no payment and provides no approval bypass.
    """
    try:
        request = RefundRequest(
            request_id=request_id, customer_id=customer_id, order_id=order_id, amount=amount,
        )
    except ValidationError:
        return error("invalid_refund_request", "Provide IDs and a positive USD amount in cents.")
    customer = store.get_customer(request.customer_id.upper())
    if customer is None:
        return error("customer_not_found", "The customer could not be found.")
    order = store.get_order(request.order_id.upper())
    if order is None:
        return error("order_not_found", "The order could not be found.")
    if order.customer_id != customer.customer_id:
        return error("order_customer_mismatch", "The order does not belong to this customer.")
    context = contexts.get(order.order_id)
    if context is None:
        return error("refund_context_missing", "Verified refund facts and balance are unavailable.")

    # The order store is authoritative for delivery dates. Facts cannot override it.
    facts = context.facts.model_copy(update={"delivered_date": order.delivered_date})
    merchandise_total = sum(
        (Decimal(str(item.unit_price)) * item.quantity for item in order.items), Decimal("0")
    )
    if context.refundable_balance > merchandise_total:
        return error("invalid_refund_context", "The verified balance exceeds merchandise value.")
    normalized = request.model_copy(update={
        "customer_id": customer.customer_id, "order_id": order.order_id,
    })
    try:
        record = refunds.create(
            normalized, facts=facts, refundable_balance=context.refundable_balance,
            requested_on=requested_on if requested_on is not None else date.today(),
        )
    except RefundConflictError:
        return error("request_conflict", "This request ID was used for different refund details.")
    except (sqlite3.Error, OSError):
        return error("refund_storage_failed", "The refund request could not be saved. Retry it.")
    return ok(record.model_dump(mode="json"))


class RefundExecutionInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    customer_id: str = Field(min_length=1)
    refund_id: str = Field(min_length=1)


@traced("tool.execute_refund", inputs=("refund_id", "customer_id"))
def execute_refund(
    store: Store, refunds: RefundStore, contexts: RefundContextStore, *,
    refund_id: str, customer_id: str,
) -> dict:
    """Execute a local simulation using persisted amount and approval only.

    The application binds customer_id to its trusted/demo session. Neither the
    customer nor model can supply an amount, approval flag, or execution date.
    """
    try:
        args = RefundExecutionInput(refund_id=refund_id, customer_id=customer_id)
    except ValidationError:
        return error("invalid_refund_execution", "Provide a refund ID and customer ID.")
    try:
        record = refunds.get(args.refund_id)
        if record is None:
            return error("refund_not_found", "The refund request could not be found.")
        customer = store.get_customer(args.customer_id.upper())
        if customer is None or record.customer_id != customer.customer_id:
            return error("refund_customer_mismatch", "The refund is not available to this customer.")
        order = store.get_order(record.order_id)
        if order is None or order.customer_id != customer.customer_id:
            return error("order_customer_mismatch", "The order does not belong to this customer.")
        context = contexts.get(order.order_id)
        if context is None:
            return error("refund_context_missing", "Verified refund facts and balance are unavailable.")
        merchandise_total = sum(
            (Decimal(str(item.unit_price)) * item.quantity for item in order.items), Decimal("0")
        )
        if context.refundable_balance > merchandise_total:
            return error("invalid_refund_context", "The verified balance exceeds merchandise value.")
        facts = context.facts.model_copy(update={"delivered_date": order.delivered_date})
        executed = refunds.execute(
            record.refund_id, facts=facts, refundable_balance=context.refundable_balance,
        )
    except RefundNotFoundError as exc:
        return error("refund_not_found", str(exc))
    except RefundExecutionError as exc:
        return error("refund_execution_blocked", str(exc))
    except (sqlite3.Error, OSError):
        return error("refund_storage_failed", "The simulated refund could not be confirmed. Retry it.")
    return ok(executed.model_dump(mode="json"))
