"""Validate customer/order details before persisting a complaint ticket."""

import sqlite3

from pydantic import ValidationError

from supportops.schemas import TicketRequest
from supportops.store import Store
from supportops.tickets import TicketConflictError, TicketStore
from supportops.tools.common import error, ok
from supportops.budget import budgeted_tool
from supportops.tracing import traced


@traced("tool.create_ticket", inputs=("request_id", "customer_id", "order_id", "description"))
@budgeted_tool
def create_ticket(
    store: Store,
    tickets: TicketStore,
    *,
    request_id: str,
    customer_id: str,
    order_id: str | None,
    description: str,
) -> dict:
    """Create a ticket for a supplied customer identity, without calling an LLM.

    The caller must bind customer_id to its authenticated/demo session. Checking
    an order's ownership here is not a replacement for authentication.
    """
    try:
        request = TicketRequest(
            request_id=request_id, customer_id=customer_id,
            order_id=order_id, description=description,
        )
    except ValidationError:
        return error(
            "invalid_ticket_request",
            "Provide a request ID, customer ID, and description of 1 to 5000 characters.",
        )

    customer = store.get_customer(request.customer_id.upper())
    if customer is None:
        return error("customer_not_found", "The customer could not be found.")

    normalized_order_id = request.order_id.upper() if request.order_id is not None else None
    if normalized_order_id is not None:
        order = store.get_order(normalized_order_id)
        if order is None:
            return error("order_not_found", "The order could not be found.")
        if order.customer_id != customer.customer_id:
            return error("order_customer_mismatch", "The order does not belong to this customer.")

    normalized_request = TicketRequest(
        request_id=request.request_id, customer_id=customer.customer_id,
        order_id=normalized_order_id, description=request.description,
    )
    try:
        ticket = tickets.create(normalized_request)
    except TicketConflictError:
        return error(
            "request_conflict", "This request ID was already used for different complaint details."
        )
    except (sqlite3.Error, OSError):
        return error("ticket_storage_failed", "The ticket could not be saved. Please try again.")
    return ok(ticket.model_dump(mode="json"))
