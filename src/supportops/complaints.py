"""Collect an order-related complaint and execute the ticket tool."""

import re
import sqlite3
from pathlib import Path

from supportops.schemas import ConversationState, IntentResult, PendingComplaint
from supportops.store import Store
from supportops.tickets import TicketStore
from supportops.tools.tickets import create_ticket
from supportops.tracing import traced


class ComplaintService:
    """Load ticket storage on first action and reuse it across turns."""

    def __init__(
        self, tickets: TicketStore | None = None, *, db_path: Path | None = None
    ) -> None:
        self._tickets = tickets
        self._db_path = db_path

    @traced("complaint.handle", inputs=("message", "decision"))
    def handle(
        self, message: str, decision: IntentResult, *, store: Store, state: ConversationState
    ) -> str:
        if state.pending_complaint is None:
            if len(message) > 5000:
                return "Please describe the complaint in no more than 5000 characters."
            state.pending_complaint = PendingComplaint(description=message)
        pending = state.pending_complaint

        explicit_ids = set(re.findall(r"\bO\d+\b", message.upper()))
        if len(explicit_ids) > 1:
            return "Please provide a single order ID for this complaint."
        if explicit_ids:
            # IDs for a write action must actually occur in the customer's text.
            pending.order_id = explicit_ids.pop()
        elif decision.order_id:
            return "Please provide the order ID for this complaint."
        elif pending.order_id is None and decision.use_previous_order:
            pending.order_id = state.last_order_id

        if pending.order_id is None:
            return "Please provide the order ID for this complaint."

        try:
            if self._tickets is None:
                self._tickets = TicketStore(self._db_path)
            result = create_ticket(
                store, self._tickets, request_id=pending.request_id,
                customer_id=state.customer_id, order_id=pending.order_id,
                description=pending.description,
            )
        except (sqlite3.Error, OSError):
            return self._retry_message()

        if not result["ok"]:
            code = result["error"]
            if code in {"order_not_found", "order_customer_mismatch"}:
                pending.order_id = None
                state.last_order_id = None
                return (
                    "I could not find that order for this customer. "
                    "Please provide the correct order ID. No ticket has been created."
                )
            if code == "ticket_storage_failed":
                return self._retry_message()
            if code == "request_conflict":
                return (
                    "This request ID already belongs to different complaint details. "
                    "The existing ticket was not changed. Type 'cancel complaint' "
                    "before starting a new complaint."
                )
            return f"{result['message']} The complaint remains pending."

        ticket = result["data"]
        state.last_ticket_id = ticket["ticket_id"]
        state.last_action = "complaint"
        state.last_order_id = ticket["order_id"]
        state.pending_complaint = None
        return (
            "Your complaint has been recorded.\n"
            f"Ticket ID: {ticket['ticket_id']}\n"
            f"Order: {ticket['order_id']}\n"
            "Status: open"
        )

    @staticmethod
    def _retry_message() -> str:
        return (
            "I could not confirm that the ticket was saved. "
            "Type 'retry' to retry the same request, or 'cancel complaint'."
        )
