"""Collect and confirm refund drafts; never expose administrator approval."""

import re
import sqlite3
from datetime import date
from decimal import Decimal
from pathlib import Path

from pydantic import ValidationError

from supportops.refund_store import RefundContextStore, RefundRecord, RefundRequest, RefundStore
from supportops.schemas import ConversationState, IntentResult, PendingRefund
from supportops.store import Store, default_data_dir
from supportops.tools.refunds import execute_refund, request_refund
from supportops.tracing import traced


def extract_amount(message: str) -> Decimal | None:
    """Accept one explicit USD amount or a bare amount in a pending draft."""
    if re.search(
        r"\b(?:EUR|GBP|HKD|CNY|RMB|CAD|AUD|JPY)(?=\s|\d|\$)|HK\$|[€£¥]",
        message, re.IGNORECASE,
    ):
        raise ValueError("Only USD merchandise refunds are supported.")
    number = r"[+-]?\d[\d,]*(?:\.\d+)?"
    bare = re.fullmatch(rf"(?:USD\s*|\$\s*)?({number})(?:\s*USD)?", message, re.IGNORECASE)
    if bare:
        values = [bare.group(1)]
    else:
        matches = re.findall(
            rf"(?:\bUSD\s*|\$\s*)({number})(?!\w|\.\d)|"
            rf"(?<![\w.])({number})\s*USD\b", message, re.IGNORECASE,
        )
        values = [first or second for first, second in matches]
    if not values:
        return None
    if len(values) != 1:
        raise ValueError("Please provide exactly one refund amount in USD.")
    if "," in values[0]:
        raise ValueError("Enter the USD amount without commas, for example USD 30.00.")
    try:
        validated = RefundRequest(
            request_id="validation", customer_id="validation", order_id="validation",
            amount=values[0],
        )
    except ValidationError as exc:
        raise ValueError("Enter a positive USD amount with no more than two decimal places.") from exc
    return validated.amount


class RefundService:
    def __init__(
        self, refunds: RefundStore | None = None, contexts: RefundContextStore | None = None,
        *, db_path: Path | None = None, requested_on: date | None = None,
    ) -> None:
        self._refunds = refunds
        self._contexts = contexts
        self._db_path = db_path
        self._requested_on = requested_on

    def _load(self) -> tuple[RefundStore, RefundContextStore]:
        if self._refunds is None:
            self._refunds = RefundStore(self._db_path)
        if self._contexts is None:
            self._contexts = RefundContextStore(default_data_dir() / "refund_contexts.json")
        return self._refunds, self._contexts

    @traced("refund.handle", inputs=("message", "decision"))
    def handle(
        self, message: str, decision: IntentResult, *, store: Store, state: ConversationState,
    ) -> str:
        if state.pending_complaint is not None:
            return "Finish or cancel the pending complaint before starting a refund draft."
        command = message.lower()
        if state.pending_refund is None:
            if command in {"retry", "retry refund", "confirm refund"}:
                return "There is no pending refund draft."
            state.pending_refund = PendingRefund()
        pending = state.pending_refund
        if pending.confirmed:
            if command not in {"retry", "retry refund", "confirm refund"}:
                return "Submission is unresolved. Use 'retry refund' with the same details."
            return self._submit(store, state)
        try:
            amount = extract_amount(message)
        except ValueError as exc:
            pending.amount = None
            return str(exc)
        ids = set(re.findall(r"\bO\d+\b", message.upper()))
        if len(ids) > 1:
            return "Please provide a single order ID for this refund."
        if ids:
            order_id = ids.pop()
            if pending.order_id is not None and pending.order_id != order_id:
                # Never carry an amount silently to a different order.
                pending.amount = None
            pending.order_id = order_id
        elif decision.order_id:
            return "Please provide the order ID in your message."
        elif pending.order_id is None and decision.use_previous_order:
            pending.order_id = state.last_order_id
        if amount is not None:
            pending.amount = amount
        if pending.order_id is None:
            return "Please provide the order ID for this refund."
        customer = store.get_customer(state.customer_id)
        order = store.get_order(pending.order_id)
        if customer is None or order is None or order.customer_id != customer.customer_id:
            pending.order_id = None
            pending.amount = None
            return "I could not find that order for this customer. Provide the correct order ID."
        if pending.amount is None:
            return "How much would you like to request? Enter an amount such as USD 30.00."
        if command == "confirm refund":
            pending.confirmed = True
            return self._submit(store, state)
        return (
            f"Refund draft: {pending.order_id}, USD {pending.amount:.2f}. "
            "Type 'confirm refund' to submit and execute if automatically approved, "
            "or 'cancel refund' to discard the draft. No refund has been executed."
        )

    def _submit(self, store: Store, state: ConversationState) -> str:
        pending = state.pending_refund
        assert pending is not None and pending.order_id is not None and pending.amount is not None
        try:
            refunds, contexts = self._load()
            result = request_refund(
                store, refunds, contexts, request_id=pending.request_id,
                customer_id=state.customer_id, order_id=pending.order_id,
                amount=pending.amount, requested_on=self._requested_on,
            )
        except (OSError, sqlite3.Error, ValueError):
            return "The submission could not be confirmed. Use 'retry refund'; details remain locked."
        if not result["ok"]:
            return f"{result['message']} Use 'retry refund' with the same request."
        record = RefundRecord.model_validate(result["data"])
        state.last_refund_id = record.refund_id
        state.last_order_id = record.order_id
        state.last_action = "refund"
        state.pending_refund = None
        if record.decision.status in {"auto_approved", "approved"}:
            return self.resume(record.refund_id, store=store, state=state, execute=True)
        return self.render(record)

    @traced("refund.resume", inputs=("refund_id", "execute"))
    def resume(
        self, refund_id: str, *, store: Store, state: ConversationState, execute: bool,
    ) -> str:
        try:
            refunds, contexts = self._load()
            record = refunds.get(refund_id)
            customer = store.get_customer(state.customer_id)
            if record is None or customer is None or record.customer_id != customer.customer_id:
                return "That refund request is not available to this customer."
            order = store.get_order(record.order_id)
            if order is None or order.customer_id != customer.customer_id:
                return "That refund request is not available to this customer."
            if execute:
                result = execute_refund(
                    store, refunds, contexts, refund_id=record.refund_id,
                    customer_id=state.customer_id,
                )
                if not result["ok"]:
                    if result["error"] == "refund_storage_failed":
                        return (
                            f"Refund ID: {record.refund_id}\n"
                            "Execution could not be confirmed. "
                            f"Use 'execute refund {record.refund_id}' to retry the same operation."
                        )
                    return (
                        f"{self.render(record)}\n{result['message']} "
                        f"Use 'execute refund {record.refund_id}' to retry after resolving the issue."
                    )
                record = RefundRecord.model_validate(result["data"])
            state.last_refund_id = record.refund_id
            state.last_action = "refund"
            return self.render(record)
        except (OSError, sqlite3.Error, ValueError):
            return f"The refund result is unavailable. Retry using refund ID {refund_id}."

    @staticmethod
    def render(record: RefundRecord) -> str:
        lines = [
            f"Refund ID: {record.refund_id}", f"Order: {record.order_id}",
            f"Amount: USD {record.amount:.2f}", f"Status: {record.decision.status}",
            record.decision.reason,
        ]
        if record.execution is not None:
            lines += [f"Transaction ID: {record.execution.transaction_id}",
                      "This was a simulated refund. No real money was transferred."]
        else:
            lines.append("No refund has been executed.")
        if record.decision.status == "pending_approval":
            lines.append("Human approval is required through the separate administrator command.")
        if record.decision.missing_fields:
            lines.append("Missing verified facts: " + ", ".join(record.decision.missing_fields))
        return "\n".join(lines)
