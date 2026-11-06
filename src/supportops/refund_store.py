"""Persist refund decisions and reserve balances; never execute payments."""

import json
import sqlite3
from contextlib import closing
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from supportops.refunds import RefundDecision, RefundFacts, evaluate_refund
from supportops.tracing import traced


class RefundRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    request_id: str = Field(min_length=1, max_length=200)
    customer_id: str = Field(min_length=1)
    order_id: str = Field(min_length=1)
    amount: Decimal = Field(gt=0, max_digits=12, decimal_places=2, allow_inf_nan=False)
    currency: str = Field(default="USD", pattern=r"^USD$")


class RefundReview(BaseModel):
    """Audit metadata supplied by a trusted local administrator."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    action: Literal["approve", "reject"]
    reviewer: str = Field(min_length=1, max_length=200)
    reason: str = Field(min_length=1, max_length=2000)
    reviewed_at: datetime


class RefundRecord(RefundRequest):
    refund_id: str
    requested_on: date
    created_at: datetime
    decision: RefundDecision
    facts: RefundFacts
    available_balance: Decimal
    eligibility_decision: RefundDecision | None = None
    review: RefundReview | None = None
    authorization_decision: RefundDecision | None = None
    execution: "RefundTransaction | None" = None


class RefundTransaction(BaseModel):
    """A local simulation ledger entry, never an external payment receipt."""

    transaction_id: str
    refund_id: str
    customer_id: str
    order_id: str
    amount: Decimal = Field(gt=0, max_digits=12, decimal_places=2, allow_inf_nan=False)
    currency: Literal["USD"] = "USD"
    simulated: Literal[True] = True
    payment_method: Literal["original_payment_method"] = "original_payment_method"
    executed_at: datetime


class RefundConflictError(Exception):
    """An idempotency key belongs to a different request."""


class RefundNotFoundError(Exception):
    """The requested refund does not exist."""


class RefundReviewConflictError(Exception):
    """A refund is not pending approval or already has another review."""


class RefundExecutionError(Exception):
    """Authorization, ledger consistency, or available balance prevents execution."""


class RefundStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path if path is not None else (
            Path(__file__).resolve().parents[2] / "data" / "runtime" / "supportops.db"
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS refund_requests (
                    refund_id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL UNIQUE,
                    customer_id TEXT NOT NULL,
                    order_id TEXT NOT NULL,
                    record_json TEXT NOT NULL
                )
            """)
            db.execute("""
                CREATE TABLE IF NOT EXISTS refund_transactions (
                    transaction_id TEXT PRIMARY KEY,
                    refund_id TEXT NOT NULL UNIQUE,
                    transaction_json TEXT NOT NULL
                )
            """)

    def get(self, refund_id: str) -> RefundRecord | None:
        with closing(sqlite3.connect(self.path)) as db:
            row = db.execute(
                "SELECT record_json FROM refund_requests WHERE refund_id = ?", (refund_id,)
            ).fetchone()
        return RefundRecord.model_validate_json(row[0]) if row else None

    @traced("refund.save", inputs=("request", "facts", "refundable_balance", "requested_on"))
    def create(
        self, request: RefundRequest, *, facts: RefundFacts,
        refundable_balance: Decimal, requested_on: date,
    ) -> RefundRecord:
        """Persist a request, returning its current record on identical retries.

        Reserve auto-approved and pending-approval amounts in the same transaction
        so concurrent requests cannot authorize more than the supplied balance.
        The supplied balance excludes completed historical refunds but must not
        already subtract reservations in this table. No payment is performed.
        """
        with closing(sqlite3.connect(self.path, timeout=15)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT record_json FROM refund_requests WHERE request_id = ?",
                (request.request_id,),
            ).fetchone()
            if row:
                record = RefundRecord.model_validate_json(row[0])
                original = {key: getattr(record, key) for key in type(request).model_fields}
                if original != request.model_dump():
                    raise RefundConflictError("The request ID has different refund details.")
                return record

            reserved = Decimal("0")
            for (raw,) in db.execute(
                "SELECT record_json FROM refund_requests WHERE order_id = ?",
                (request.order_id,),
            ):
                previous = RefundRecord.model_validate_json(raw)
                if previous.decision.status in {
                    "auto_approved", "pending_approval", "approved", "executed",
                }:
                    reserved += previous.amount
            available = refundable_balance - reserved
            decision = evaluate_refund(
                facts, amount=request.amount, refundable_balance=available,
                requested_on=requested_on,
            )
            record = RefundRecord(
                **request.model_dump(), refund_id=f"R-{uuid4().hex.upper()}",
                requested_on=requested_on, created_at=datetime.now(timezone.utc),
                decision=decision, eligibility_decision=decision.model_copy(deep=True),
                facts=facts, available_balance=available,
            )
            db.execute(
                "INSERT INTO refund_requests VALUES (?, ?, ?, ?, ?)",
                (record.refund_id, record.request_id, record.customer_id,
                 record.order_id, record.model_dump_json()),
            )
            return record

    @traced("refund.execute", inputs=("refund_id", "facts", "refundable_balance"))
    def execute(
        self, refund_id: str, *, facts: RefundFacts, refundable_balance: Decimal,
    ) -> RefundRecord:
        """Atomically authorize and record a simulated refund exactly once.

        Balance is the trusted baseline after external historical refunds, before
        this database's reservations AND simulated transactions. Executed amounts
        remain consumed, so completing a refund never restores spending capacity.
        """
        with closing(sqlite3.connect(self.path, timeout=15)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT record_json FROM refund_requests WHERE refund_id = ?", (refund_id,)
            ).fetchone()
            if row is None:
                raise RefundNotFoundError("The refund request could not be found.")
            record = RefundRecord.model_validate_json(row[0])
            existing = db.execute(
                "SELECT transaction_json FROM refund_transactions WHERE refund_id = ?",
                (refund_id,),
            ).fetchone()
            if record.decision.status == "executed":
                if existing and record.execution == RefundTransaction.model_validate_json(existing[0]):
                    return record
                raise RefundExecutionError("The execution record and ledger do not match.")
            if existing or record.execution is not None:
                raise RefundExecutionError("The refund has an inconsistent transaction record.")
            if record.decision.status not in {"auto_approved", "approved"}:
                raise RefundExecutionError("Only approved refunds can be executed.")
            if record.decision.status == "auto_approved" and record.amount > Decimal("50.00"):
                raise RefundExecutionError("Automatic approval cannot exceed USD 50.00.")
            if record.decision.status == "approved" and (
                record.review is None or record.review.action != "approve"
            ):
                raise RefundExecutionError("A saved human approval is required.")

            committed = Decimal("0")
            for (raw,) in db.execute(
                "SELECT record_json FROM refund_requests WHERE order_id = ? AND refund_id != ?",
                (record.order_id, refund_id),
            ):
                other = RefundRecord.model_validate_json(raw)
                if other.decision.status in {
                    "auto_approved", "pending_approval", "approved", "executed",
                }:
                    committed += other.amount
            eligibility = evaluate_refund(
                facts, amount=record.amount, refundable_balance=refundable_balance - committed,
                requested_on=record.requested_on,
            )
            if eligibility.status not in {"auto_approved", "pending_approval"}:
                raise RefundExecutionError(eligibility.reason)

            transaction = RefundTransaction(
                transaction_id=f"SIM-{uuid4().hex.upper()}", refund_id=record.refund_id,
                customer_id=record.customer_id, order_id=record.order_id,
                amount=record.amount, executed_at=datetime.now(timezone.utc),
            )
            record.authorization_decision = record.decision.model_copy(deep=True)
            record.execution = transaction
            record.decision = RefundDecision(
                status="executed", reason="Simulated refund recorded. No real payment was sent.",
            )
            db.execute(
                "INSERT INTO refund_transactions VALUES (?, ?, ?)",
                (transaction.transaction_id, refund_id, transaction.model_dump_json()),
            )
            db.execute(
                "UPDATE refund_requests SET record_json = ? WHERE refund_id = ?",
                (record.model_dump_json(), refund_id),
            )
            return record

    @traced("refund.review", inputs=("refund_id", "action", "reviewer", "reason"))
    def review(
        self, refund_id: str, *, action: Literal["approve", "reject"],
        reviewer: str, reason: str,
    ) -> RefundRecord:
        """Review only pending approvals in a serialized transaction; never pay.

        This is an administrator API, not a customer/LLM tool. Its caller must
        enforce access to the local database and provide a trusted reviewer.
        Exact retries return the saved audit metadata, including its timestamp.
        """
        review = RefundReview(
            action=action, reviewer=reviewer, reason=reason,
            reviewed_at=datetime.now(timezone.utc),
        )
        with closing(sqlite3.connect(self.path, timeout=15)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT record_json FROM refund_requests WHERE refund_id = ?", (refund_id,)
            ).fetchone()
            if row is None:
                raise RefundNotFoundError("The refund request could not be found.")
            record = RefundRecord.model_validate_json(row[0])
            if record.review is not None:
                saved = record.review
                if (saved.action, saved.reviewer, saved.reason) == (
                    review.action, review.reviewer, review.reason,
                ):
                    return record
                raise RefundReviewConflictError("This refund already has a different review.")
            if record.decision.status != "pending_approval":
                raise RefundReviewConflictError("Only pending approvals can be reviewed.")
            if record.eligibility_decision is None:
                # Existing JSON records need no destructive database migration.
                record.eligibility_decision = record.decision.model_copy(deep=True)
            record.decision = RefundDecision(
                status="approved" if review.action == "approve" else "rejected",
                reason=f"Human review: {review.reason}",
            )
            record.review = review
            db.execute(
                "UPDATE refund_requests SET record_json = ? WHERE refund_id = ?",
                (record.model_dump_json(), refund_id),
            )
            return record


class RefundContext(BaseModel):
    """Server-maintained sample facts and balance after historical refunds."""

    model_config = ConfigDict(extra="forbid")
    order_id: str = Field(min_length=1)
    facts: RefundFacts
    refundable_balance: Decimal = Field(
        ge=0, max_digits=12, decimal_places=2, allow_inf_nan=False,
    )


class RefundContextStore:
    def __init__(self, path: Path) -> None:
        entries = [RefundContext.model_validate(item) for item in json.loads(
            path.read_text(encoding="utf-8")
        )]
        if len({entry.order_id for entry in entries}) != len(entries):
            raise ValueError("Refund context order IDs must be unique.")
        self._entries = {entry.order_id: entry for entry in entries}

    def get(self, order_id: str) -> RefundContext | None:
        entry = self._entries.get(order_id)
        return entry.model_copy(deep=True) if entry else None
