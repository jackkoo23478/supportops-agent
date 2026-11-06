"""Persist support tickets in SQLite."""

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from supportops.schemas import Ticket, TicketRequest


class TicketConflictError(Exception):
    """A request ID was reused with different complaint details."""


class TicketStore:
    def __init__(self, path: Path | None = None) -> None:
        if path is None:
            path = Path(__file__).resolve().parents[2] / "data" / "runtime" / "supportops.db"
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

        with closing(sqlite3.connect(self.path)) as db:
            with db:
                db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS tickets (
                        ticket_id TEXT PRIMARY KEY,
                        request_id TEXT NOT NULL UNIQUE,
                        customer_id TEXT NOT NULL,
                        order_id TEXT,
                        description TEXT NOT NULL,
                        status TEXT NOT NULL CHECK (status IN ('open', 'closed')),
                        created_at TEXT NOT NULL
                    )
                    """
                )

    def create(self, request: TicketRequest) -> Ticket:
        """Create a ticket, or return an identical retry's persisted result."""
        ticket_id = f"T-{uuid4().hex.upper()}"
        created_at = datetime.now(timezone.utc).isoformat()

        with closing(sqlite3.connect(self.path)) as db:
            db.row_factory = sqlite3.Row
            with db:
                db.execute(
                    """
                    INSERT INTO tickets (
                        ticket_id, request_id, customer_id, order_id,
                        description, status, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(request_id) DO NOTHING
                    """,
                    (
                        ticket_id, request.request_id, request.customer_id,
                        request.order_id, request.description, "open", created_at,
                    ),
                )
                row = db.execute(
                    "SELECT * FROM tickets WHERE request_id = ?", (request.request_id,)
                ).fetchone()
                if row is None:
                    raise RuntimeError("The ticket could not be retrieved.")
                ticket = Ticket.model_validate(dict(row))
                if (
                    ticket.customer_id != request.customer_id
                    or ticket.order_id != request.order_id
                    or ticket.description != request.description
                ):
                    raise TicketConflictError(
                        "The request ID belongs to different complaint details."
                    )
                return ticket

    def get(self, ticket_id: str) -> Ticket | None:
        """Retrieve a persisted ticket by its server-generated ID."""
        with closing(sqlite3.connect(self.path)) as db:
            db.row_factory = sqlite3.Row
            row = db.execute(
                "SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,)
            ).fetchone()
            return Ticket.model_validate(dict(row)) if row is not None else None
