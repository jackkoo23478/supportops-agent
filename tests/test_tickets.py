"""Persistence, ownership, failure, and idempotency checks for ticket creation."""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from unittest.mock import Mock

import pytest

from supportops.tickets import TicketStore
from supportops.tools.tickets import create_ticket


@pytest.fixture
def tickets(tmp_path):
    return TicketStore(tmp_path / "runtime" / "supportops.db")


@pytest.fixture
def details():
    return {
        "request_id": "complaint-001", "customer_id": "C001",
        "order_id": "O1001", "description": "The item arrived damaged.",
    }


def count_tickets(tickets):
    with closing(sqlite3.connect(tickets.path)) as db:
        return db.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]


def test_ticket_is_saved_and_can_be_reopened(store, tickets, details):
    result = create_ticket(store, tickets, **details)
    assert result["ok"] is True
    assert result["data"]["ticket_id"].startswith("T-")
    assert result["data"]["status"] == "open"
    json.dumps(result)
    reopened = TicketStore(tickets.path)
    saved = reopened.get(result["data"]["ticket_id"])
    assert saved is not None
    assert saved.description == details["description"]
    assert saved.order_id == "O1001"
    assert saved.created_at.utcoffset().total_seconds() == 0


def test_identical_retry_returns_same_ticket_and_timestamp(store, tickets, details):
    first = create_ticket(store, tickets, **details)
    retry = create_ticket(store, tickets, **details)
    assert retry == first
    assert count_tickets(tickets) == 1


def test_simultaneous_retries_create_only_one_ticket(store, tickets, details):
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: create_ticket(store, tickets, **details), range(2)))
    assert all(result["ok"] for result in results)
    assert results[0]["data"]["ticket_id"] == results[1]["data"]["ticket_id"]
    assert count_tickets(tickets) == 1


def test_new_request_id_creates_new_action_even_with_same_description(store, tickets, details):
    first = create_ticket(store, tickets, **details)
    second = create_ticket(store, tickets, **{**details, "request_id": "complaint-002"})
    assert first["data"]["ticket_id"] != second["data"]["ticket_id"]
    assert count_tickets(tickets) == 2


def test_request_id_conflict_keeps_original_ticket(store, tickets, details):
    first = create_ticket(store, tickets, **details)
    conflict = create_ticket(store, tickets, **{**details, "description": "Another issue."})
    assert conflict["error"] == "request_conflict"
    assert count_tickets(tickets) == 1
    assert tickets.get(first["data"]["ticket_id"]).description == details["description"]


@pytest.mark.parametrize(("changes", "code"), [
    ({"customer_id": "C999"}, "customer_not_found"),
    ({"order_id": "O9999"}, "order_not_found"),
    ({"customer_id": "C002"}, "order_customer_mismatch"),
    ({"description": "   "}, "invalid_ticket_request"),
    ({"description": "x" * 5001}, "invalid_ticket_request"),
    ({"request_id": "  "}, "invalid_ticket_request"),
    ({"order_id": "  "}, "invalid_ticket_request"),
])
def test_invalid_details_do_not_create_tickets(store, tickets, details, changes, code):
    result = create_ticket(store, tickets, **{**details, **changes})
    assert result["ok"] is False
    assert result["error"] == code
    assert count_tickets(tickets) == 0


def test_identifier_case_and_whitespace_are_normalized(store, tickets, details):
    first = create_ticket(store, tickets, **{
        **details, "customer_id": " c001 ", "order_id": " o1001 ",
        "description": " The item arrived damaged. ",
    })
    assert first["ok"] is True
    assert first == create_ticket(store, tickets, **details)


def test_complaints_can_be_unrelated_to_an_order(store, tickets, details):
    result = create_ticket(store, tickets, **{**details, "order_id": None})
    assert result["ok"] is True
    assert result["data"]["order_id"] is None


@pytest.mark.parametrize("failure", [sqlite3.OperationalError("locked"), OSError("disk full")])
def test_storage_failure_never_reports_success(store, details, failure):
    tickets = Mock(spec=TicketStore)
    tickets.create.side_effect = failure
    result = create_ticket(store, tickets, **details)
    assert result["ok"] is False
    assert result["error"] == "ticket_storage_failed"
    assert "data" not in result


def test_unknown_ticket_returns_none(tickets):
    assert tickets.get("T-UNKNOWN") is None


def test_quotes_in_description_are_saved_as_data(store, tickets, details):
    description = "It says 'damaged'; please investigate."
    result = create_ticket(store, tickets, **{**details, "description": description})
    assert tickets.get(result["data"]["ticket_id"]).description == description
    assert count_tickets(tickets) == 1
