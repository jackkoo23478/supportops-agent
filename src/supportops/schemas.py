"""Data models shared by tools, the agent loop and the evals."""

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class OrderStatus(str, Enum):
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


class OrderItem(BaseModel):
    sku: str
    name: str
    quantity: int = Field(ge=1)
    unit_price: float = Field(ge=0)


class Order(BaseModel):
    order_id: str
    customer_id: str
    status: OrderStatus
    items: list[OrderItem]
    total: float = Field(ge=0)
    order_date: date
    delivered_date: date | None = None


class Customer(BaseModel):
    customer_id: str
    name: str
    email: str
    tier: Literal["standard", "vip"] = "standard"
    joined: date

class IntentResult(BaseModel):
    intent: Literal["order_status", "policy_question", "refund", "complaint", "other"]
    order_id: str | None = None
    use_previous_order: bool = False


class PendingComplaint(BaseModel):
    """One complaint action; keep its ID across retries."""

    request_id: str = Field(default_factory=lambda: uuid4().hex)
    description: str = Field(min_length=1, max_length=5000)
    order_id: str | None = None


class PendingRefund(BaseModel):
    """An in-memory draft; lock details once submission is attempted."""

    request_id: str = Field(default_factory=lambda: uuid4().hex)
    order_id: str | None = None
    amount: Decimal | None = None
    confirmed: bool = False


class ConversationState(BaseModel):
    customer_id: str = "C001"
    last_order_id: str | None = None
    pending_complaint: PendingComplaint | None = None
    last_ticket_id: str | None = None
    pending_refund: PendingRefund | None = None
    last_refund_id: str | None = None
    last_action: Literal["complaint", "refund"] | None = None


class PolicyChunk(BaseModel):
    source_id: str = Field(min_length=1)
    document: str = Field(min_length=1)
    content: str = Field(min_length=1)


class PolicyAnswer(BaseModel):
    """Structured policy reply; cited IDs must also be checked against retrieval."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    answer: str = Field(min_length=1)
    source_ids: list[str]
    insufficient_evidence: bool


class TicketRequest(BaseModel):
    """Validated input for one idempotent ticket-creation action."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    request_id: str = Field(min_length=1)
    customer_id: str = Field(min_length=1)
    order_id: str | None = Field(default=None, min_length=1)
    description: str = Field(min_length=1, max_length=5000)


class Ticket(TicketRequest):
    ticket_id: str
    status: Literal["open", "closed"] = "open"
    created_at: datetime
