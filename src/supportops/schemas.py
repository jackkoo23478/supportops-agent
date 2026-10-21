"""Data models shared by tools, the agent loop and the evals."""

from datetime import date
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


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
