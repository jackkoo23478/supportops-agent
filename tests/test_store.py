import json

import pytest
from pydantic import ValidationError

from supportops.schemas import OrderStatus
from supportops.store import Store


def test_get_order_returns_typed_order(store):
    order = store.get_order("O1001")
    assert order is not None
    assert order.customer_id == "C001"
    assert order.status == OrderStatus.DELIVERED
    assert order.total == 59.90


def test_get_order_unknown_returns_none(store):
    assert store.get_order("O9999") is None


def test_get_customer(store):
    customer = store.get_customer("C002")
    assert customer is not None
    assert customer.name == "Ben Lau"
    assert customer.tier == "vip"


def test_get_customer_unknown_returns_none(store):
    assert store.get_customer("C999") is None


def test_orders_for_customer(store):
    ids = sorted(o.order_id for o in store.orders_for_customer("C001"))
    assert ids == ["O1001", "O1002"]


def test_orders_for_unknown_customer_is_empty(store):
    assert store.orders_for_customer("C999") == []


def test_every_sample_order_belongs_to_a_known_customer(store):
    for customer_id in ("C001", "C002", "C003"):
        assert store.orders_for_customer(customer_id)


def test_invalid_data_fails_at_load_time(tmp_path):
    bad_order = {
        "order_id": "O1",
        "customer_id": "C1",
        "status": "delivered",
        "items": [{"sku": "X", "name": "Thing", "quantity": -1, "unit_price": 1.0}],
        "total": 1.0,
        "order_date": "2026-10-01",
    }
    (tmp_path / "orders.json").write_text(json.dumps([bad_order]), encoding="utf-8")
    (tmp_path / "customers.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValidationError):
        Store(tmp_path)


def test_data_dir_can_be_overridden_by_env(tmp_path, monkeypatch):
    (tmp_path / "orders.json").write_text("[]", encoding="utf-8")
    (tmp_path / "customers.json").write_text("[]", encoding="utf-8")
    monkeypatch.setenv("SUPPORTOPS_DATA_DIR", str(tmp_path))
    assert Store().get_order("O1001") is None
