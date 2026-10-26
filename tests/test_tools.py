from supportops.tools.customers import get_customer
from supportops.tools.orders import get_order


def test_get_order_found(store):
    result = get_order(store, "O1005")
    assert result["ok"] is True
    assert result["data"]["total"] == 320.0
    assert result["data"]["status"] == "delivered"
    # dates are plain strings so the result can be sent to the model as JSON
    assert result["data"]["order_date"] == "2026-10-10"


def test_get_order_tolerates_case_and_whitespace(store):
    result = get_order(store, "  o1001 ")
    assert result["ok"] is True
    assert result["data"]["order_id"] == "O1001"


def test_get_order_not_found_returns_error_not_exception(store):
    result = get_order(store, "O9999")
    assert result == {
        "ok": False,
        "error": "order_not_found",
        "message": "No order with id O9999",
    }


def test_get_customer_found_includes_order_ids(store):
    result = get_customer(store, "C001")
    assert result["ok"] is True
    assert result["data"]["name"] == "Amy Chan"
    assert sorted(result["data"]["order_ids"]) == ["O1001", "O1002"]


def test_get_customer_tolerates_case(store):
    assert get_customer(store, "c003")["ok"] is True


def test_get_customer_not_found(store):
    result = get_customer(store, "C999")
    assert result["ok"] is False
    assert result["error"] == "customer_not_found"
