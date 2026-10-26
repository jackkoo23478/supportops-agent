"""Order lookup tool."""

from supportops.store import Store
from supportops.tools.common import error, ok


def get_order(store: Store, order_id: str) -> dict:
    """Look up one order by its id (for example O1001)."""
    order = store.get_order(order_id.strip().upper())
    if order is None:
        return error("order_not_found", f"No order with id {order_id.strip()}")
    return ok(order.model_dump(mode="json"))