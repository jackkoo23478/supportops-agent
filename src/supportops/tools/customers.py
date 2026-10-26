"""Customer lookup tool."""

from supportops.store import Store
from supportops.tools.common import error, ok


def get_customer(store: Store, customer_id: str) -> dict:
    """Look up one customer by id (for example C001), including their order ids."""
    customer = store.get_customer(customer_id.strip().upper())
    if customer is None:
        return error("customer_not_found", f"No customer with id {customer_id.strip()}")
    data = customer.model_dump(mode="json")
    data["order_ids"] = [o.order_id for o in store.orders_for_customer(customer.customer_id)]
    return ok(data)