"""In-memory data store loaded from JSON files. Stands in for a real database."""

import json
import os
from pathlib import Path

from supportops.schemas import Customer, Order


def default_data_dir() -> Path:
    override = os.environ.get("SUPPORTOPS_DATA_DIR")
    if override:
        return Path(override)
    # store.py -> supportops -> src -> repo root
    return Path(__file__).resolve().parents[2] / "data" / "sample"


def _load_json(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


class Store:
    def __init__(self, data_dir: Path | None = None) -> None:
        data_dir = data_dir or default_data_dir()
        self._orders = {
            o["order_id"]: Order(**o) for o in _load_json(data_dir / "orders.json")
        }
        self._customers = {
            c["customer_id"]: Customer(**c) for c in _load_json(data_dir / "customers.json")
        }

    def get_order(self, order_id: str) -> Order | None:
        return self._orders.get(order_id)

    def get_customer(self, customer_id: str) -> Customer | None:
        return self._customers.get(customer_id)

    def orders_for_customer(self, customer_id: str) -> list[Order]:
        return [o for o in self._orders.values() if o.customer_id == customer_id]