"""Create a sample complaint ticket without calling an LLM."""

import argparse
import json
from pathlib import Path

from supportops.store import Store
from supportops.tickets import TicketStore
from supportops.tools.tickets import create_ticket


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, help="Override the runtime SQLite database path.")
    parser.add_argument("--request-id", default="demo-damaged-order-001")
    args = parser.parse_args()
    result = create_ticket(
        Store(), TicketStore(args.db), request_id=args.request_id,
        customer_id="C001", order_id="O1001", description="The item arrived damaged.",
    )
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
