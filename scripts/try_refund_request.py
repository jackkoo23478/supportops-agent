"""Create a persistent demo refund request; never execute a payment."""

import argparse
import json
from datetime import date
from pathlib import Path

from supportops.refund_store import RefundContextStore, RefundStore
from supportops.store import Store, default_data_dir
from supportops.tools.refunds import request_refund
from supportops.tracing import add_trace_arguments, command_trace


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path)
    parser.add_argument("--request-id", default="demo-refund-001")
    parser.add_argument("--customer-id", default="C001")
    parser.add_argument("--order-id", default="O1001")
    parser.add_argument("--amount", default="30.00")
    parser.add_argument("--requested-on", type=date.fromisoformat, default=date.today())
    add_trace_arguments(parser)
    args = parser.parse_args()
    with command_trace(args, "command.request_refund", inputs={
        "request_id": args.request_id, "customer_id": args.customer_id,
        "order_id": args.order_id, "amount": args.amount, "requested_on": args.requested_on,
    }) as active:
        result = request_refund(
            Store(), RefundStore(args.db),
            RefundContextStore(default_data_dir() / "refund_contexts.json"),
            request_id=args.request_id, customer_id=args.customer_id, order_id=args.order_id,
            amount=args.amount, requested_on=args.requested_on,
        )
        active.record(result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
