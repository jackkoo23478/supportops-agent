"""Record an approved refund in the local simulation ledger; never send money."""

import argparse
import json
import sqlite3
from pathlib import Path

from pydantic import ValidationError

from supportops.refund_store import RefundContextStore, RefundStore
from supportops.store import Store, default_data_dir
from supportops.tools.common import error
from supportops.tools.refunds import execute_refund
from supportops.tracing import add_trace_arguments, command_trace


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path)
    parser.add_argument("--refund-id", required=True)
    parser.add_argument("--customer-id", required=True, help="Demo identity, not authentication.")
    add_trace_arguments(parser)
    args = parser.parse_args(argv)
    with command_trace(args, "command.execute_refund", inputs={
        "refund_id": args.refund_id, "customer_id": args.customer_id,
    }) as active:
        try:
            result = execute_refund(
                Store(), RefundStore(args.db),
                RefundContextStore(default_data_dir() / "refund_contexts.json"),
                refund_id=args.refund_id, customer_id=args.customer_id,
            )
        except (sqlite3.Error, OSError, ValueError, ValidationError):
            result = error("refund_setup_failed", "The database or verified refund data is unavailable.")
        active.record(result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
