"""Review a pending refund from a trusted local administrator terminal; never pay."""

import argparse
import json
import sqlite3
from pathlib import Path

from pydantic import ValidationError

from supportops.refund_store import (
    RefundNotFoundError, RefundReviewConflictError, RefundStore,
)
from supportops.tools.common import error, ok
from supportops.tracing import add_trace_arguments, command_trace


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path)
    parser.add_argument("--refund-id", required=True)
    parser.add_argument("--decision", choices=["approve", "reject"], required=True)
    parser.add_argument("--reviewer", required=True, help="Demo audit label; not authentication.")
    parser.add_argument("--reason", required=True)
    add_trace_arguments(parser)
    args = parser.parse_args(argv)
    with command_trace(args, "command.review_refund", inputs={
        "refund_id": args.refund_id, "decision": args.decision,
        "reviewer": args.reviewer, "reason": args.reason,
    }) as active:
        try:
            record = RefundStore(args.db).review(
                args.refund_id.strip(), action=args.decision,
                reviewer=args.reviewer, reason=args.reason,
            )
            result = ok(record.model_dump(mode="json"))
        except RefundNotFoundError as exc:
            result = error("refund_not_found", str(exc))
        except RefundReviewConflictError as exc:
            result = error("review_conflict", str(exc))
        except ValidationError:
            result = error("invalid_review", "Provide a nonblank reviewer and review reason.")
        except (sqlite3.Error, OSError):
            result = error("refund_storage_failed", "The review could not be saved. Retry it.")
        active.record(result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
