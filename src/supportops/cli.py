"""Command line entry point with single-message and interactive modes."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

from supportops.complaints import ComplaintService
from supportops.config import load_settings
from supportops.llm import LLMError, OllamaClient, make_llm
from supportops.loop import run_message
from supportops.policy_answers import PolicyAnswerService, PolicySearchError
from supportops.refund_dialogue import RefundService
from supportops.schemas import ConversationState
from supportops.store import Store
from supportops.tracing import JsonlTracer, add_trace_arguments


def run_interactive(
    *,
    llm: OllamaClient,
    store: Store,
    policy_answers: PolicyAnswerService | None = None,
    complaints: ComplaintService | None = None,
    refunds: RefundService | None = None,
    customer_id: str = "C001",
    tracer: JsonlTracer | None = None,
) -> int:
    """Read messages while preserving order context."""
    state = ConversationState(customer_id=customer_id.strip().upper())
    session_id = uuid4().hex
    if complaints is None:
        complaints = ComplaintService()
    if policy_answers is None:
        policy_answers = PolicyAnswerService()
    if refunds is None:
        refunds = RefundService()

    print("SupportOps Agent")
    print(f"Demo customer: {state.customer_id}")
    print("Type 'exit' or 'quit' to leave.")
    print("Use 'cancel complaint' to discard a pending complaint, or 'retry' after a save error.")
    print("Refunds: 'confirm refund', 'cancel refund', 'refund status', or 'execute refund R-...'.")

    try:
        while True:
            message = input("You: ").strip()

            if message.lower() in {"exit", "quit"}:
                print("Goodbye.")
                return 0

            if not message:
                continue

            try:
                reply = run_message(
                    message,
                    llm=llm,
                    store=store,
                    state=state,
                    policy_answers=policy_answers,
                    complaints=complaints,
                    refunds=refunds,
                    tracer=tracer,
                    session_id=session_id,
                )
            except PolicySearchError as exc:
                print(str(exc), file=sys.stderr)
                continue
            except LLMError as exc:
                print(
                    f"The model could not complete the request: {exc}",
                    file=sys.stderr,
                )
                continue

            print(f"Agent: {reply}")

    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye.")
        return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SupportOps Agent: policies, orders, complaints and simulated refunds."
    )
    parser.add_argument(
        "message",
        nargs="?",
        help="Handle one message, or omit it to start interactive mode.",
    )
    parser.add_argument(
        "--customer-id", default="C001", help="Select a demo customer; this is not authentication."
    )
    parser.add_argument("--ticket-db", type=Path, help="Override the runtime ticket database path.")
    parser.add_argument("--refund-db", type=Path, help="Override the runtime refund database path.")
    add_trace_arguments(parser)
    args = parser.parse_args(argv)
    if not args.customer_id.strip():
        parser.error("customer-id must not be blank.")

    if args.message is not None and not args.message.strip():
        print("Please enter a message.", file=sys.stderr)
        return 1

    try:
        try:
            settings = load_settings()
        except ValueError as exc:
            print(f"Invalid configuration: {exc}", file=sys.stderr)
            return 1
        store = Store()
        llm = make_llm(settings)
        policy_answers = PolicyAnswerService()
        complaints = ComplaintService(db_path=args.ticket_db)
        refunds = RefundService(db_path=args.refund_db)
        tracer = None if args.no_trace else JsonlTracer(args.trace_file)

        if args.message is None:
            return run_interactive(
                llm=llm,
                store=store,
                policy_answers=policy_answers,
                complaints=complaints,
                refunds=refunds,
                customer_id=args.customer_id,
                tracer=tracer,
            )

        reply = run_message(
            args.message,
            llm=llm,
            store=store,
            policy_answers=policy_answers,
            complaints=complaints,
            refunds=refunds,
            state=ConversationState(customer_id=args.customer_id.strip().upper()),
            tracer=tracer,
        )
    except PolicySearchError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except LLMError as exc:
        print(
            f"The model could not complete the request: {exc}",
            file=sys.stderr,
        )
        return 1

    print(reply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
