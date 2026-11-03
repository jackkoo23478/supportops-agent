"""Command line entry point with single-message and interactive modes."""

import argparse
import sys
from collections.abc import Sequence

from supportops.config import load_settings
from supportops.llm import LLMError, OllamaClient, make_llm
from supportops.loop import run_message
from supportops.store import Store


def run_interactive(
    *,
    llm: OllamaClient,
    store: Store,
) -> int:
    """Read messages until the user exits."""
    print("SupportOps Agent")
    print("Type 'exit' or 'quit' to leave.")

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
                )
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
        description="SupportOps Agent: query a sample order."
    )
    parser.add_argument(
        "message",
        nargs="?",
        help="Handle one message, or omit it to start interactive mode.",
    )
    args = parser.parse_args(argv)

    if args.message is not None and not args.message.strip():
        print("Please enter a message.", file=sys.stderr)
        return 1

    try:
        settings = load_settings()
        store = Store()
        llm = make_llm(settings)

        if args.message is None:
            return run_interactive(
                llm=llm,
                store=store,
            )

        reply = run_message(
            args.message,
            llm=llm,
            store=store,
        )
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