"""Inspect saved JSONL traces without rerunning tools or model calls."""

import argparse
import json
import sys
from pathlib import Path


def read_events(path: Path) -> list[dict]:
    events = []
    invalid = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
            if not isinstance(event, dict) or not all(
                key in event for key in ("trace_id", "span_id", "name", "phase")
            ):
                raise ValueError("Invalid event")
            events.append(event)
        except (ValueError, TypeError):
            invalid += 1
    if invalid:
        print(f"Skipped {invalid} invalid or incomplete trace lines.", file=sys.stderr)
    return events


def summarize_trace(events: list[dict]) -> dict:
    completed = [event for event in events if event["phase"] == "end"]
    root = next((event for event in completed if event.get("parent_span_id") is None), None)
    calls = [event for event in completed if event["name"] == "llm.chat"]
    return {
        "trace_id": events[0]["trace_id"],
        "status": root["status"] if root else "incomplete",
        "elapsed_ms": root.get("elapsed_ms") if root else None,
        "llm_calls": len(calls),
        "calls_with_usage": sum("prompt_tokens" in event.get("metrics", {}) for event in calls),
        "prompt_tokens": sum(event.get("metrics", {}).get("prompt_tokens", 0) for event in calls),
        "completion_tokens": sum(
            event.get("metrics", {}).get("completion_tokens", 0) for event in calls
        ),
        "error_spans": sum(event.get("status") == "error" for event in completed),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path", nargs="?", type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "runtime" / "traces.jsonl",
    )
    parser.add_argument("--trace-id")
    parser.add_argument("--session-id")
    parser.add_argument("--refund-id", help="Find traces mentioning this refund ID.")
    parser.add_argument("--last", type=int, default=1)
    parser.add_argument("--details", action="store_true", help="Also show saved inputs/outputs.")
    args = parser.parse_args(argv)
    if args.last < 1:
        parser.error("--last must be positive.")
    try:
        events = read_events(args.path)
    except OSError as exc:
        print(f"Could not read trace file: {exc}", file=sys.stderr)
        return 1
    grouped: dict[str, list[dict]] = {}
    for event in events:
        if args.session_id and event.get("session_id") != args.session_id:
            continue
        grouped.setdefault(event["trace_id"], []).append(event)
    selected = [
        trace for trace_id, trace in grouped.items()
        if (not args.trace_id or trace_id == args.trace_id)
        and (not args.refund_id or args.refund_id in json.dumps(trace))
    ]
    if not args.trace_id:
        selected = selected[-args.last:]
    if not selected:
        print("No matching traces found.")
        return 1
    for trace in selected:
        summary = summarize_trace(trace)
        print(
            f"Trace {summary['trace_id']} | {summary['status']} | "
            f"{summary['elapsed_ms']} ms | LLM calls: {summary['llm_calls']} | "
            f"reported tokens in/out: {summary['prompt_tokens']}/{summary['completion_tokens']} | "
            f"usage records: {summary['calls_with_usage']} | "
            f"error spans: {summary['error_spans']}"
        )
        ends = {event["span_id"]: event for event in trace if event["phase"] == "end"}
        starts = [event for event in trace if event["phase"] == "start"]
        by_id = {event["span_id"]: event for event in starts}
        for event in starts:
            depth = 0
            parent = event.get("parent_span_id")
            seen = set()
            while parent in by_id and parent not in seen:
                seen.add(parent)
                depth += 1
                parent = by_id[parent].get("parent_span_id")
            end = ends.get(event["span_id"], {})
            print(
                f"{'  ' * depth}{event['name']} | {end.get('status', 'incomplete')} | "
                f"{end.get('elapsed_ms', '?')} ms"
            )
            if args.details:
                print(json.dumps({
                    "inputs": event.get("inputs"), "output": end.get("output"),
                    "error": end.get("error"), "metrics": end.get("metrics"),
                }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
