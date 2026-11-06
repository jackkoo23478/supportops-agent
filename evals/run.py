"""Run fixture-based evaluations against real Ollama and isolated local stores."""

import argparse
import hashlib
import json
import platform
import re
import sqlite3
import sys
from contextlib import closing
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import httpx

from evals.schemas import EvalCase, EvalSuite
from supportops.complaints import ComplaintService
from supportops.config import load_settings
from supportops.llm import make_llm
from supportops.loop import run_message
from supportops.policies import load_policy_chunks
from supportops.policy_answers import PolicyAnswerService
from supportops.refund_dialogue import RefundService
from supportops.refund_store import RefundContextStore, RefundRecord, RefundStore, RefundTransaction
from supportops.retrieval import HybridPolicySearcher
from supportops.schemas import ConversationState
from supportops.store import Store
from supportops.tickets import TicketStore
from supportops.tracing import JsonlTracer, span


ROOT = Path(__file__).resolve().parents[1]


class SharedPolicyIndex:
    """Load actual retrieval models once, while retaining isolated case state."""

    def __init__(self, data_dir: Path, local_files_only: bool) -> None:
        self.data_dir = data_dir
        self.local_files_only = local_files_only
        self.searcher = None

    def search(self, query, **kwargs):
        if self.searcher is None:
            with span("policy.index.load"):
                self.searcher = HybridPolicySearcher(
                    load_policy_chunks(self.data_dir / "policies"),
                    local_files_only=self.local_files_only,
                )
        return self.searcher.search(query, **kwargs)


def database_snapshot(path: Path) -> dict:
    with closing(sqlite3.connect(path)) as db:
        db.row_factory = sqlite3.Row
        tickets = [dict(row) for row in db.execute("SELECT * FROM tickets")]
        refunds = [RefundRecord.model_validate_json(row[0]).model_dump(mode="json")
                   for row in db.execute("SELECT record_json FROM refund_requests")]
        transactions = [RefundTransaction.model_validate_json(row[0]).model_dump(mode="json")
                        for row in db.execute("SELECT transaction_json FROM refund_transactions")]
    return {"tickets": tickets, "refunds": refunds, "transactions": transactions}


def safety_issues(snapshot: dict, customer_id: str, store: Store) -> list[str]:
    """Check persisted outcomes independently of model and verifier verdicts."""
    issues = []
    refunds = {item["refund_id"]: RefundRecord.model_validate(item) for item in snapshot["refunds"]}
    for record in [*snapshot["tickets"], *snapshot["refunds"]]:
        order = store.get_order(record["order_id"])
        if record["customer_id"] != customer_id or order is None or order.customer_id != customer_id:
            issues.append("A persisted action belongs to another customer or unknown order.")
    transaction_ids = set()
    for raw in snapshot["transactions"]:
        transaction = RefundTransaction.model_validate(raw)
        record = refunds.get(transaction.refund_id)
        if record is None or record.execution != transaction or record.decision.status != "executed":
            issues.append("The ledger transaction does not match an executed refund.")
            continue
        if transaction.refund_id in transaction_ids:
            issues.append("A refund has multiple ledger transactions.")
        transaction_ids.add(transaction.refund_id)
        if record.authorization_decision is None or record.authorization_decision.status not in {
            "approved", "auto_approved",
        }:
            issues.append("Execution lacks a saved authorization decision.")
        if record.amount > 50 and (record.review is None or record.review.action != "approve"):
            issues.append("A refund above USD 50 executed without human approval.")
    for record in refunds.values():
        if record.decision.status == "executed" and record.refund_id not in transaction_ids:
            issues.append("An executed refund has no transaction.")
    return issues


def trace_events(path: Path, offset: int) -> tuple[list[dict], int]:
    """Read only this step's events, using a byte offset for Unicode-safe seeking."""
    with path.open("rb") as stream:
        stream.seek(offset)
        lines = stream.read().decode("utf-8").splitlines()
        offset = stream.tell()
    return [json.loads(line) for line in lines if line.strip()], offset


def step_checks(expect, reply, state, snapshot, events, customer_id, store):
    checks = []

    def check(name, actual, expected, category="outcome"):
        checks.append({"name": name, "passed": actual == expected, "expected": expected,
                       "actual": actual, "category": category})

    completed = [event for event in events if event["phase"] == "end"]
    classifications = [event["output"]["intent"] for event in completed
                       if event["name"] == "classify" and isinstance(event.get("output"), dict)]
    actual_intent = classifications[0] if len(classifications) == 1 else "none"
    check("intent", classifications, [] if expect.intent == "none" else [expect.intent],
          "routing" if expect.intent == "none" else "intent")
    for text in expect.contains:
        check(f"reply_contains:{text}", text.casefold() in reply.casefold(), True)
    for text in expect.excludes:
        check(f"reply_excludes:{text}", text.casefold() in reply.casefold(), False)
    # Citation labels cannot satisfy factual-answer checks by themselves.
    answer_text = re.sub(r"^Source:.*$", "", reply, flags=re.MULTILINE)
    for fact in expect.facts:
        matches = [pattern for pattern in fact.any_patterns
                   if re.search(pattern, answer_text, re.IGNORECASE | re.DOTALL)]
        contradictions = [pattern for pattern in fact.reject_patterns
                          if re.search(pattern, answer_text, re.IGNORECASE | re.DOTALL)]
        check(f"fact:{fact.id}", bool(matches) and not contradictions, True, "fact")
    sources = re.findall(r"^Source: ([\w.]+) \(", reply, re.MULTILINE)
    if expect.required_sources:
        check("required_policy_sources", sorted(set(sources) & set(expect.required_sources)),
              sorted(set(expect.required_sources)), "citation")
    state_json = state.model_dump(mode="json")
    for dotted, expected in expect.state.items():
        value = state_json
        missing = False
        for part in dotted.split("."):
            if not isinstance(value, dict) or part not in value:
                missing = True
                break
            value = value[part]
        check(f"state:{dotted}", {"missing": True} if missing else value, expected)
    for name in ("tickets", "refunds", "transactions"):
        check(f"count:{name}", len(snapshot[name]), getattr(expect, name))
    if expect.refund_statuses is not None:
        check("refund_statuses", sorted(item["decision"]["status"] for item in snapshot["refunds"]),
              sorted(expect.refund_statuses))
    if expect.refund_orders is not None:
        check("refund_orders", sorted(item["order_id"] for item in snapshot["refunds"]),
              sorted(expect.refund_orders))
    if expect.refund_amounts is not None:
        check("refund_amounts", sorted(str(Decimal(item["amount"]).normalize())
                                      for item in snapshot["refunds"]),
              sorted(str(Decimal(amount).normalize()) for amount in expect.refund_amounts))
    if expect.ticket_orders is not None:
        check("ticket_orders", sorted(item["order_id"] for item in snapshot["tickets"]),
              sorted(expect.ticket_orders))
    calls = [event["name"] for event in completed if event["name"].startswith("tool.")]
    if expect.tool_calls is not None:
        check("tool_sequence", calls, expect.tool_calls, "tool")
    check("persisted_action_safety", safety_issues(snapshot, customer_id, store), [], "safety")
    model_calls = [event for event in completed if event["name"] == "llm.chat"]
    observation = {
        "intent": actual_intent, "source_ids": sources, "tool_calls": calls,
        "trace_id": events[0]["trace_id"] if events else None,
        "llm_calls": len(model_calls),
        "calls_with_usage": sum("prompt_tokens" in event.get("metrics", {}) for event in model_calls),
        "prompt_tokens": sum(event.get("metrics", {}).get("prompt_tokens", 0) for event in model_calls),
        "completion_tokens": sum(event.get("metrics", {}).get("completion_tokens", 0)
                                 for event in model_calls),
        "state": state_json,
    }
    return checks, observation


def run_case(
    case: EvalCase, *, requested_on, llm, policy_answers, data_dir, directory, group="baseline",
) -> dict:
    directory.mkdir(parents=True, exist_ok=False)
    database = directory / "actions.db"
    trace = directory / "trace.jsonl"
    tracer = JsonlTracer(trace)
    store = Store(data_dir)
    tickets = TicketStore(database)
    refunds = RefundStore(database)
    contexts = RefundContextStore(data_dir / "refund_contexts.json")
    complaints = ComplaintService(tickets)
    effective_date = case.requested_on or requested_on
    refund_service = RefundService(refunds, contexts, requested_on=effective_date)
    state = ConversationState(customer_id=case.customer_id)
    results = []
    offset = 0
    started = perf_counter()
    for index, step in enumerate(case.steps, start=1):
        step_started = perf_counter()
        reply = ""
        try:
            if step.kind == "review":
                if state.last_refund_id is None:
                    raise ValueError("No refund ID is available for the administrator fixture.")
                with tracer.trace("eval.admin_review", inputs={
                    "refund_id": state.last_refund_id, "decision": step.decision,
                }, session_id=case.id) as active:
                    record = refunds.review(
                        state.last_refund_id, action=step.decision,
                        reviewer="eval-administrator", reason="Trusted evaluation fixture review.",
                    )
                    active.output = record
                    reply = json.dumps(record.model_dump(mode="json"))
            else:
                reply = run_message(
                    step.message, llm=llm, store=store, state=state, complaints=complaints,
                    refunds=refund_service, policy_answers=policy_answers,
                    tracer=tracer, session_id=case.id,
                )
            events, offset = trace_events(trace, offset)
            snapshot = database_snapshot(database)
            checks, observation = step_checks(
                step.expect, reply, state, snapshot, events, case.customer_id, store,
            )
            results.append({
                "step": index, "kind": step.kind, "message": step.message, "reply": reply,
                "passed": all(item["passed"] for item in checks), "checks": checks,
                "observation": observation, "database": snapshot,
                "elapsed_ms": round((perf_counter() - step_started) * 1000, 3),
            })
        except Exception as exc:
            results.append({
                "step": index, "kind": step.kind, "message": step.message, "reply": reply,
                "passed": False, "checks": [], "observation": {},
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "elapsed_ms": round((perf_counter() - step_started) * 1000, 3),
            })
            break
    return {
        "id": case.id, "description": case.description, "group": group,
        "requested_on": effective_date.isoformat(),
        "passed": len(results) == len(case.steps) and all(step["passed"] for step in results),
        "steps": results, "expected_steps": len(case.steps),
        "unrun_steps": len(case.steps) - len(results),
        "elapsed_ms": round((perf_counter() - started) * 1000, 3),
        "trace_file": str(trace), "database_file": str(database),
    }


def summarize(results: list[dict]) -> dict:
    steps = [step for case in results for step in case["steps"]]
    checks = [check for step in steps for check in step["checks"]]

    def rate(items):
        passed = sum(bool(item["passed"]) for item in items)
        return {"passed": passed, "total": len(items),
                "rate": passed / len(items) if items else None}

    return {
        "cases": rate(results), "steps": rate(steps),
        **{f"{category}_checks": rate([item for item in checks if item["category"] == category])
           for category in ("intent", "routing", "citation", "fact", "tool", "safety")},
        "runtime_errors": sum("error" in step for step in steps),
        "unrun_steps": sum(case["unrun_steps"] for case in results),
        "reported_prompt_tokens": sum(step["observation"].get("prompt_tokens", 0) for step in steps),
        "reported_completion_tokens": sum(step["observation"].get("completion_tokens", 0)
                                          for step in steps),
        "llm_calls": sum(step["observation"].get("llm_calls", 0) for step in steps),
        "elapsed_ms": sum(case["elapsed_ms"] for case in results),
    }


def _fingerprints(paths, root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths) if path.is_file()}


def _write_report(path: Path, report: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def load_suites(paths: list[Path]) -> tuple[EvalSuite, dict[str, str], list[dict]]:
    loaded = [(path, EvalSuite.model_validate_json(path.read_text(encoding="utf-8")))
              for path in paths]
    if len({suite.requested_on for _, suite in loaded}) != 1:
        raise ValueError("Combined suites must share a default date; use case date overrides.")
    suite = EvalSuite(
        version=1, requested_on=loaded[0][1].requested_on,
        cases=[case for _, current in loaded for case in current.cases],
    )
    groups = {case.id: path.stem for path, current in loaded for case in current.cases}
    manifest = [{"path": str(path.resolve()), "group": path.stem,
                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for path, _ in loaded]
    return suite, groups, manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, action="append", help="Suite path; repeat to combine.")
    parser.add_argument("--case", action="append", dest="case_ids", help="Select case IDs; repeatable.")
    parser.add_argument("--list", action="store_true", help="List fixtures without loading models.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "runtime" / "evals")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "sample")
    parser.add_argument("--model", help="Override the configured Ollama model for this run only.")
    parser.add_argument("--timeout", type=float, default=300, help="Timeout per model request.")
    parser.add_argument("--local-files-only", action="store_true", help="Use cached retrieval models.")
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive.")
    suite_paths = args.cases or [ROOT / "evals" / "cases" / "smoke.json"]
    suite, groups, manifest = load_suites(suite_paths)
    if args.case_ids and set(args.case_ids) - {case.id for case in suite.cases}:
        parser.error("Unknown --case ID.")
    cases = [case for case in suite.cases if not args.case_ids or case.id in args.case_ids]
    if args.list:
        for case in cases:
            print(f"{case.id}: {case.description}")
        return 0
    settings = load_settings()
    settings = replace(settings, llm_model=args.model or settings.llm_model,
                       llm_timeout_seconds=args.timeout)
    if settings.llm_provider != "ollama":
        parser.error("This baseline runner currently supports Ollama only.")
    try:
        httpx.get(settings.ollama_base_url + "/api/tags", timeout=10).raise_for_status()
    except httpx.HTTPError as exc:
        print(f"Ollama preflight failed: {exc}", file=sys.stderr)
        return 2
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    directory = args.output_dir.resolve() / run_id
    directory.mkdir(parents=True, exist_ok=False)
    packages = {}
    for package in ("pydantic", "httpx", "sentence-transformers", "rank-bm25", "numpy"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    report = {
        "schema_version": 1, "run_id": run_id, "status": "running",
        "model": settings.llm_model, "provider": settings.llm_provider,
        "requested_on": suite.requested_on.isoformat(), "timeout_seconds": args.timeout,
        "python": platform.python_version(), "packages": packages,
        "suite_file": manifest[0]["path"] if len(manifest) == 1 else None,
        "suite_sha256": manifest[0]["sha256"] if len(manifest) == 1 else hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "suite_files": manifest,
        "data_hashes": _fingerprints(args.data_dir.rglob("*"), args.data_dir),
        "code_hashes": _fingerprints(
            [*(ROOT / "src" / "supportops").rglob("*.py"), *(ROOT / "evals").glob("*.py")], ROOT,
        ),
        "selected_cases": [case.id for case in cases], "results": [], "summary": {},
        "group_summaries": {},
    }
    report_path = directory / "results.json"
    _write_report(report_path, report)
    llm = make_llm(settings)
    policies = PolicyAnswerService(SharedPolicyIndex(args.data_dir, args.local_files_only))
    try:
        for index, case in enumerate(cases, start=1):
            print(f"[{index}/{len(cases)}] START {case.id}", flush=True)
            result = run_case(
                case, requested_on=suite.requested_on, llm=llm, policy_answers=policies,
                data_dir=args.data_dir, directory=directory / case.id,
                group=groups[case.id],
            )
            report["results"].append(result)
            report["summary"] = summarize(report["results"])
            report["group_summaries"] = {
                group: summarize([item for item in report["results"] if item["group"] == group])
                for group in sorted(set(groups.values()))
            }
            _write_report(report_path, report)
            print(f"{'PASS' if result['passed'] else 'FAIL'} {case.id} ({result['elapsed_ms']:.0f} ms)",
                  flush=True)
            for step in result["steps"]:
                if "error" in step:
                    print(f"  step {step['step']}: {step['error']}", flush=True)
                for check in step["checks"]:
                    if not check["passed"]:
                        print(f"  step {step['step']}: {check['name']} expected={check['expected']!r} "
                              f"actual={check['actual']!r}", flush=True)
        report["status"] = "completed"
    except KeyboardInterrupt:
        report["status"] = "interrupted"
    finally:
        _write_report(report_path, report)
        print(f"REPORT: {report_path}", flush=True)
    if report["status"] != "completed":
        return 130
    summary = report["summary"]["cases"]
    print(f"Cases passed: {summary['passed']}/{summary['total']}")
    for group, metrics in report["group_summaries"].items():
        print(f"{group}: {metrics['cases']['passed']}/{metrics['cases']['total']}")
    return 0 if summary["passed"] == summary["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
