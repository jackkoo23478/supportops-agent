"""Tracing must explain execution without changing outcomes or duplicating writes."""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from decimal import Decimal
from unittest.mock import Mock

import httpx
import numpy as np
import pytest

from scripts.review_refund import main as review_main
from scripts.show_trace import main as show_main, read_events, summarize_trace
from supportops import cli
from supportops.llm import LLMError, OllamaClient
from supportops.loop import run_message
from supportops.policy_answers import PolicyAnswerService
from supportops.refund_dialogue import RefundService
from supportops.refund_store import RefundContextStore, RefundStore
from supportops.retrieval import HybridPolicySearcher
from supportops.schemas import ConversationState, IntentResult, PolicyAnswer, PolicyChunk
from supportops.store import default_data_dir
from supportops.tools.orders import get_order
from supportops.tools.refunds import request_refund
from supportops.tracing import JsonlTracer, span, traced
from supportops.verification import PolicyEvidence, PolicyVerification


def ends(path, name=None):
    return [event for event in read_events(path)
            if event["phase"] == "end" and (name is None or event["name"] == name)]


def test_message_traces_share_session_but_have_distinct_ids_and_state_snapshots(tmp_path, store):
    path = tmp_path / "trace.jsonl"
    tracer = JsonlTracer(path)
    llm = Mock(spec=OllamaClient)
    llm.chat_json.return_value = IntentResult(intent="order_status", order_id="O1001")
    state = ConversationState()
    for _ in range(2):
        run_message(
            "Check O1001", llm=llm, store=store, state=state,
            tracer=tracer, session_id="session-one",
        )
    events = read_events(path)
    assert {event["session_id"] for event in events} == {"session-one"}
    trace_ids = list(dict.fromkeys(event["trace_id"] for event in events))
    assert len(trace_ids) == 2
    for trace_id in trace_ids:
        group = [event for event in events if event["trace_id"] == trace_id]
        assert [event["sequence"] for event in group] == list(range(1, len(group) + 1))
        start_ids = {event["span_id"] for event in group if event["phase"] == "start"}
        assert start_ids == {event["span_id"] for event in group if event["phase"] == "end"}
        assert all(event["parent_span_id"] in start_ids or event["parent_span_id"] is None
                   for event in group)
    first_start = events[0]
    assert first_start["inputs"]["state_before"]["last_order_id"] is None
    assert ends(path, "message")[0]["output"]["state_after"]["last_order_id"] == "O1001"
    assert ends(path, "classify")[0]["output"]["intent"] == "order_status"
    assert ends(path, "tool.get_order")[0]["output"]["data"]["order_id"] == "O1001"


def test_structured_retries_record_every_model_call_and_usage(tmp_path, store):
    path = tmp_path / "usage.jsonl"
    contents = iter(["not json", '{"intent":"order_status","order_id":"O1001"}'])

    def respond(request):
        return httpx.Response(200, json={
            "message": {"content": next(contents)}, "prompt_eval_count": 10,
            "eval_count": 5, "total_duration": 2000000,
        })

    llm = OllamaClient("http://test", "test-model", transport=httpx.MockTransport(respond))
    run_message("Check O1001", store=store, llm=llm, tracer=JsonlTracer(path))
    calls = ends(path, "llm.chat")
    assert len(calls) == 2
    assert all(call["metrics"] == {
        "prompt_tokens": 10, "completion_tokens": 5, "model_duration_ms": 2.0,
    } for call in calls)
    starts = [event for event in read_events(path) if event["name"] == "llm.chat"
              and event["phase"] == "start"]
    assert starts[0]["inputs"]["model"] == "test-model"
    assert ends(path, "llm.validation_retry")[0]["status"] == "error"
    summary = summarize_trace(read_events(path))
    assert summary["prompt_tokens"] == 20
    assert summary["completion_tokens"] == 10
    assert summary["llm_calls"] == 2


def test_exceptions_are_logged_and_context_is_restored(tmp_path, store):
    path = tmp_path / "errors.jsonl"
    tracer = JsonlTracer(path)
    llm = Mock(spec=OllamaClient)
    llm.chat_json.side_effect = LLMError("Model unavailable")
    with pytest.raises(LLMError):
        run_message("Check O1001", store=store, llm=llm, tracer=tracer)
    assert ends(path, "message")[0]["error"]["type"] == "LLMError"
    # A call outside an active trace must not inherit a failed trace's context.
    before = len(read_events(path))
    get_order(store, "O1001")
    assert len(read_events(path)) == before
    with tracer.trace("next"):
        get_order(store, "O9999")
    error = ends(path, "tool.get_order")[-1]
    assert error["status"] == "error"
    assert error["error"]["code"] == "order_not_found"


def test_known_secret_fields_are_redacted_and_inputs_are_snapshotted(tmp_path):
    path = tmp_path / "redaction.jsonl"
    data = {"authorization": "secret-value", "nested": {"api_key": "key-value"},
            "amount": Decimal("30.00"), "items": ["original"]}
    with JsonlTracer(path).trace("test", inputs=data) as active:
        data["items"].append("later")
        active.output = {"access_token": "another-secret"}
    events = read_events(path)
    assert events[0]["inputs"]["items"] == ["original"]
    assert events[0]["inputs"]["amount"] == "30.00"
    raw = path.read_text(encoding="utf-8")
    assert "secret-value" not in raw and "key-value" not in raw
    assert "another-secret" not in raw


def test_failed_trace_writes_never_duplicate_or_interrupt_refund(tmp_path, store, capsys):
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")
    tracer = JsonlTracer(blocked / "trace.jsonl")
    refunds = RefundStore(tmp_path / "refund.db")
    contexts = RefundContextStore(default_data_dir() / "refund_contexts.json")
    service = RefundService(refunds, contexts, requested_on=date(2026, 11, 6))
    state = ConversationState()
    llm = Mock(spec=OllamaClient)
    llm.chat_json.return_value = IntentResult(intent="refund")
    for message in ("Refund O1001 USD 30", "confirm refund", "retry refund"):
        reply = run_message(
            message, store=store, llm=llm, state=state, refunds=service, tracer=tracer,
        )
    assert "Status: executed" in reply
    with sqlite3.connect(refunds.path) as db:
        assert db.execute("SELECT COUNT(*) FROM refund_requests").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM refund_transactions").fetchone()[0] == 1
    assert capsys.readouterr().err.count("Trace output is unavailable") == 1


def test_parallel_traces_do_not_mix_parentage_or_break_jsonl(tmp_path):
    path = tmp_path / "parallel.jsonl"
    tracer = JsonlTracer(path)

    def run(index):
        with tracer.trace("root", session_id=str(index)):
            with span("child") as active:
                active.output = index

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(run, range(8)))
    events = read_events(path)
    assert len(events) == 32
    for session_id in map(str, range(8)):
        group = [event for event in events if event["session_id"] == session_id]
        assert len({event["trace_id"] for event in group}) == 1
        assert len(group) == 4
        assert next(event for event in group if event["phase"] == "end"
                    and event["name"] == "child")["output"] == int(session_id)


def test_policy_trace_records_search_ranking_and_citation_decision(tmp_path, store):
    chunks = [PolicyChunk(source_id=f"returns.{index}", document="returns.md", content=text)
              for index, text in enumerate([
                  "Unused items need original packaging.", "Return within 30 days.",
                  "Proof of purchase is required.",
              ])]
    embedder = Mock()
    embedder.encode_document.return_value = np.eye(3)
    embedder.encode_query.return_value = np.ones(3)
    reranker = Mock()
    reranker.score.return_value = [3, 2, 1]
    searcher = HybridPolicySearcher(chunks, embedder=embedder, reranker=reranker)
    llm = Mock(spec=OllamaClient)
    llm.chat_json.side_effect = [IntentResult(intent="policy_question"), PolicyAnswer(
        answer="Keep the original packaging.", source_ids=["returns.0"], insufficient_evidence=False,
    ), PolicyVerification(supported=True, evidence=[PolicyEvidence(
        sentence="Keep the original packaging.", source_id="returns.0",
        quote="Unused items need original packaging.",
    )])]
    path = tmp_path / "policy.jsonl"
    reply = run_message(
        "Do returns need original packaging?", llm=llm, store=store,
        policy_answers=PolicyAnswerService(searcher), tracer=JsonlTracer(path),
    )
    assert "Source: returns.0" in reply
    names = {event["name"] for event in read_events(path)}
    assert {"retrieval.bm25", "embedding.query", "retrieval.semantic", "retrieval.rrf",
            "retrieval.rerank", "policy.citation_check", "tool.search_policy"} <= names
    assert len(ends(path, "tool.search_policy")[0]["output"]["data"]["matches"]) == 3
    assert ends(path, "policy.citation_check")[0]["output"]["accepted"] is True


def test_review_command_trace_links_to_refund_id(tmp_path, store, capsys):
    refunds = RefundStore(tmp_path / "review.db")
    contexts = RefundContextStore(default_data_dir() / "refund_contexts.json")
    result = request_refund(
        store, refunds, contexts, request_id="review", customer_id="C001", order_id="O1001",
        amount="59.90", requested_on=date(2026, 11, 6),
    )
    refund_id = result["data"]["refund_id"]
    path = tmp_path / "review.jsonl"
    assert review_main([
        "--db", str(refunds.path), "--refund-id", refund_id, "--decision", "approve",
        "--reviewer", "admin", "--reason", "Verified.", "--trace-file", str(path),
    ]) == 0
    assert ends(path, "refund.review")[0]["output"]["decision"]["status"] == "approved"
    assert show_main([str(path), "--refund-id", refund_id]) == 0
    assert "refund.review" in capsys.readouterr().out


def test_inspector_handles_incomplete_lines_and_unfinished_spans(tmp_path, capsys):
    path = tmp_path / "partial.jsonl"
    event = {"trace_id": "one", "span_id": "root", "parent_span_id": None,
             "name": "message", "phase": "start"}
    path.write_text(json.dumps(event) + '\n{"partial":', encoding="utf-8")
    assert show_main([str(path)]) == 0
    output = capsys.readouterr()
    assert "incomplete" in output.out
    assert "Skipped 1" in output.err


def test_cli_can_disable_or_override_tracing(monkeypatch, capsys, tmp_path, store):
    llm = Mock(spec=OllamaClient)
    llm.chat_json.return_value = IntentResult(intent="order_status", order_id="O1001")
    monkeypatch.setattr(cli, "load_settings", lambda: object())
    monkeypatch.setattr(cli, "Store", lambda: store)
    monkeypatch.setattr(cli, "make_llm", lambda settings: llm)
    path = tmp_path / "cli.jsonl"
    assert cli.main(["Check O1001", "--trace-file", str(path), "--no-trace"]) == 0
    assert not path.exists()
    assert cli.main(["Check O1001", "--trace-file", str(path)]) == 0
    assert ends(path, "message")[0]["output"]["state_after"]["last_order_id"] == "O1001"


def test_decorated_tool_error_keeps_original_return_value(tmp_path):
    @traced("tool.failure", inputs=("value",))
    def fail(value):
        return {"ok": False, "error": "invalid", "message": value}

    path = tmp_path / "tool.jsonl"
    with JsonlTracer(path).trace("root"):
        result = fail("bad input")
    assert ends(path, "tool.failure")[0]["output"] == result
    assert ends(path, "tool.failure")[0]["status"] == "error"
