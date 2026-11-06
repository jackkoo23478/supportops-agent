"""Local JSONL spans with context-local parentage and best-effort writes."""

import inspect
import json
import sys
import threading
from argparse import ArgumentParser, Namespace
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from functools import wraps
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

from pydantic import BaseModel


SECRET_KEYS = {
    "api_key", "anthropic_api_key", "openai_api_key", "authorization",
    "password", "secret", "access_token", "refresh_token",
}


def _json_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _json_value(value.model_dump(mode="json"))
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]" if str(key).lower() in SECRET_KEYS else _json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, (Decimal, Path)):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return f"<{type(value).__name__}>"


@dataclass
class Span:
    output: Any = None
    status: str = "ok"
    error: dict | None = None
    metrics: dict = field(default_factory=dict)

    def record(self, result: Any) -> None:
        self.output = result
        if isinstance(result, dict) and result.get("ok") is False:
            self.status = "error"
            self.error = {"code": result.get("error"), "message": result.get("message")}


@dataclass
class _Context:
    tracer: "JsonlTracer"
    trace_id: str
    session_id: str
    span_id: str
    sequence: list[int]


_current: ContextVar[_Context | None] = ContextVar("supportops_trace", default=None)


class JsonlTracer:
    """Append spans without changing application behavior if tracing fails."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._warned = False

    def emit(self, data: dict) -> None:
        try:
            # Serialize now so later state mutations cannot change earlier events.
            line = json.dumps(_json_value(data), ensure_ascii=False, allow_nan=False) + "\n"
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as stream:
                    stream.write(line)
        except (OSError, TypeError, ValueError):
            if not self._warned:
                self._warned = True
                print("Trace output is unavailable; processing will continue.", file=sys.stderr)

    @contextmanager
    def trace(self, name: str, *, inputs: Any = None, session_id: str | None = None):
        """Start a new trace. Nested operations inherit it through ContextVar."""
        root = _Context(self, uuid4().hex, session_id or uuid4().hex, "", [0])
        token = _current.set(root)
        try:
            with span(name, inputs=inputs) as active:
                yield active
        finally:
            _current.reset(token)


@contextmanager
def span(name: str, *, inputs: Any = None):
    parent = _current.get()
    active = Span()
    if parent is None:
        yield active
        return
    span_id = uuid4().hex
    context = _Context(parent.tracer, parent.trace_id, parent.session_id, span_id, parent.sequence)

    def emit(phase: str, **fields: Any) -> None:
        context.sequence[0] += 1
        context.tracer.emit({
            "schema_version": 1, "timestamp": datetime.now(timezone.utc),
            "trace_id": context.trace_id, "session_id": context.session_id,
            "span_id": span_id, "parent_span_id": parent.span_id or None,
            "sequence": context.sequence[0], "name": name, "phase": phase, **fields,
        })

    started = perf_counter()
    emit("start", inputs=inputs)
    token = _current.set(context)
    try:
        yield active
    except BaseException as exc:
        active.status = "error"
        active.error = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        emit(
            "end", status=active.status, output=active.output, error=active.error,
            elapsed_ms=round((perf_counter() - started) * 1000, 3), metrics=active.metrics,
        )
        _current.reset(token)


def traced(name: str, *, inputs: tuple[str, ...] = ()):
    """Trace selected arguments and the returned value; never serialize clients."""
    def decorate(function):
        signature = inspect.signature(function)

        @wraps(function)
        def wrapped(*args, **kwargs):
            if _current.get() is None:
                return function(*args, **kwargs)
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            selected = {key: bound.arguments[key] for key in inputs}
            if name in {"llm.chat", "llm.structured"}:
                selected["model"] = bound.arguments["self"].model
            with span(name, inputs=selected) as active:
                result = function(*args, **kwargs)
                active.record(result)
                if name == "llm.chat":
                    active.metrics = {
                        "prompt_tokens": result.prompt_tokens,
                        "completion_tokens": result.completion_tokens,
                        "model_duration_ms": result.duration_ms,
                    }
                return result

        return wrapped
    return decorate


def add_trace_arguments(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--trace-file", type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "runtime" / "traces.jsonl",
        help="Local JSONL trace output path.",
    )
    parser.add_argument("--no-trace", action="store_true", help="Disable local tracing.")


@contextmanager
def command_trace(args: Namespace, name: str, *, inputs: Any = None):
    if args.no_trace:
        token = _current.set(None)
        try:
            yield Span()
        finally:
            _current.reset(token)
    else:
        with JsonlTracer(args.trace_file).trace(name, inputs=inputs) as active:
            yield active
