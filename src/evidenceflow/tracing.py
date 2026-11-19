"""RunTrace recorder: the TraceRecorder protocol, a no-op implementation, an in-memory
implementation and a safe recording helper (PROJECT_SPEC §12.5).

This module only observes; it never changes any answer or exception propagation:
- The components (DenseRerankerRetriever, Generator, RAGService) default ``trace_recorder`` to
  ``None``, in which case the caller does not build a payload at all, so the overhead matches an
  untraced deployment;
- Components always record through ``safe_record``, and any exception from the recorder itself is
  swallowed with only a warning log line.
- When shared components serve concurrent requests, ``trace_scope`` pins the recorder of the
  current thread/coroutine for one run and components pick it up through ``resolve_recorder``; an
  explicitly injected recorder still wins.
Importing ``evidenceflow.eval`` is not allowed (runtime-side modules must not depend on the
evaluation layer in reverse).
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Protocol

from evidenceflow.domain.context import BuiltContext
from evidenceflow.domain.chunk import DocumentChunk
from evidenceflow.domain.trace import RunTrace, TraceEvent

logger = logging.getLogger(__name__)

_COMPLETED_SUFFIX = ".completed"
_STARTED_SUFFIX = ".started"
_VIEWER_FIELDS = {
    "agent.search.started": {"query", "document_id"},
    "agent.search.completed": {"passages"},
    "retrieval.branch.planned": {"queries"},
    "agent.open.completed": {"passages", "anchor"},
    "agent.memory.reused": {"passages"},
    "agent.scope.selected": {"document_id", "file_name"},
}
_ACTIVE_RECORDER: ContextVar["TraceRecorder | None"] = ContextVar(
    "evidenceflow_active_trace_recorder", default=None
)


class TraceRecorder(Protocol):
    """The minimal interface components need to record trace events."""

    def record(self, name: str, **payload: Any) -> None:
        """Record one event; the payload may only contain the §12.5 allowlisted fields."""


class NullTraceRecorder:
    """A recorder that does nothing, for callers to use explicitly (components default to None,
    not to this)."""

    def record(self, name: str, **payload: Any) -> None:
        """Discard the event."""


@contextmanager
def trace_scope(recorder: TraceRecorder | None) -> Iterator[TraceRecorder | None]:
    """Make ``recorder`` the recorder of the current run in this context, restoring it on exit.

    A ContextVar is isolated per thread and per asyncio task, so several requests can share one set
    of Runtime components and still send their events only to their own recorder, without swapping
    ``trace_recorder`` on the shared components between requests.
    """

    token = _ACTIVE_RECORDER.set(recorder)
    try:
        yield recorder
    finally:
        _ACTIVE_RECORDER.reset(token)


def resolve_recorder(explicit: TraceRecorder | None) -> TraceRecorder | None:
    """A recorder passed explicitly at construction time wins; otherwise the current recorder set
    by ``trace_scope`` is used."""

    return explicit if explicit is not None else _ACTIVE_RECORDER.get()


def safe_record(recorder: TraceRecorder | None, name: str, **payload: Any) -> None:
    """Record one event safely: return immediately when the recorder is None; if the recorder
    raises, log only the exception class name and swallow it.

    Tracing is side-channel observation, so a defect in the recorder must never change the answer
    or the original exception propagation path. The log carries only the exception class name, not
    the message, to avoid writing potentially content-bearing text into the log.
    """

    if recorder is None:
        return
    try:
        recorder.record(name, **payload)
    except Exception as exc:  # noqa: BLE001 - observation failure must not affect the main flow
        logger.warning(
            "trace recorder failed", extra={"trace_event": name, "error_type": type(exc).__name__}
        )


def record_built_context(recorder: TraceRecorder | None, context: BuiltContext) -> None:
    """记录实际纳入的证据身份和 token 用量，正文只记录摘要值。"""

    if recorder is None:
        return
    safe_record(
        recorder,
        "context.completed",
        included_count=context.included_count,
        skipped_count=context.skipped_count,
        total_tokens=context.total_tokens,
        entries=[
            {"source_label": entry.source_label, "chunk_id": entry.chunk_id, "token_count": entry.token_count,
             **(entry.source_reference.model_dump(mode="json") if getattr(recorder, "tool_details", False) else {})}
            for entry in context.entries
        ],
        context_text_sha256=hashlib.sha256(context.context_text.encode("utf-8")).hexdigest(),
    )


def passage_identity(chunk: DocumentChunk) -> dict[str, Any]:
    """给取证面板提供可回读的原文身份，正文由快照接口按需读取。"""
    return chunk.model_dump(include={"chunk_id", "document_id", "file_name", "page_start", "page_end"})


def _utc_now() -> datetime:
    """The current UTC instant (replaceable with a fixed clock in tests)."""

    return datetime.now(timezone.utc)


class RunTraceRecorder:
    """Keep one run's events in memory and produce a ``RunTrace`` on ``finish``.

    Usage: ``start_run(query)`` -> components call ``record`` -> ``finish()``. The same instance
    can be reused across runs, but ``start_run`` must be called again before each run; earlier
    events are cleared.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.perf_counter,
        utc_now: Callable[[], datetime] = _utc_now,
        on_event: Callable[[TraceEvent], None] | None = None,
        tool_details: bool = False,
    ) -> None:
        """注入计时函数及可选的展示回调；回调在事件入列后收到副本。"""

        self._clock = clock
        self._utc_now = utc_now
        self._on_event = on_event
        self.tool_details = tool_details
        self._started = False
        self._run_id = ""
        self._query_sha256 = ""
        self._started_at_utc = ""
        self._origin = 0.0
        self._events: list[TraceEvent] = []
        # Record the elapsed_ms of the latest started event for each stage prefix, used to compute
        # duration_ms for X.completed.
        self._last_started_ms: dict[str, float] = {}

    def start_run(self, query: str, *, run_id: str | None = None) -> None:
        """Start a new run: reset the events and record the query digest, the start instant and the
        monotonic clock origin."""

        self._run_id = run_id if run_id is not None else str(uuid.uuid4())
        # Store only the sha256 of the UTF-8 query; the raw query never appears in the trace.
        self._query_sha256 = hashlib.sha256(query.encode("utf-8")).hexdigest()
        self._started_at_utc = self._utc_now().isoformat()
        self._origin = self._clock()
        self._events = []
        self._last_started_ms = {}
        self._started = True

    def record(self, name: str, **payload: Any) -> None:
        """Append one event; a call before ``start_run`` is only logged and dropped, never
        raised."""

        if not self._started:
            logger.warning("trace event recorded before start_run; dropped", extra={"trace_event": name})
            return
        # elapsed_ms is based on the monotonic clock and is unaffected by system time adjustments.
        elapsed_ms = (self._clock() - self._origin) * 1000.0
        duration_ms: float | None = None
        if name.endswith(_STARTED_SUFFIX):
            self._last_started_ms[name[: -len(_STARTED_SUFFIX)]] = elapsed_ms
        elif name.endswith(_COMPLETED_SUFFIX):
            # A duration exists only when a matching X.started is present
            # (citation_validation.completed has no started event).
            started_ms = self._last_started_ms.get(name[: -len(_COMPLETED_SUFFIX)])
            if started_ms is not None:
                duration_ms = elapsed_ms - started_ms
        event = TraceEvent(
            seq=len(self._events), name=name, elapsed_ms=elapsed_ms,
            duration_ms=duration_ms, payload=dict(payload),
        )
        self._events.append(event)
        if self._on_event is not None:
            self._on_event(event.model_copy(deep=True))

    @property
    def events(self) -> list[TraceEvent]:
        """A copy of the events recorded so far in the current run (read-only view)."""

        return [event.model_copy(deep=True) for event in self._events]

    def finish(self) -> RunTrace:
        """Finish the current run and return the RunTrace; the last event must be run.completed or
        run.failed."""

        if not self._events:
            raise ValueError("run trace has no events; cannot finish")
        last = self._events[-1].name
        if last == "run.completed":
            status = "completed"
        elif last == "run.failed":
            status = "failed"
        else:
            raise ValueError(f"run trace must end with run.completed or run.failed, got {last!r}")
        events = self.events
        if self.tool_details:
            # 取证面板通过 SSE 独立读取查询及原文身份，通用 RunTrace 保留原有脱敏合同。
            for event in events:
                for key in _VIEWER_FIELDS.get(event.name, ()):
                    event.payload.pop(key, None)
                if event.name == "context.completed":
                    event.payload["entries"] = [
                        {key: row[key] for key in ("source_label", "chunk_id", "token_count")}
                        for row in event.payload["entries"]
                    ]
        return RunTrace(
            run_id=self._run_id,
            query_sha256=self._query_sha256,
            started_at_utc=self._started_at_utc,
            completed_at_utc=self._utc_now().isoformat(),
            status=status,
            events=events,
        )
