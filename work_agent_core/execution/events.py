from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol
import threading
import time


@dataclass(frozen=True)
class ExecutionEvent:
    execution_id: str
    index: int
    type: str
    ts_ms: int
    phase: str
    summary: str
    payload: dict[str, Any]
    visibility: str = "user"

    def as_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "event_index": self.index,
            "type": self.type,
            "ts_ms": self.ts_ms,
            "phase": self.phase,
            "summary": self.summary,
            "payload": self.payload,
            "visibility": self.visibility,
        }


class ExecutionEventSink(Protocol):
    def emit(
        self,
        event_type: str,
        *,
        phase: str,
        summary: str,
        payload: dict[str, Any] | None = None,
        visibility: str = "user",
    ) -> ExecutionEvent: ...


class EventEmitter:
    """Thread-safe bridge from a backend to durable events and current Turn UI."""

    def __init__(
        self,
        execution_id: str,
        append: Callable[[ExecutionEvent], ExecutionEvent],
        on_event: Callable[[ExecutionEvent], None] | None = None,
        *,
        initial_index: int = -1,
    ) -> None:
        self.execution_id = execution_id
        self._append = append
        self._on_event = on_event
        self._index = initial_index
        self._lock = threading.RLock()

    def emit(
        self,
        event_type: str,
        *,
        phase: str,
        summary: str,
        payload: dict[str, Any] | None = None,
        visibility: str = "user",
    ) -> ExecutionEvent:
        with self._lock:
            self._index += 1
            event = ExecutionEvent(
                execution_id=self.execution_id,
                index=self._index,
                type=event_type,
                ts_ms=int(time.time() * 1000),
                phase=phase,
                summary=summary,
                payload=dict(payload or {}),
                visibility=visibility,
            )
            stored = self._append(event)
        if self._on_event is not None:
            self._on_event(stored)
        return stored
