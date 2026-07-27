from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import time

from .turn_store import AgentTurn, TurnStore


class TurnCancelled(RuntimeError):
    """Raised when the user cancels the current single-agent turn."""


@dataclass
class TurnRuntime:
    store: TurnStore
    turn: AgentTurn
    started_at: float

    @classmethod
    def start(
        cls,
        store: TurnStore,
        *,
        conversation_id: str,
        trace_id: str = "",
        profile: str = "",
        model: str = "",
        route: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> "TurnRuntime":
        return cls(
            store=store,
            turn=store.create(
                conversation_id=conversation_id,
                trace_id=trace_id,
                profile=profile,
                model=model,
                route=route,
                metadata=metadata,
            ),
            started_at=time.monotonic(),
        )

    @classmethod
    def resume(cls, store: TurnStore, turn_id: str) -> "TurnRuntime":
        turn = store.mark_running(turn_id)
        return cls(store=store, turn=turn, started_at=time.monotonic())

    @property
    def turn_id(self) -> str:
        return self.turn.id

    def cancelled(self) -> bool:
        return self.store.is_cancel_requested(self.turn.id)

    def raise_if_cancelled(self) -> None:
        if self.cancelled():
            raise TurnCancelled("用户停止了当前轮。")

    def emit(self, event: dict[str, Any]) -> dict[str, Any]:
        payload = dict(event)
        payload.setdefault("turn_id", self.turn.id)
        payload.setdefault("conversation_id", self.turn.conversation_id)
        if "elapsed_ms" not in payload:
            payload["elapsed_ms"] = int((time.monotonic() - self.started_at) * 1000)
        return self.store.append_event(self.turn.id, payload)

    def initial_event(self) -> dict[str, Any]:
        return self.emit(
            {
                "event": "turn",
                "turn_id": self.turn.id,
                "conversation_id": self.turn.conversation_id,
                "turn_status": self.turn.status,
                "trace_id": self.turn.trace_id,
                "profile": self.turn.profile,
                "model": self.turn.model,
            }
        )

    def cancel_event(self, *, message: str = "已停止当前轮处理。") -> dict[str, Any]:
        self.store.mark_cancelled(self.turn.id, reason=message)
        return self.emit(
            {
                "event": "cancelled",
                "message": message,
                "turn_status": "cancelled",
            }
        )

    def fail_event(self, error: Exception) -> dict[str, Any]:
        message = str(error) or type(error).__name__
        self.store.mark_failed(self.turn.id, error=message)
        return self.emit(
            {
                "event": "error",
                "message": message,
                "type": type(error).__name__,
                "detail": message,
            }
        )
