from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json
import re
import threading
import time
import uuid


TURN_SCHEMA_VERSION = 1
DEFAULT_TURN_DIR = Path("meet_files/conversation_history/turns")
TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}
_TURN_LOCK = threading.RLock()


@dataclass
class AgentTurn:
    id: str
    conversation_id: str
    status: str = "running"
    trace_id: str = ""
    profile: str = ""
    model: str = ""
    route: str = ""
    created_at: int = field(default_factory=lambda: int(time.time()))
    updated_at: int = field(default_factory=lambda: int(time.time()))
    latest_event_index: int = -1
    events: list[dict[str, Any]] = field(default_factory=list)
    final_message: str = ""
    error: str = ""
    cancel_requested: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": TURN_SCHEMA_VERSION,
            "id": self.id,
            "conversation_id": self.conversation_id,
            "status": self.status,
            "trace_id": self.trace_id,
            "profile": self.profile,
            "model": self.model,
            "route": self.route,
            "created_at": self.created_at,
            "updated_at": int(time.time()),
            "latest_event_index": self.latest_event_index,
            "events": [sanitize_turn_event(event) for event in self.events],
            "final_message": self.final_message,
            "error": self.error,
            "cancel_requested": self.cancel_requested,
            "metadata": sanitize_json_value(self.metadata),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any], *, fallback_id: str) -> AgentTurn:
        events = payload.get("events") if isinstance(payload.get("events"), list) else []
        return cls(
            id=sanitize_turn_id(payload.get("id")) or fallback_id,
            conversation_id=sanitize_conversation_ref(payload.get("conversation_id")),
            status=sanitize_turn_status(payload.get("status")),
            trace_id=str(payload.get("trace_id") or ""),
            profile=str(payload.get("profile") or ""),
            model=str(payload.get("model") or ""),
            route=str(payload.get("route") or ""),
            created_at=max(0, int(payload.get("created_at") or time.time())),
            updated_at=max(0, int(payload.get("updated_at") or time.time())),
            latest_event_index=sanitize_event_index(payload.get("latest_event_index")),
            events=[sanitize_turn_event(event) for event in events if isinstance(event, dict)],
            final_message=str(payload.get("final_message") or ""),
            error=str(payload.get("error") or ""),
            cancel_requested=bool(payload.get("cancel_requested")),
            metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
        )


class TurnStore:
    """Persistent single-agent turn runtime state.

    A turn is one user-submitted agent run inside one conversation. This is not
    a multi-agent job queue: it only records status, cancellation intent and the
    SSE/activity events for the current single-agent ReAct loop.
    """

    def __init__(self, workspace_root: str | Path, *, turn_dir: str | Path = DEFAULT_TURN_DIR) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        directory = Path(turn_dir)
        if not directory.is_absolute():
            directory = self.workspace_root / directory
        self.turn_dir = directory

    def create(
        self,
        *,
        conversation_id: str,
        trace_id: str = "",
        profile: str = "",
        model: str = "",
        route: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> AgentTurn:
        turn = AgentTurn(
            id=new_turn_id(),
            conversation_id=sanitize_conversation_ref(conversation_id),
            trace_id=str(trace_id or ""),
            profile=str(profile or ""),
            model=str(model or ""),
            route=str(route or ""),
            metadata=sanitize_json_value(metadata or {}),
        )
        with _TURN_LOCK:
            self._write(turn)
        return turn

    def load(self, turn_id: str) -> AgentTurn:
        safe_id = sanitize_turn_id(turn_id)
        if not safe_id:
            raise ValueError("turn_id is required")
        path = self._path_for(safe_id)
        with _TURN_LOCK:
            if not path.is_file():
                raise FileNotFoundError(f"Turn not found: {safe_id}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError(f"Invalid turn payload: {safe_id}")
            return AgentTurn.from_payload(payload, fallback_id=safe_id)

    def append_event(self, turn_id: str, event: dict[str, Any]) -> dict[str, Any]:
        with _TURN_LOCK:
            turn = self.load(turn_id)
            next_index = len(turn.events)
            stored_event = sanitize_turn_event(
                {
                    **event,
                    "turn_id": turn.id,
                    "conversation_id": turn.conversation_id,
                    "event_index": next_index,
                    "ts_ms": int(time.time() * 1000),
                }
            )
            turn.events.append(stored_event)
            turn.latest_event_index = next_index
            apply_event_status(turn, stored_event)
            turn.updated_at = int(time.time())
            self._write(turn)
            return stored_event

    def set_pending_approval(self, turn_id: str, pending_approval: dict[str, Any]) -> AgentTurn:
        with _TURN_LOCK:
            turn = self.load(turn_id)
            turn.metadata["pending_approval"] = sanitize_json_value(pending_approval)
            turn.status = "waiting_approval"
            turn.updated_at = int(time.time())
            self._write(turn)
            return turn

    def pending_approval(self, turn_id: str) -> dict[str, Any]:
        turn = self.load(turn_id)
        pending = turn.metadata.get("pending_approval")
        return pending if isinstance(pending, dict) else {}

    def pending_approval_for_conversation(
        self,
        conversation_id: str,
    ) -> tuple[str, dict[str, Any]] | None:
        safe_conversation_id = sanitize_conversation_ref(conversation_id)
        if not safe_conversation_id:
            return None

        newest: tuple[AgentTurn, dict[str, Any]] | None = None
        with _TURN_LOCK:
            if not self.turn_dir.is_dir():
                return None
            for path in self.turn_dir.glob("*.json"):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if not isinstance(payload, dict):
                    continue
                turn = AgentTurn.from_payload(payload, fallback_id=path.stem)
                if turn.conversation_id != safe_conversation_id:
                    continue
                if turn.status not in {"waiting_approval", "running", "failed"}:
                    continue
                pending = turn.metadata.get("pending_approval")
                if not isinstance(pending, dict) or not pending:
                    continue
                if newest is None or turn.updated_at >= newest[0].updated_at:
                    newest = (turn, pending)

        if newest is None:
            return None
        turn, pending = newest
        return turn.id, pending

    def mark_running(self, turn_id: str) -> AgentTurn:
        with _TURN_LOCK:
            turn = self.load(turn_id)
            if turn.status not in TERMINAL_STATUSES:
                turn.status = "running"
            turn.updated_at = int(time.time())
            self._write(turn)
            return turn

    def clear_pending_approval(self, turn_id: str) -> AgentTurn:
        with _TURN_LOCK:
            turn = self.load(turn_id)
            turn.metadata.pop("pending_approval", None)
            turn.updated_at = int(time.time())
            self._write(turn)
            return turn

    def discard_pending_for_conversation(self, conversation_id: str) -> int:
        """Retire stale approval/running turns before a conversation rewind."""

        safe_conversation_id = sanitize_conversation_ref(conversation_id)
        if not safe_conversation_id:
            return 0
        discarded = 0
        with _TURN_LOCK:
            if not self.turn_dir.is_dir():
                return 0
            for path in self.turn_dir.glob("*.json"):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if not isinstance(payload, dict):
                    continue
                turn = AgentTurn.from_payload(payload, fallback_id=path.stem)
                if turn.conversation_id != safe_conversation_id:
                    continue
                pending = turn.metadata.get("pending_approval")
                if not isinstance(pending, dict) or not pending:
                    continue
                turn.metadata.pop("pending_approval", None)
                turn.cancel_requested = True
                if turn.status not in TERMINAL_STATUSES:
                    turn.status = "cancelled"
                    turn.error = "会话已从更早的用户消息重新发送。"
                self._write(turn)
                discarded += 1
        return discarded

    def request_cancel(self, turn_id: str) -> AgentTurn:
        with _TURN_LOCK:
            turn = self.load(turn_id)
            turn.cancel_requested = True
            if turn.status not in TERMINAL_STATUSES:
                turn.status = "running"
            turn.updated_at = int(time.time())
            self._write(turn)
            return turn

    def mark_cancelled(self, turn_id: str, *, reason: str = "用户停止了当前轮。") -> AgentTurn:
        with _TURN_LOCK:
            turn = self.load(turn_id)
            turn.cancel_requested = True
            turn.status = "cancelled"
            turn.error = reason
            turn.updated_at = int(time.time())
            self._write(turn)
            return turn

    def mark_failed(self, turn_id: str, *, error: str) -> AgentTurn:
        with _TURN_LOCK:
            turn = self.load(turn_id)
            if turn.status not in TERMINAL_STATUSES:
                turn.status = "failed"
            turn.error = str(error or "")
            turn.updated_at = int(time.time())
            self._write(turn)
            return turn

    def is_cancel_requested(self, turn_id: str) -> bool:
        try:
            return self.load(turn_id).cancel_requested
        except FileNotFoundError:
            return False

    def events_after(self, turn_id: str, *, after: int = -1) -> list[dict[str, Any]]:
        turn = self.load(turn_id)
        return [
            event
            for event in turn.events
            if int(event.get("event_index") if event.get("event_index") is not None else -1) > after
        ]

    def _path_for(self, turn_id: str) -> Path:
        return self.turn_dir / f"{sanitize_turn_id(turn_id)}.json"

    def _write(self, turn: AgentTurn) -> None:
        turn.id = sanitize_turn_id(turn.id)
        turn.conversation_id = sanitize_conversation_ref(turn.conversation_id)
        turn.status = sanitize_turn_status(turn.status)
        turn.updated_at = int(time.time())
        path = self._path_for(turn.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(turn.to_payload(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(path)


def new_turn_id() -> str:
    return f"turn-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"


def apply_event_status(turn: AgentTurn, event: dict[str, Any]) -> None:
    event_name = str(event.get("event") or "")
    if event_name == "activity" and (
        event.get("command_status") == "approval_required" or event.get("approval_required")
    ):
        turn.status = "waiting_approval"
        return
    if event_name == "final":
        turn.status = "waiting_approval" if event.get("waiting_approval") else "succeeded"
        turn.final_message = str(event.get("content") or "")
        turn.error = ""
        return
    if event_name == "cancelled":
        turn.status = "cancelled"
        turn.error = str(event.get("message") or event.get("detail") or "用户停止了当前轮。")
        return
    if event_name == "error":
        turn.status = "failed"
        turn.error = str(event.get("message") or event.get("detail") or "")
        return


def sanitize_turn_id(raw_value: Any) -> str:
    text = str(raw_value or "").strip()
    text = re.sub(r"[^A-Za-z0-9_.:-]+", "-", text).strip(".-")
    return text[:120]


def sanitize_conversation_ref(raw_value: Any) -> str:
    text = str(raw_value or "").strip()
    text = re.sub(r"[^A-Za-z0-9_.:-]+", "-", text).strip(".-")
    return text[:120]


def sanitize_turn_status(raw_value: Any) -> str:
    status = str(raw_value or "running").strip()
    allowed = {"queued", "running", "waiting_approval", "succeeded", "failed", "cancelled"}
    return status if status in allowed else "running"


def sanitize_event_index(raw_value: Any) -> int:
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return -1


def sanitize_turn_event(raw_event: Any) -> dict[str, Any]:
    if not isinstance(raw_event, dict):
        return {}
    return sanitize_json_value(raw_event)


def sanitize_json_value(value: Any, *, depth: int = 0) -> Any:
    if depth > 12:
        return str(value)
    if isinstance(value, dict):
        return {str(key): sanitize_json_value(item, depth=depth + 1) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_json_value(item, depth=depth + 1) for item in value]
    if isinstance(value, tuple):
        return [sanitize_json_value(item, depth=depth + 1) for item in value]
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)
