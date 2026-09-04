"""Append-only session event log and the projections derived from it.

The log is the source of truth for one conversation. LLM message history is
*derived* from it, never stored separately, so the model input can always be
reconstructed from what was durably recorded.

Three layers:

``SessionEvent``
    One immutable durable fact. ``surface_op`` decides whether it participates
    in the model-facing projection.

surface
    The ordered projection of message-producing events. Compaction appends a
    ``compaction/replacement`` that *shadows* the events it cites instead of
    deleting them, so the raw log keeps full fidelity while the model sees the
    compacted view.

projections
    ``derive_messages()`` builds the exact provider message list from the
    surface. ``derive_transcript()`` walks append-origin events only, which is
    what a human reader wants: a landed replacement shadows history the reader
    has already seen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Iterator, Mapping, Sequence
import json
import time


SESSION_LOG_FORMAT_VERSION = 1

# Surface participation. ``none`` is log-only: recorded, never sent to a model.
SURFACE_APPEND = "append"
SURFACE_REPLACE = "replace"
SURFACE_NONE = "none"

# Lifecycle
SESSION_CREATED = "session/created"
TURN_START = "turn/start"
TURN_END = "turn/end"
STEP_START = "step/start"
STEP_END = "step/end"

# Model-visible
USER_MESSAGE = "user/message"
ASSISTANT_MESSAGE = "assistant/message"
TOOL_CALL = "tool/call"
TOOL_RESULT = "tool/result"
CONTEXT_INJECTED = "context/injected"
COMPACTION_REPLACEMENT = "compaction/replacement"

# Log-only
ASSISTANT_CHUNK = "assistant/chunk"
REQUEST_HEADER = "request/header"
APPROVAL_REQUESTED = "approval/requested"
APPROVAL_RESOLVED = "approval/resolved"
PLAN_UPDATED = "plan/updated"
AGENT_ERROR = "agent/error"
FACT_RECORDED = "memory/fact"

# Human timeline and business state.  These are durable facts, but they are not
# provider messages.  The model-facing projection and the human/product
# projections deliberately diverge here: a delivery receipt or a bell reminder
# must survive a restart without being replayed as if the user had said it.
PROACTIVE_MESSAGE = "message/proactive"
MESSAGE_READ = "message/read"
ARTIFACT_CREATED = "artifact/created"
ARTIFACT_VERIFIED = "artifact/verified"
DELIVERY_UPDATED = "delivery/updated"
SESSION_CHECKPOINT = "session/checkpoint"
SESSION_METADATA_UPDATED = "session/metadata_updated"
EXTERNAL_EVENT_QUEUED = "external/queued"
EXTERNAL_EVENT_CONSUMED = "external/consumed"

DEFAULT_SURFACE_OPS: dict[str, str] = {
    SESSION_CREATED: SURFACE_NONE,
    TURN_START: SURFACE_NONE,
    TURN_END: SURFACE_NONE,
    STEP_START: SURFACE_NONE,
    STEP_END: SURFACE_NONE,
    USER_MESSAGE: SURFACE_APPEND,
    ASSISTANT_MESSAGE: SURFACE_APPEND,
    TOOL_CALL: SURFACE_NONE,
    TOOL_RESULT: SURFACE_APPEND,
    CONTEXT_INJECTED: SURFACE_APPEND,
    COMPACTION_REPLACEMENT: SURFACE_REPLACE,
    ASSISTANT_CHUNK: SURFACE_NONE,
    REQUEST_HEADER: SURFACE_NONE,
    APPROVAL_REQUESTED: SURFACE_NONE,
    APPROVAL_RESOLVED: SURFACE_NONE,
    PLAN_UPDATED: SURFACE_NONE,
    AGENT_ERROR: SURFACE_NONE,
    FACT_RECORDED: SURFACE_NONE,
    PROACTIVE_MESSAGE: SURFACE_NONE,
    MESSAGE_READ: SURFACE_NONE,
    ARTIFACT_CREATED: SURFACE_NONE,
    ARTIFACT_VERIFIED: SURFACE_NONE,
    DELIVERY_UPDATED: SURFACE_NONE,
    SESSION_CHECKPOINT: SURFACE_NONE,
    SESSION_METADATA_UPDATED: SURFACE_NONE,
    EXTERNAL_EVENT_QUEUED: SURFACE_NONE,
    EXTERNAL_EVENT_CONSUMED: SURFACE_NONE,
}

# ``interrupted`` is the one turn-end reason no loop emits. Recovery synthesizes
# it for a turn whose ``turn/start`` never got a matching end, so the value
# itself proves where the record came from without a second marker.
TURN_END_COMPLETED = "completed"
TURN_END_ABORTED = "aborted"
TURN_END_FAILED = "failed"
TURN_END_DISPOSED = "disposed"
TURN_END_INTERRUPTED = "interrupted"
LOOP_EMITTED_TURN_END_REASONS = frozenset(
    {TURN_END_COMPLETED, TURN_END_ABORTED, TURN_END_FAILED, TURN_END_DISPOSED}
)

ABORTED_BEFORE_DISPATCH = "ABORTED_BEFORE_DISPATCH"


class SessionLogError(RuntimeError):
    """Raised when an append or a projection would break a log invariant."""


@dataclass(frozen=True)
class SessionEvent:
    seq: int
    type: str
    ts_ms: int
    data: Mapping[str, Any]
    source_event_seqs: tuple[int, ...] = ()
    surface_op: str = SURFACE_APPEND

    def to_row(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "type": self.type,
            "ts_ms": self.ts_ms,
            "data": json.dumps(thaw_json_value(self.data), ensure_ascii=False),
            "source_event_seqs": json.dumps(list(self.source_event_seqs)),
            "surface_op": self.surface_op,
        }

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> SessionEvent:
        raw_sources = row.get("source_event_seqs")
        if isinstance(raw_sources, str):
            raw_sources = json.loads(raw_sources or "[]")
        raw_data = row.get("data")
        if isinstance(raw_data, str):
            raw_data = json.loads(raw_data or "{}")
        return cls(
            seq=int(row["seq"]),
            type=str(row["type"]),
            ts_ms=int(row.get("ts_ms") or 0),
            data=freeze_json_value(snapshot_json_value(raw_data) or {}),
            source_event_seqs=tuple(int(item) for item in (raw_sources or ())),
            surface_op=str(row.get("surface_op") or SURFACE_APPEND),
        )


@dataclass(frozen=True)
class SurfaceEntry:
    """One model-visible node, plus the seqs a replacement would shadow."""

    seq: int
    message: Mapping[str, Any]
    shadowed: bool = False


@dataclass
class SessionHeader:
    session_id: str
    version: int = SESSION_LOG_FORMAT_VERSION
    created_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    cwd: str = ""
    project_id: str = ""
    seed_length: int = 0
    parent_session: str = ""

    def to_payload(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "version": self.version,
            "created_at_ms": self.created_at_ms,
            "cwd": self.cwd,
            "project_id": self.project_id,
            "seed_length": self.seed_length,
            "parent_session": self.parent_session,
        }


class SessionLog:
    """In-memory append-only log with an incrementally maintained surface.

    Durable data is snapshotted and frozen at ``append``. A caller that keeps a
    reference to the dict it passed in cannot reach back into recorded history
    by mutating it later, which is the failure mode that lets a rendered reply
    quietly rewrite a message the model already saw.
    """

    def __init__(self, header: SessionHeader, events: Iterable[SessionEvent] = ()) -> None:
        self.header = header
        self._events: list[SessionEvent] = []
        self._surface: list[SurfaceEntry] = []
        self._shadowed_seqs: set[int] = set()
        for event in events:
            self._accept(event)

    # -- reading ---------------------------------------------------------

    @property
    def seq(self) -> int:
        """Next sequence number this log will assign."""
        return len(self._events)

    @property
    def events(self) -> tuple[SessionEvent, ...]:
        return tuple(self._events)

    def event(self, seq: int) -> SessionEvent:
        if not 0 <= seq < len(self._events):
            raise SessionLogError(f"事件 seq 不存在：{seq}")
        return self._events[seq]

    def iter_type(self, event_type: str) -> Iterator[SessionEvent]:
        return (event for event in self._events if event.type == event_type)

    def latest(self, event_type: str) -> SessionEvent | None:
        for event in reversed(self._events):
            if event.type == event_type:
                return event
        return None

    # -- writing ---------------------------------------------------------

    def append(
        self,
        event_type: str,
        data: Mapping[str, Any] | None = None,
        *,
        source_event_seqs: Sequence[int] = (),
        surface_op: str | None = None,
        ts_ms: int | None = None,
    ) -> SessionEvent:
        snapshot = snapshot_json_value(dict(data or {}))
        if snapshot is None:
            raise SessionLogError(f"{event_type} 的 data 不是可持久化的 JSON 值。")
        resolved_op = surface_op or DEFAULT_SURFACE_OPS.get(event_type, SURFACE_APPEND)
        if resolved_op not in {SURFACE_APPEND, SURFACE_REPLACE, SURFACE_NONE}:
            raise SessionLogError(f"未知 surface_op：{resolved_op}")
        sources = tuple(int(item) for item in source_event_seqs)
        for cited in sources:
            if not 0 <= cited < len(self._events):
                raise SessionLogError(f"{event_type} 引用了不存在的事件 seq：{cited}")
        if resolved_op == SURFACE_REPLACE and not sources:
            raise SessionLogError("replacement 事件必须声明它遮蔽的 source_event_seqs。")
        event = SessionEvent(
            seq=len(self._events),
            type=str(event_type),
            ts_ms=int(ts_ms if ts_ms is not None else time.time() * 1000),
            data=freeze_json_value(snapshot),
            source_event_seqs=sources,
            surface_op=resolved_op,
        )
        self._accept(event)
        return event

    def _accept(self, event: SessionEvent) -> None:
        if event.seq != len(self._events):
            raise SessionLogError(
                f"事件 seq 必须连续：期望 {len(self._events)}，收到 {event.seq}"
            )
        self._events.append(event)
        if event.surface_op == SURFACE_NONE:
            return
        if event.surface_op == SURFACE_REPLACE:
            for cited in event.source_event_seqs:
                self._shadowed_seqs.add(cited)
            self._surface = [
                SurfaceEntry(entry.seq, entry.message, shadowed=True)
                if entry.seq in self._shadowed_seqs
                else entry
                for entry in self._surface
            ]
        message = project_event_message(event)
        if message is None:
            return
        self._surface.append(SurfaceEntry(seq=event.seq, message=message))

    # -- projections -----------------------------------------------------

    def derive_messages(self) -> list[dict[str, Any]]:
        """Project the model-facing message list from the live surface."""
        return [
            thaw_json_value(entry.message)
            for entry in self._surface
            if not entry.shadowed
        ]

    def derive_transcript(self) -> list[dict[str, Any]]:
        """Project append-origin history, for a human reader.

        A landed replacement shadows history the reader already saw, so the
        transcript deliberately ignores shadowing and reads the raw log.
        """
        return [
            thaw_json_value(entry.message)
            for entry in self._surface
            if not is_replacement_entry(self._events[entry.seq])
        ]

    def derive_timeline(self) -> list[dict[str, Any]]:
        """Project the human chat timeline from the same durable log.

        Provider-only context and tool protocol messages are intentionally
        absent.  Proactive messages are present even though they never enter
        ``derive_messages``.  Read state is folded from later immutable events
        rather than mutating an earlier message in place.
        """

        messages: list[dict[str, Any]] = []
        by_id: dict[str, dict[str, Any]] = {}
        for event in self._events:
            item = project_timeline_message(self.header.session_id, event)
            if item is not None:
                messages.append(item)
                by_id[str(item["id"])] = item
                continue
            if event.type != MESSAGE_READ:
                continue
            message_id = str(event.data.get("message_id") or "").strip()
            target = by_id.get(message_id)
            if target is not None:
                target["read"] = bool(event.data.get("read", True))
                target["read_at"] = int(event.ts_ms)
        return [dict(item) for item in messages]

    def derive_artifacts(self) -> list[dict[str, Any]]:
        """Fold artifact lifecycle events into the current artifact ledger."""

        order: list[str] = []
        ledger: dict[str, dict[str, Any]] = {}
        for event in self._events:
            if event.type not in {ARTIFACT_CREATED, ARTIFACT_VERIFIED, DELIVERY_UPDATED}:
                continue
            data = thaw_json_value(event.data)
            artifact_id = str(data.get("artifact_id") or data.get("path") or "").strip()
            if not artifact_id:
                continue
            if artifact_id not in ledger:
                order.append(artifact_id)
                ledger[artifact_id] = {
                    "artifact_id": artifact_id,
                    "path": str(data.get("path") or ""),
                    "status": "created",
                    "created_at": int(event.ts_ms),
                }
            item = ledger[artifact_id]
            if event.type == ARTIFACT_CREATED:
                item.update({key: value for key, value in data.items() if value is not None})
                item["status"] = str(data.get("status") or item.get("status") or "created")
            elif event.type == ARTIFACT_VERIFIED:
                item["verified"] = bool(data.get("verified", True))
                item["verification"] = str(data.get("verification") or "")
                item["status"] = "verified" if item["verified"] else "verification_failed"
                item["verified_at"] = int(event.ts_ms)
            else:
                item["delivery_status"] = str(data.get("status") or "")
                item["delivery_channel"] = str(data.get("channel") or "")
                item["delivery_detail"] = str(data.get("detail") or "")
                item["delivery_updated_at"] = int(event.ts_ms)
        return [dict(ledger[key]) for key in order]

    def latest_checkpoint(self) -> dict[str, Any]:
        event = self.latest(SESSION_CHECKPOINT)
        return thaw_json_value(event.data) if event is not None else {}

    def derive_metadata(self) -> dict[str, Any]:
        """Fold immutable metadata updates into one current metadata view."""

        metadata: dict[str, Any] = {}
        if self.header.project_id:
            metadata["project_id"] = self.header.project_id
        for event in self.iter_type(SESSION_METADATA_UPDATED):
            data = thaw_json_value(event.data)
            values = data.get("values") if isinstance(data.get("values"), dict) else {}
            removed = data.get("removed") if isinstance(data.get("removed"), list) else []
            metadata.update(values)
            for key in removed:
                metadata.pop(str(key), None)
        return metadata

    def pending_external_events(self) -> list[dict[str, Any]]:
        """Return queued user/timer/channel events not yet consumed by a turn."""

        consumed = {
            str(event.data.get("event_id") or "")
            for event in self.iter_type(EXTERNAL_EVENT_CONSUMED)
        }
        pending: list[dict[str, Any]] = []
        for event in self.iter_type(EXTERNAL_EVENT_QUEUED):
            data = thaw_json_value(event.data)
            event_id = str(data.get("event_id") or "")
            if event_id and event_id not in consumed:
                pending.append({**data, "queued_at": int(event.ts_ms), "seq": event.seq})
        return pending

    @property
    def surface(self) -> tuple[SurfaceEntry, ...]:
        return tuple(self._surface)


def project_event_message(event: SessionEvent) -> dict[str, Any] | None:
    """Canonical per-event projection shared by derivation and reconstruction."""
    data = event.data
    if event.type == USER_MESSAGE:
        content = thaw_json_value(data.get("content"))
        if isinstance(content, list):
            return {"role": "user", "content": content} if content else None
        text = str(content or "")
        return {"role": "user", "content": text} if text.strip() else None
    if event.type == CONTEXT_INJECTED:
        content = str(data.get("content") or "")
        role = str(data.get("role") or "system")
        return {"role": role, "content": content} if content.strip() else None
    if event.type == ASSISTANT_MESSAGE:
        message: dict[str, Any] = {
            "role": "assistant",
            "content": str(data.get("content") or ""),
        }
        tool_calls = thaw_json_value(data.get("tool_calls") or ())
        if tool_calls:
            message["tool_calls"] = tool_calls
        reasoning = str(data.get("reasoning_content") or "")
        if reasoning:
            message["reasoning_content"] = reasoning
        if not message["content"] and not tool_calls:
            return None
        return message
    if event.type == TOOL_RESULT:
        call_id = str(data.get("call_id") or "").strip()
        if not call_id:
            return None
        message = {
            "role": "tool",
            "tool_call_id": call_id,
            "content": str(data.get("content") or ""),
        }
        name = str(data.get("name") or "").strip()
        if name:
            message["name"] = name
        return message
    if event.type == COMPACTION_REPLACEMENT:
        content = str(data.get("content") or "")
        return {"role": "assistant", "content": content} if content.strip() else None
    return None


def project_timeline_message(session_id: str, event: SessionEvent) -> dict[str, Any] | None:
    """Project one event into a human-visible message, when it is one."""

    data = event.data
    role = ""
    content: Any = ""
    channel = str(data.get("channel") or "chat")
    unread = False
    if event.type == USER_MESSAGE:
        role = "user"
        content = thaw_json_value(data.get("content"))
    elif event.type == ASSISTANT_MESSAGE:
        role = "assistant"
        content = str(data.get("content") or "")
    elif event.type == PROACTIVE_MESSAGE:
        role = "assistant"
        content = str(data.get("content") or "")
        channel = str(data.get("channel") or "friday")
        unread = bool(data.get("unread", True))
    else:
        return None
    if isinstance(content, list):
        # The browser timeline stores text plus attachment references, never raw
        # image bytes.  Multimodal provider blocks remain available in the event.
        content = "\n".join(
            str(block.get("text") or "")
            for block in content
            if isinstance(block, Mapping) and block.get("type") == "text"
        )
    text = str(content or "")
    if not text.strip():
        return None
    message_id = str(data.get("message_id") or f"{session_id}:{event.seq}")
    return {
        "id": message_id,
        "event_seq": event.seq,
        "role": role,
        "content": text,
        "channel": channel,
        "createdAt": int(event.ts_ms),
        "read": not unread,
    }


def is_replacement_entry(event: SessionEvent) -> bool:
    return event.surface_op == SURFACE_REPLACE


# ---------------------------------------------------------------------------
# Crash recovery
# ---------------------------------------------------------------------------

def synthesize_interrupted_turn_ends(events: Sequence[SessionEvent]) -> list[SessionEvent]:
    """Close turns whose ``turn/start`` never received a matching end.

    Reload preserves an interrupted turn rather than truncating it. The
    synthesized end carries ``interrupted``, the one reason a running loop never
    writes, so a later reader can tell recovery apart from a real outcome.
    Undispatched tool calls receive ``ABORTED_BEFORE_DISPATCH`` results so the
    reconstructed history stays structurally legal without a repair pass.
    """

    tail = list(events)
    open_turn: SessionEvent | None = None
    open_step: SessionEvent | None = None
    answered: set[str] = set()
    pending_calls: list[SessionEvent] = []
    for event in tail:
        if event.type == TURN_START:
            open_turn = event
            open_step = None
            pending_calls = []
            answered = set()
        elif event.type == TURN_END:
            open_turn = None
            open_step = None
            pending_calls = []
        elif event.type == STEP_START:
            open_step = event
        elif event.type == STEP_END:
            open_step = None
        elif event.type == TOOL_CALL:
            pending_calls.append(event)
        elif event.type == TOOL_RESULT:
            answered.add(str(event.data.get("call_id") or ""))
    if open_turn is None:
        return []

    synthesized: list[SessionEvent] = []
    next_seq = len(tail)
    now_ms = int(time.time() * 1000)
    for call in pending_calls:
        call_id = str(call.data.get("call_id") or "")
        if not call_id or call_id in answered:
            continue
        synthesized.append(
            SessionEvent(
                seq=next_seq,
                type=TOOL_RESULT,
                ts_ms=now_ms,
                data=freeze_json_value(
                    {
                        "call_id": call_id,
                        "name": str(call.data.get("name") or ""),
                        "content": "Error: tool call aborted before dispatch",
                        "error": ABORTED_BEFORE_DISPATCH,
                        "synthesized": True,
                    }
                ),
                source_event_seqs=(call.seq,),
                surface_op=SURFACE_APPEND,
            )
        )
        next_seq += 1
    if open_step is not None:
        synthesized.append(
            SessionEvent(
                seq=next_seq,
                type=STEP_END,
                ts_ms=now_ms,
                data=freeze_json_value(
                    {"step": open_step.data.get("step"), "synthesized": True}
                ),
                source_event_seqs=(open_step.seq,),
                surface_op=SURFACE_NONE,
            )
        )
        next_seq += 1
    synthesized.append(
        SessionEvent(
            seq=next_seq,
            type=TURN_END,
            ts_ms=now_ms,
            data=freeze_json_value(
                {
                    "turn_id": str(open_turn.data.get("turn_id") or ""),
                    "reason": {"kind": TURN_END_INTERRUPTED},
                    "synthesized": True,
                }
            ),
            source_event_seqs=(open_turn.seq,),
            surface_op=SURFACE_NONE,
        )
    )
    return synthesized


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------

def check_session_invariants(log: SessionLog) -> list[str]:
    """Replay structural checks over a log and return every violation found.

    Returning violations instead of raising lets a caller run this over the
    whole existing history during a migration without the first bad record
    hiding the rest.
    """

    problems: list[str] = []
    open_turn: str | None = None
    open_step: int | None = None
    calls_in_step: dict[str, int] = {}
    results_in_step: set[str] = set()

    def close_step(at_seq: int) -> None:
        nonlocal calls_in_step, results_in_step
        for call_id, call_seq in calls_in_step.items():
            if call_id not in results_in_step:
                problems.append(
                    f"seq={at_seq}: tool/call {call_id}（seq={call_seq}）在本 step 内没有配对的 tool/result"
                )
        calls_in_step = {}
        results_in_step = set()

    for index, event in enumerate(log.events):
        if event.seq != index:
            problems.append(f"seq 不连续：位置 {index} 上的事件 seq={event.seq}")
        if event.type == TURN_START:
            if open_turn is not None:
                problems.append(f"seq={event.seq}: turn/start 出现在未关闭的 turn {open_turn} 内")
            open_turn = str(event.data.get("turn_id") or "")
        elif event.type == TURN_END:
            if open_turn is None:
                problems.append(f"seq={event.seq}: turn/end 没有对应的 turn/start")
            if open_step is not None:
                problems.append(f"seq={event.seq}: turn/end 时 step {open_step} 仍未关闭")
                close_step(event.seq)
                open_step = None
            reason = event.data.get("reason")
            kind = str(reason.get("kind") if isinstance(reason, Mapping) else "")
            synthesized = bool(event.data.get("synthesized"))
            if kind == TURN_END_INTERRUPTED and not synthesized:
                problems.append(
                    f"seq={event.seq}: interrupted 只能由崩溃恢复合成，运行中的循环不得写入"
                )
            if kind not in LOOP_EMITTED_TURN_END_REASONS and kind != TURN_END_INTERRUPTED:
                problems.append(f"seq={event.seq}: 未知的 turn/end reason：{kind!r}")
            open_turn = None
        elif event.type == STEP_START:
            if open_turn is None:
                problems.append(f"seq={event.seq}: step/start 出现在任何 turn 之外")
            if open_step is not None:
                problems.append(f"seq={event.seq}: step/start 出现在未关闭的 step {open_step} 内")
            open_step = int(event.data.get("step") or 0)
        elif event.type == STEP_END:
            if open_step is None:
                problems.append(f"seq={event.seq}: step/end 没有对应的 step/start")
            close_step(event.seq)
            open_step = None
        elif event.type == TOOL_CALL:
            call_id = str(event.data.get("call_id") or "")
            if not call_id:
                problems.append(f"seq={event.seq}: tool/call 缺少 call_id")
            else:
                calls_in_step[call_id] = event.seq
        elif event.type == TOOL_RESULT:
            results_in_step.add(str(event.data.get("call_id") or ""))
        if event.surface_op == SURFACE_REPLACE:
            for cited in event.source_event_seqs:
                if cited >= event.seq:
                    problems.append(
                        f"seq={event.seq}: replacement 只能遮蔽更早的事件，却引用了 seq={cited}"
                    )

    if open_turn is not None:
        problems.append(f"日志结束时 turn {open_turn} 仍未关闭；重放前应先合成 interrupted 结束事件")
    return problems


# ---------------------------------------------------------------------------
# Durable value handling
# ---------------------------------------------------------------------------

def snapshot_json_value(value: Any) -> Any:
    """Validate and copy a plain JSON value in one pass.

    Returns ``None`` for input that cannot be persisted, so a caller gets one
    accepted representation instead of a check followed by a second read.
    """

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                return None
            copied = snapshot_json_value(item)
            if copied is None and item is not None:
                return None
            result[key] = copied
        return result
    if isinstance(value, (list, tuple)):
        items: list[Any] = []
        for item in value:
            copied = snapshot_json_value(item)
            if copied is None and item is not None:
                return None
            items.append(copied)
        return items
    return None


def freeze_json_value(value: Any) -> Any:
    """Deep-freeze a snapshotted value so recorded history cannot be edited."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: freeze_json_value(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json_value(item) for item in value)
    return value


def thaw_json_value(value: Any) -> Any:
    """Convert a frozen value back to plain JSON-serializable containers."""
    if isinstance(value, Mapping):
        return {key: thaw_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [thaw_json_value(item) for item in value]
    return value
