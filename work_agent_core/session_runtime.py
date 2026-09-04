"""The conversation object the agent loop mutates.

Replaces the old pattern of keeping two parallel message lists in lockstep.
There is one durable log; the provider message list is derived from it on every
step, and per-request assembly is recorded so a later reader can reconstruct
exactly what the model received.

Surface events are history: append-only turn runtime contexts, user turns,
assistant replies, tool results, and compaction replacements. The fixed system
prompt and genuinely request-only compatibility blocks belong in
``request/header``; a timestamp or other turn context must never be re-rendered
over an earlier position in the model-visible timeline.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence
import hashlib
import json
import threading

from .session_log import (
    AGENT_ERROR,
    APPROVAL_REQUESTED,
    APPROVAL_RESOLVED,
    ARTIFACT_CREATED,
    ARTIFACT_VERIFIED,
    ASSISTANT_MESSAGE,
    COMPACTION_REPLACEMENT,
    CONTEXT_INJECTED,
    DELIVERY_UPDATED,
    EXTERNAL_EVENT_CONSUMED,
    EXTERNAL_EVENT_QUEUED,
    MESSAGE_READ,
    PLAN_UPDATED,
    PROACTIVE_MESSAGE,
    REQUEST_HEADER,
    SESSION_CHECKPOINT,
    SESSION_METADATA_UPDATED,
    SessionEvent,
    SessionLog,
    STEP_END,
    STEP_START,
    TOOL_CALL,
    TOOL_RESULT,
    TURN_END,
    TURN_END_ABORTED,
    TURN_END_COMPLETED,
    TURN_END_FAILED,
    TURN_START,
    USER_MESSAGE,
)
from .session_log_store import SessionLogWriter


class ConversationRuntime:
    """One conversation's live log plus the request assembly built from it."""

    def __init__(self, log: SessionLog, writer: SessionLogWriter | None = None) -> None:
        self.log = log
        self.writer = writer
        self._open_turn_id = ""
        self._open_step: int | None = None

    @classmethod
    def from_messages(
        cls,
        messages: Sequence[Mapping[str, Any]],
        *,
        session_id: str = "ephemeral",
    ) -> "ConversationRuntime":
        """Build a runtime seeded from a flat message list.

        Used by callers that still hold plain history — tests, one-shot runs,
        and the migration path — so they exercise the same code as a live
        conversation instead of a parallel shortcut.
        """

        from .session_migration import build_seed_log

        return cls(build_seed_log(session_id, list(messages)))

    # -- committing ------------------------------------------------------

    def _append(self, event_type: str, data: Mapping[str, Any], **kwargs: Any) -> SessionEvent:
        event = self.log.append(event_type, dict(data), **kwargs)
        if self.writer is not None:
            self.writer.on_event(event)
        return event

    def flush(self) -> None:
        if self.writer is not None:
            self.writer.flush()

    # -- lifecycle -------------------------------------------------------

    def begin_turn(self, turn_id: str, **meta: Any) -> SessionEvent:
        self._open_turn_id = str(turn_id)
        return self._append(TURN_START, {"turn_id": self._open_turn_id, **meta})

    def resume_turn(self, turn_id: str, *, open_step: int | None = None) -> None:
        """Attach a live runtime to a deliberately parked durable turn.

        Approval is a pause, not a second user turn.  No event is appended
        here: the original ``turn/start`` (and optional ``step/start``) is
        already durable and must remain the sole opening boundary.
        """

        self._open_turn_id = str(turn_id)
        self._open_step = int(open_step) if open_step is not None else None

    def end_turn(self, kind: str = TURN_END_COMPLETED, detail: str = "") -> SessionEvent | None:
        """Close the open turn, closing a dangling step first.

        Returns ``None`` when no turn is open so a second close on an error
        path cannot corrupt the enclosure.
        """

        if not self._open_turn_id:
            return None
        if self._open_step is not None:
            self.end_step(self._open_step)
        reason: dict[str, Any] = {"kind": kind}
        if detail:
            reason["detail"] = detail
        event = self._append(TURN_END, {"turn_id": self._open_turn_id, "reason": reason})
        self._open_turn_id = ""
        return event

    def begin_step(self, step: int) -> SessionEvent:
        self._open_step = int(step)
        return self._append(STEP_START, {"step": self._open_step})

    def end_step(self, step: int | None = None) -> SessionEvent | None:
        if self._open_step is None:
            return None
        event = self._append(STEP_END, {"step": int(step if step is not None else self._open_step)})
        self._open_step = None
        return event

    @property
    def turn_is_open(self) -> bool:
        return bool(self._open_turn_id)

    # -- history ---------------------------------------------------------

    def record_user(self, content: Any, *, source: str = "human", **extra: Any) -> SessionEvent:
        """Record text or an OpenAI-compatible multimodal user payload.

        Tool-loaded images are injected for one model step as ``content``
        blocks.  Coercing that list with ``str(...)`` destroys the native
        image semantics and turns the Base64 payload into a fake human chat
        message.  The event log already snapshots JSON values, so preserve the
        blocks here and let the web persistence boundary dehydrate them.
        """

        normalized = content if isinstance(content, list) else str(content or "")
        return self._append(USER_MESSAGE, {"content": normalized, "source": source, **extra})

    def record_assistant(
        self,
        content: str,
        *,
        tool_calls: Sequence[Mapping[str, Any]] = (),
        reasoning_content: str = "",
        finish_reason: str = "",
        usage: Mapping[str, Any] | None = None,
        chunk_seqs: Sequence[int] = (),
    ) -> SessionEvent:
        return self._append(
            ASSISTANT_MESSAGE,
            {
                "content": str(content or ""),
                "tool_calls": [dict(call) for call in tool_calls],
                "reasoning_content": str(reasoning_content or ""),
                "finish_reason": str(finish_reason or ""),
                "usage": dict(usage or {}),
            },
            source_event_seqs=tuple(chunk_seqs),
        )

    def record_tool_call(self, call_id: str, name: str, arguments: Any) -> SessionEvent:
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        return self._append(
            TOOL_CALL, {"call_id": str(call_id), "name": str(name), "arguments": arguments}
        )

    def record_tool_result(
        self, call_id: str, name: str, content: str, *, error: str = ""
    ) -> SessionEvent:
        data: dict[str, Any] = {
            "call_id": str(call_id),
            "name": str(name),
            "content": str(content or ""),
        }
        if error:
            data["error"] = error
        return self._append(TOOL_RESULT, data)

    def append_message(self, message: Mapping[str, Any]) -> SessionEvent | None:
        """Record one provider-shaped message as the events it stands for.

        An assistant message carrying ``tool_calls`` also emits one ``tool/call``
        per call, so the dispatch is durable in its own right and a result can
        be paired with it even if the process dies before the tool runs.
        """

        if not isinstance(message, Mapping):
            # Rejecting at the boundary is the point: history that cannot be
            # projected must never enter the log in the first place.
            return None
        role = str(message.get("role") or "")
        if role == "user":
            return self.record_user(message.get("content"))
        if role == "assistant":
            tool_calls = list(message.get("tool_calls") or ())
            event = self.record_assistant(
                str(message.get("content") or ""),
                tool_calls=tool_calls,
                reasoning_content=str(message.get("reasoning_content") or ""),
            )
            for call in tool_calls:
                function = call.get("function") if isinstance(call.get("function"), Mapping) else {}
                self.record_tool_call(
                    str(call.get("id") or ""),
                    str(function.get("name") or ""),
                    function.get("arguments") or "{}",
                )
            return event
        if role == "tool":
            return self.record_tool_result(
                str(message.get("tool_call_id") or ""),
                str(message.get("name") or ""),
                str(message.get("content") or ""),
            )
        if role == "system":
            return self.record_context_injection(str(message.get("content") or ""), kind="system")
        return None

    def record_context_injection(
        self, content: str, *, kind: str = "context", role: str = "system"
    ) -> SessionEvent:
        """Record context the model will see that is not conversation history.

        Repair prompts and injected notes have to survive into later steps, so
        they are durable surface entries rather than a scratch list that the
        next derivation would silently drop.
        """

        return self._append(
            CONTEXT_INJECTED, {"kind": kind, "role": role, "content": str(content or "")}
        )

    def record_compaction(self, replaced_seqs: Sequence[int], content: str) -> SessionEvent:
        return self._append(
            COMPACTION_REPLACEMENT,
            {"content": str(content or "")},
            source_event_seqs=tuple(int(item) for item in replaced_seqs),
        )

    # -- log-only --------------------------------------------------------

    def record_request_header(
        self,
        *,
        system_prompt: str,
        late_blocks: Sequence[Mapping[str, Any]],
        tool_names: Sequence[str],
        profile: str,
        model: str,
        endpoint: str = "",
        step: int = 0,
        reason: str = "initial",
        **params: Any,
    ) -> SessionEvent:
        """Snapshot everything in the request except derived history.

        History is already the log, so recording it again would double the
        size for no reconstruction value. Everything else — prompt text, the
        late blocks, which tools were visible, the route and its parameters —
        exists only at request time and is lost unless recorded here.
        """

        return self._append(
            REQUEST_HEADER,
            {
                "reason": reason,
                "step": int(step),
                "system_prompt": system_prompt,
                "system_prompt_sha1": _sha1(system_prompt),
                "late_blocks": [dict(block) for block in late_blocks],
                "tool_names": list(tool_names),
                "profile": profile,
                "model": model,
                "endpoint": endpoint,
                "params": {key: value for key, value in params.items() if value is not None},
            },
        )

    def record_plan(self, plan: Any, explanation: str = "") -> SessionEvent:
        return self._append(PLAN_UPDATED, {"plan": plan, "explanation": str(explanation or "")})

    def record_approval_requested(self, action_id: str, payload: Mapping[str, Any]) -> SessionEvent:
        return self._append(APPROVAL_REQUESTED, {"action_id": str(action_id), **dict(payload)})

    def record_approval_resolved(self, action_id: str, *, granted: bool, source: str = "") -> SessionEvent:
        return self._append(
            APPROVAL_RESOLVED,
            {"action_id": str(action_id), "granted": bool(granted), "source": str(source)},
        )

    def record_error(self, phase: str, error: BaseException, *, traceback_lines: Sequence[str] = ()) -> SessionEvent:
        return self._append(
            AGENT_ERROR,
            {
                "phase": str(phase),
                "type": type(error).__name__,
                "message": str(error),
                "traceback": list(traceback_lines)[-16:],
            },
        )

    # -- human timeline and product state --------------------------------

    def record_proactive_message(
        self,
        content: str,
        *,
        message_id: str,
        channel: str = "friday",
        unread: bool = True,
        source: str = "assistant",
    ) -> SessionEvent:
        return self._append(
            PROACTIVE_MESSAGE,
            {
                "message_id": str(message_id),
                "content": str(content or ""),
                "channel": str(channel or "friday"),
                "unread": bool(unread),
                "source": str(source or "assistant"),
            },
        )

    def record_message_read(self, message_id: str, *, read: bool = True) -> SessionEvent:
        return self._append(
            MESSAGE_READ,
            {"message_id": str(message_id), "read": bool(read)},
        )

    def record_artifact(
        self,
        path: str,
        *,
        artifact_id: str = "",
        kind: str = "file",
        title: str = "",
        origin: str = "tool",
        tool_name: str = "",
        turn_id: str = "",
        status: str = "created",
        **extra: Any,
    ) -> SessionEvent:
        resolved_id = str(artifact_id or path).strip()
        return self._append(
            ARTIFACT_CREATED,
            {
                "artifact_id": resolved_id,
                "path": str(path),
                "kind": str(kind or "file"),
                "title": str(title or ""),
                "origin": str(origin or "tool"),
                "tool_name": str(tool_name or ""),
                "turn_id": str(turn_id or self._open_turn_id),
                "status": str(status or "created"),
                **extra,
            },
        )

    def record_artifact_verification(
        self,
        artifact_id: str,
        *,
        verified: bool,
        verification: str = "",
    ) -> SessionEvent:
        return self._append(
            ARTIFACT_VERIFIED,
            {
                "artifact_id": str(artifact_id),
                "verified": bool(verified),
                "verification": str(verification or ""),
            },
        )

    def record_delivery(
        self,
        artifact_id: str,
        *,
        status: str,
        channel: str = "chat",
        detail: str = "",
    ) -> SessionEvent:
        return self._append(
            DELIVERY_UPDATED,
            {
                "artifact_id": str(artifact_id),
                "status": str(status),
                "channel": str(channel or "chat"),
                "detail": str(detail or ""),
            },
        )

    def record_checkpoint(self, checkpoint: Mapping[str, Any]) -> SessionEvent:
        return self._append(SESSION_CHECKPOINT, dict(checkpoint))

    def record_metadata_update(
        self,
        values: Mapping[str, Any] | None = None,
        *,
        removed: Sequence[str] = (),
    ) -> SessionEvent:
        return self._append(
            SESSION_METADATA_UPDATED,
            {
                "values": dict(values or {}),
                "removed": [str(key) for key in removed],
            },
        )

    def queue_external_event(
        self,
        event_id: str,
        *,
        kind: str,
        payload: Mapping[str, Any],
        priority: str = "normal",
        source: str = "external",
    ) -> SessionEvent:
        return self._append(
            EXTERNAL_EVENT_QUEUED,
            {
                "event_id": str(event_id),
                "kind": str(kind),
                "payload": dict(payload),
                "priority": str(priority or "normal"),
                "source": str(source or "external"),
            },
        )

    def consume_external_event(self, event_id: str, *, turn_id: str = "") -> SessionEvent:
        return self._append(
            EXTERNAL_EVENT_CONSUMED,
            {
                "event_id": str(event_id),
                "turn_id": str(turn_id or self._open_turn_id),
            },
        )

    # -- request assembly ------------------------------------------------

    def build_request_messages(
        self,
        system_prompt: str,
        late_blocks: Sequence[Mapping[str, Any]] = (),
    ) -> list[dict[str, Any]]:
        """Assemble the exact provider message list for one step.

        Legacy late blocks sit immediately before the newest user message.
        Normal per-turn context is already an append-only surface event beside
        its user message, so later requests replay it byte-for-byte.
        """

        derived = self.log.derive_messages()
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        latest_user_index = next(
            (
                index
                for index in range(len(derived) - 1, -1, -1)
                if derived[index].get("role") == "user"
            ),
            len(derived),
        )
        messages.extend(derived[:latest_user_index])
        messages.extend(dict(block) for block in late_blocks)
        messages.extend(derived[latest_user_index:])
        return messages

    # -- compaction target ------------------------------------------------

    def active_turn_surface_seqs(self) -> list[int]:
        """Seqs of the live surface entries after the newest user message.

        Compaction folds the implementation path of the turn in progress; the
        prompt that started it stays, so the model keeps the actual request.
        """

        visible = [entry for entry in self.log.surface if not entry.shadowed]
        last_user_position = None
        for position, entry in enumerate(visible):
            if self.log.event(entry.seq).type == USER_MESSAGE:
                last_user_position = position
        if last_user_position is None:
            return []
        return [entry.seq for entry in visible[last_user_position + 1 :]]


class SessionRegistry:
    """Holds the live runtime for each conversation currently in memory.

    Loading from storage synthesizes an interrupted end for any turn left open,
    which is right after a crash and wrong while a turn is legitimately parked —
    an approval wait keeps its turn open across HTTP requests on purpose. A
    conversation that is live in this registry is served from memory, so only a
    genuinely absent process reaches the recovery path.
    """

    def __init__(self, store: Any) -> None:
        self.store = store
        self._live: dict[str, ConversationRuntime] = {}
        self._lock = threading.RLock()

    def acquire(self, conversation_id: str, *, writer_factory: Any = None) -> ConversationRuntime:
        with self._lock:
            existing = self._live.get(conversation_id)
            if existing is not None:
                return existing
            log = self.store.load(conversation_id)
            writer = writer_factory(conversation_id) if writer_factory else None
            runtime = ConversationRuntime(log, writer)
            self._live[conversation_id] = runtime
            return runtime

    def peek(self, conversation_id: str) -> ConversationRuntime | None:
        with self._lock:
            return self._live.get(conversation_id)

    def release(self, conversation_id: str) -> None:
        with self._lock:
            runtime = self._live.pop(conversation_id, None)
        if runtime is not None:
            runtime.flush()

    def flush_all(self) -> None:
        with self._lock:
            runtimes = list(self._live.values())
        for runtime in runtimes:
            runtime.flush()


def _sha1(text: str) -> str:
    return hashlib.sha1(str(text or "").encode("utf-8")).hexdigest()


TURN_END_KINDS = {
    "completed": TURN_END_COMPLETED,
    "aborted": TURN_END_ABORTED,
    "failed": TURN_END_FAILED,
}
