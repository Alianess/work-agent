"""Event-log backed conversation repository.

The JSON ``SessionStore`` predates the append-only log.  It remains useful as
an inspectable compatibility snapshot, but it must not decide what survived a
restart.  This repository gives the application one boundary for:

* one-time migration of legacy JSON history into the log;
* rebuilding the bounded working session from an in-log checkpoint; and
* publishing a new checkpoint after the turn facts are durable.

The checkpoint is a derived cache *inside the same log*.  Raw turn facts stay
immutable while compaction can keep a smaller provider working set.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .session_log import (
    COMPACTION_REPLACEMENT,
    SESSION_CHECKPOINT,
    SessionLog,
    project_event_message,
    thaw_json_value,
)
from .session_log_store import SessionLogStore
from .session_migration import build_seed_log
from .session_runtime import ConversationRuntime
from .session_store import ConversationSession, SessionStore, sanitize_runtime_message


CHECKPOINT_SCHEMA_VERSION = 1


class ConversationRepository:
    """Load and checkpoint conversations with the event log as authority."""

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        session_store: SessionStore,
        log_store: SessionLogStore,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.session_store = session_store
        self.log_store = log_store

    def load(
        self,
        conversation_id: str,
        *,
        display_messages: Sequence[dict[str, Any]] = (),
        exclude_last_user: bool = True,
        recover_interrupted: bool = True,
    ) -> ConversationSession:
        loaded = self.session_store.load(conversation_id)
        legacy = loaded if isinstance(loaded, ConversationSession) else ConversationSession(
            id=conversation_id,
            messages=list(getattr(loaded, "messages", []) or []),
            summary=str(getattr(loaded, "summary", "") or ""),
            summary_message_count=max(
                0,
                int(getattr(loaded, "summary_message_count", 0) or 0),
            ),
            recall_episodes=list(getattr(loaded, "recall_episodes", []) or []),
            compaction_events=list(getattr(loaded, "compaction_events", []) or []),
            metadata=dict(getattr(loaded, "metadata", {}) or {}),
        )
        self._ensure_log(
            legacy,
            display_messages=display_messages,
            exclude_last_user=exclude_last_user,
        )
        log = self.log_store.load(
            conversation_id,
            recover_interrupted=recover_interrupted,
        )
        return self._project_session(log, legacy)

    def checkpoint(self, session: ConversationSession) -> ConversationSession:
        """Append the current bounded working set, then refresh legacy JSON."""

        log = self.log_store.load_live(session.id)
        runtime = ConversationRuntime(log)
        current_metadata = log.derive_metadata()
        removed = [key for key in current_metadata if key not in session.metadata]
        if session.metadata != current_metadata or removed:
            runtime.record_metadata_update(session.metadata, removed=removed)
        runtime.record_checkpoint(self._checkpoint_payload(session))
        pending = log.events[self.log_store.next_seq(session.id) :]
        if pending:
            self.log_store.append(session.id, pending)
        # Compatibility only: every field can be rebuilt from the log and its
        # latest checkpoint.  No caller should load this JSON directly to make
        # recovery decisions.
        self.session_store.save(session)
        return session

    def timeline(self, conversation_id: str) -> list[dict[str, Any]]:
        if not self.log_store.next_seq(conversation_id):
            return []
        return self.log_store.load_live(conversation_id).derive_timeline()

    def _ensure_log(
        self,
        legacy: ConversationSession,
        *,
        display_messages: Sequence[dict[str, Any]],
        exclude_last_user: bool,
    ) -> None:
        if self.log_store.next_seq(legacy.id):
            # Logs created during the earlier mirror phase may not have a
            # checkpoint yet.  Land one once, using the last complete legacy
            # projection, so future loads no longer depend on that file.
            existing = self.log_store.load_live(legacy.id)
            if existing.latest(SESSION_CHECKPOINT) is None:
                if not legacy.messages:
                    legacy.messages = existing.derive_transcript()
                self._append_initial_checkpoint(existing, legacy)
            return

        if not legacy.messages and display_messages:
            self.session_store.bootstrap_from_display_messages(
                legacy,
                list(display_messages),
                exclude_last_user=exclude_last_user,
            )
        log = build_seed_log(
            legacy.id,
            legacy.messages,
            project_id=str(legacy.metadata.get("project_id") or ""),
            cwd=str(self.workspace_root),
        )
        self._append_initial_checkpoint(log, legacy, persist=False)
        self.log_store.put_header(log.header)
        self.log_store.append(legacy.id, log.events)

    def _append_initial_checkpoint(
        self,
        log: SessionLog,
        legacy: ConversationSession,
        *,
        persist: bool = True,
    ) -> None:
        start_seq = log.seq
        runtime = ConversationRuntime(log)
        if legacy.metadata:
            runtime.record_metadata_update(legacy.metadata)
        runtime.record_checkpoint(self._checkpoint_payload(legacy))
        if persist:
            self.log_store.append(legacy.id, log.events[start_seq:])

    def _project_session(
        self,
        log: SessionLog,
        legacy: ConversationSession,
    ) -> ConversationSession:
        checkpoint_event = log.latest(SESSION_CHECKPOINT)
        checkpoint = (
            thaw_json_value(checkpoint_event.data)
            if checkpoint_event is not None
            else {}
        )
        working = checkpoint.get("working_messages")
        if isinstance(working, list):
            messages = [
                message
                for message in (sanitize_runtime_message(item) for item in working)
                if message
            ]
            for event in log.events[(checkpoint_event.seq + 1) if checkpoint_event else 0 :]:
                if event.type == COMPACTION_REPLACEMENT:
                    continue
                message = project_event_message(event)
                cleaned = sanitize_runtime_message(message) if message is not None else {}
                if cleaned:
                    messages.append(cleaned)
        else:
            messages = log.derive_transcript()

        return ConversationSession(
            id=log.header.session_id,
            messages=messages,
            summary=str(checkpoint.get("summary") or ""),
            summary_message_count=max(0, int(checkpoint.get("summary_message_count") or 0)),
            recall_episodes=[
                item for item in (checkpoint.get("recall_episodes") or []) if isinstance(item, dict)
            ],
            compaction_events=[
                item for item in (checkpoint.get("compaction_events") or []) if isinstance(item, dict)
            ],
            created_at=legacy.created_at,
            updated_at=legacy.updated_at,
            metadata=log.derive_metadata(),
        )

    @staticmethod
    def _checkpoint_payload(session: ConversationSession) -> dict[str, Any]:
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "working_messages": list(session.messages),
            "summary": str(session.summary or ""),
            "summary_message_count": max(0, int(session.summary_message_count or 0)),
            "recall_episodes": list(session.recall_episodes),
            "compaction_events": list(session.compaction_events),
        }
