"""Durable backend for the append-only session event log.

Persistence is deliberately not part of ``SessionLog``: the log commits in
memory and notifies, and this module subscribes to those commits and writes
behind them. That split is what lets a turn keep running while writes batch,
and lets ``flush`` be an explicit ordering and error-observation checkpoint
rather than a hidden cost on every append.

One row per event, and the row maps 1:1 onto the event. There is no second
persisted schema to keep in sync with the event vocabulary.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Iterator, Sequence
import json
import sqlite3
import threading
import time

from .session_log import (
    freeze_json_value,
    snapshot_json_value,
    SessionEvent,
    SessionHeader,
    SessionLog,
    SessionLogError,
    synthesize_interrupted_turn_ends,
)


DEFAULT_BATCH_WINDOW_SECONDS = 0.25


@dataclass(frozen=True)
class SessionLogRevision:
    """Opaque token identifying one storage source and one revision of a log.

    Comparing revisions is cheap and does not require loading the events.
    """

    source: str
    session_id: str
    next_seq: int
    updated_at_ms: int


class SessionLogStore:
    """SQLite-backed append-only event storage."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._ensure_schema()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """Open one connection, commit or roll back, then always close it.

        ``with sqlite3.connect(...)`` is a transaction scope, not a closing
        scope: leaving connections open keeps the WAL and shared-memory files
        alive and leaks a handle per call.
        """

        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            with connection:
                yield connection
        finally:
            connection.close()

    def _ensure_schema(self) -> None:
        with self._lock, self._transaction() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS session_events (
                    session_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    type TEXT NOT NULL,
                    ts_ms INTEGER NOT NULL,
                    data TEXT NOT NULL,
                    source_event_seqs TEXT NOT NULL,
                    surface_op TEXT NOT NULL,
                    PRIMARY KEY (session_id, seq)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS session_headers (
                    session_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    updated_at_ms INTEGER NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS session_events_type_idx"
                " ON session_events(session_id, type, seq)"
            )

    # -- headers ---------------------------------------------------------

    def put_header(self, header: SessionHeader) -> None:
        with self._lock, self._transaction() as connection:
            connection.execute(
                "INSERT INTO session_headers(session_id, payload, updated_at_ms) VALUES(?,?,?)"
                " ON CONFLICT(session_id) DO UPDATE SET payload=excluded.payload,"
                " updated_at_ms=excluded.updated_at_ms",
                (header.session_id, json.dumps(header.to_payload(), ensure_ascii=False), _now_ms()),
            )

    def header(self, session_id: str) -> SessionHeader | None:
        with self._lock, self._transaction() as connection:
            row = connection.execute(
                "SELECT payload FROM session_headers WHERE session_id=?", (session_id,)
            ).fetchone()
        if row is None:
            return None
        payload = json.loads(row["payload"])
        return SessionHeader(
            session_id=str(payload.get("session_id") or session_id),
            version=int(payload.get("version") or 1),
            created_at_ms=int(payload.get("created_at_ms") or 0),
            cwd=str(payload.get("cwd") or ""),
            project_id=str(payload.get("project_id") or ""),
            seed_length=int(payload.get("seed_length") or 0),
            parent_session=str(payload.get("parent_session") or ""),
        )

    # -- events ----------------------------------------------------------

    def next_seq(self, session_id: str) -> int:
        with self._lock, self._transaction() as connection:
            row = connection.execute(
                "SELECT MAX(seq) AS max_seq FROM session_events WHERE session_id=?",
                (session_id,),
            ).fetchone()
        return 0 if row is None or row["max_seq"] is None else int(row["max_seq"]) + 1

    def append(self, session_id: str, events: Sequence[SessionEvent]) -> int:
        """Append a contiguous batch. Returns the new next-seq.

        The first event's seq must equal the stored next-seq. Rejecting a gap
        here is what keeps the log a total order that can be replayed.
        """

        if not events:
            return self.next_seq(session_id)
        with self._lock:
            expected = self.next_seq(session_id)
            if events[0].seq != expected:
                raise SessionLogError(
                    f"追加批次不连续：存储下一个 seq={expected}，批次首个 seq={events[0].seq}"
                )
            for offset, event in enumerate(events):
                if event.seq != expected + offset:
                    raise SessionLogError(
                        f"追加批次内部不连续：位置 {offset} 期望 seq={expected + offset}，实际 {event.seq}"
                    )
            rows = []
            for event in events:
                row = event.to_row()
                rows.append(
                    (
                        session_id,
                        row["seq"],
                        row["type"],
                        row["ts_ms"],
                        row["data"],
                        row["source_event_seqs"],
                        row["surface_op"],
                    )
                )
            with self._transaction() as connection:
                connection.executemany(
                    "INSERT INTO session_events"
                    "(session_id, seq, type, ts_ms, data, source_event_seqs, surface_op)"
                    " VALUES(?,?,?,?,?,?,?)",
                    rows,
                )
                connection.execute(
                    "INSERT INTO session_headers(session_id, payload, updated_at_ms) VALUES(?,?,?)"
                    " ON CONFLICT(session_id) DO UPDATE SET updated_at_ms=excluded.updated_at_ms",
                    (session_id, json.dumps({"session_id": session_id}), _now_ms()),
                )
            return expected + len(events)

    def read(self, session_id: str, *, from_seq: int = 0) -> list[SessionEvent]:
        with self._lock, self._transaction() as connection:
            rows = connection.execute(
                "SELECT seq, type, ts_ms, data, source_event_seqs, surface_op"
                " FROM session_events WHERE session_id=? AND seq>=? ORDER BY seq",
                (session_id, int(from_seq)),
            ).fetchall()
        return [SessionEvent.from_row(dict(row)) for row in rows]

    def revision(self, session_id: str) -> SessionLogRevision:
        with self._lock, self._transaction() as connection:
            row = connection.execute(
                "SELECT MAX(seq) AS max_seq FROM session_events WHERE session_id=?",
                (session_id,),
            ).fetchone()
            header_row = connection.execute(
                "SELECT updated_at_ms FROM session_headers WHERE session_id=?", (session_id,)
            ).fetchone()
        next_seq = 0 if row is None or row["max_seq"] is None else int(row["max_seq"]) + 1
        return SessionLogRevision(
            source=str(self.db_path),
            session_id=session_id,
            next_seq=next_seq,
            updated_at_ms=int(header_row["updated_at_ms"]) if header_row else 0,
        )

    def list_sessions(self) -> list[str]:
        with self._lock, self._transaction() as connection:
            rows = connection.execute(
                "SELECT DISTINCT session_id FROM session_events ORDER BY session_id"
            ).fetchall()
        return [str(row["session_id"]) for row in rows]

    # -- loading ---------------------------------------------------------

    def load(self, session_id: str) -> SessionLog:
        """Materialize a log, closing any turn that never received an end.

        Reload preserves an interrupted turn rather than truncating it, and the
        synthesized closure is durable: a later reader sees the same history
        this call returned.
        """

        header = self.header(session_id) or SessionHeader(session_id=session_id)
        events = self.read(session_id)
        recovered = synthesize_interrupted_turn_ends(events)
        if recovered:
            self.append(session_id, recovered)
            events = events + recovered
        return SessionLog(header, events)


class SessionLogWriter:
    """Write-behind batching for one session's committed events.

    The first pending event starts a fixed batching window; later events join
    it without resetting the deadline, so a burst cannot starve durability.
    A rejected write is retained and automatic retry pauses — an explicit
    ``flush`` retries immediately and is where the failure becomes observable.
    """

    def __init__(
        self,
        store: SessionLogStore,
        session_id: str,
        *,
        window_seconds: float = DEFAULT_BATCH_WINDOW_SECONDS,
    ) -> None:
        self.store = store
        self.session_id = session_id
        self.window_seconds = max(0.0, float(window_seconds))
        self._condition = threading.Condition()
        self._pending: list[SessionEvent] = []
        self._deadline: float | None = None
        self._paused = False
        self._last_error: Exception | None = None
        self._closed = False
        self._thread = threading.Thread(target=self._run, name=f"session-log-{session_id}", daemon=True)
        self._thread.start()

    def on_event(self, event: SessionEvent) -> None:
        with self._condition:
            self._pending.append(event)
            if self._deadline is None:
                self._deadline = time.monotonic() + self.window_seconds
            self._condition.notify_all()

    def flush(self) -> None:
        """Drain pending writes and surface any retained failure."""
        with self._condition:
            self._paused = False
            self._deadline = time.monotonic()
            self._condition.notify_all()
            while self._pending and self._last_error is None and not self._closed:
                self._condition.wait(timeout=5.0)
            error = self._last_error
            if error is not None:
                self._last_error = None
                raise error

    def close(self) -> None:
        try:
            self.flush()
        finally:
            with self._condition:
                self._closed = True
                self._condition.notify_all()
            self._thread.join(timeout=5.0)

    @property
    def pending_count(self) -> int:
        with self._condition:
            return len(self._pending)

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._closed and (
                    not self._pending or self._paused or self._deadline is None
                ):
                    self._condition.wait(timeout=1.0)
                if self._closed and not self._pending:
                    return
                now = time.monotonic()
                if self._deadline is not None and now < self._deadline:
                    self._condition.wait(timeout=self._deadline - now)
                    continue
                batch = list(self._pending)
                if not batch:
                    self._deadline = None
                    continue
            try:
                self.store.append(self.session_id, batch)
            except Exception as error:  # retained; automatic retry pauses
                with self._condition:
                    self._last_error = error
                    self._paused = True
                    self._condition.notify_all()
                    # A closing writer must not spin on a batch that will keep
                    # failing. The retained events and the error stay readable.
                    if self._closed:
                        return
                continue
            with self._condition:
                del self._pending[: len(batch)]
                self._deadline = time.monotonic() + self.window_seconds if self._pending else None
                self._condition.notify_all()


def _now_ms() -> int:
    return int(time.time() * 1000)


def attach_writer(log: SessionLog, writer: SessionLogWriter, *, from_seq: int = 0) -> None:
    """Queue every already-committed event at or after ``from_seq``."""
    for event in log.events:
        if event.seq >= from_seq:
            writer.on_event(event)


def append_and_record(
    log: SessionLog,
    writer: SessionLogWriter | None,
    event_type: str,
    data: Iterable | dict | None = None,
    **kwargs,
) -> SessionEvent:
    """Commit to the log, then hand the committed event to the writer.

    Commit-then-notify keeps in-memory truth ahead of durability by exactly one
    batching window, and never the other way round.
    """

    event = log.append(event_type, data, **kwargs)
    if writer is not None:
        writer.on_event(event)
    return event


class DurableTurnMirror:
    """Mirror one turn's events into the conversation's durable log.

    A turn's runtime is seeded from the prepared request window, so its own
    sequence numbers restart at zero. The durable log is append-only across the
    whole conversation, so mirrored events are re-stamped onto its next-seq and
    any cited source seqs shift by the same offset. Seed events are not
    mirrored — they project history the durable log already holds — except for
    the prompt that opened this turn, which arrives inside the window and would
    otherwise leave the record unable to say what was asked.
    """

    def __init__(
        self,
        store: SessionLogStore,
        conversation_id: str,
        *,
        skip_before_seq: int,
        window_seconds: float = DEFAULT_BATCH_WINDOW_SECONDS,
    ) -> None:
        self.store = store
        self.conversation_id = conversation_id
        self.skip_before_seq = int(skip_before_seq)
        self.window_seconds = window_seconds
        self._prologue: list[tuple[str, dict]] = []
        self._pending: list[SessionEvent] = []
        self._offset: int | None = None
        self._lock = threading.RLock()
        self.last_error: Exception | None = None

    def record_prompt(self, content: str, **extra) -> None:
        """Carry the turn's opening prompt into the durable record."""
        text = str(content or "").strip()
        if not text:
            return
        with self._lock:
            self._prologue.append(("user/message", {"content": text, **extra}))

    def on_event(self, event: SessionEvent) -> None:
        if event.seq < self.skip_before_seq:
            return
        with self._lock:
            self._pending.append(event)

    def flush(self) -> None:
        """Write everything buffered. Never raises into the turn."""
        with self._lock:
            prologue = list(self._prologue)
            batch = list(self._pending)
            self._prologue.clear()
            self._pending.clear()
            if not prologue and not batch:
                return
            base = self.store.next_seq(self.conversation_id)
            events: list[SessionEvent] = []
            for index, (event_type, data) in enumerate(prologue):
                events.append(
                    SessionEvent(
                        seq=base + index,
                        type=event_type,
                        ts_ms=_now_ms(),
                        data=freeze_json_value(snapshot_json_value(data) or {}),
                        surface_op="append",
                    )
                )
            if self._offset is None:
                self._offset = base + len(prologue) - self.skip_before_seq
            offset = self._offset
            for event in batch:
                sources = tuple(
                    seq + offset
                    for seq in event.source_event_seqs
                    if seq >= self.skip_before_seq
                )
                events.append(replace(event, seq=event.seq + offset, source_event_seqs=sources))
        try:
            self.store.append(self.conversation_id, events)
        except Exception as error:
            # Durability is observability here, not the product. A failed
            # mirror must not take down a turn that otherwise succeeded.
            self.last_error = error
