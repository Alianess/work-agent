from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable
import json
import sqlite3
import time

from .session_store import SessionStore, sanitize_conversation_id


MEMORY_SCHEMA_VERSION = 1
MAX_MEMORY_CONTENT_CHARS = 24_000


@dataclass(frozen=True)
class MemorySource:
    conversation_id: str
    conversation_title: str
    project_id: str
    summary: str
    summary_message_count: int
    conversation_updated_at: int


class CrossChatMemoryStore:
    """Account-local, user-manageable memories derived from chat summaries."""

    def __init__(self, session_store: SessionStore, database_path: str | Path | None = None) -> None:
        self.session_store = session_store
        self.database_path = Path(
            database_path or session_store.session_dir.parent / "cross_chat_memories.sqlite3"
        ).resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def sync(self) -> int:
        sources = list(iter_memory_sources(self.session_store))
        now = int(time.time())
        changed = 0
        with self._connect() as connection:
            for source in sources:
                source_hash = summary_hash(source.summary)
                memory_id = memory_id_for(source.conversation_id)
                row = connection.execute(
                    "SELECT state, source_hash FROM cross_chat_memories WHERE memory_id = ?",
                    (memory_id,),
                ).fetchone()
                if row is None:
                    connection.execute(
                        """
                        INSERT INTO cross_chat_memories(
                            memory_id, conversation_id, conversation_title, project_id,
                            content, source_summary, source_hash, summary_message_count,
                            conversation_updated_at, state, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'automatic', ?, ?)
                        """,
                        (
                            memory_id,
                            source.conversation_id,
                            source.conversation_title,
                            source.project_id,
                            source.summary,
                            source.summary,
                            source_hash,
                            source.summary_message_count,
                            source.conversation_updated_at,
                            now,
                            now,
                        ),
                    )
                    changed += 1
                    continue
                state = str(row["state"] or "automatic")
                old_hash = str(row["source_hash"] or "")
                connection.execute(
                    """
                    UPDATE cross_chat_memories
                    SET conversation_title = ?, project_id = ?, source_summary = ?,
                        source_hash = ?, summary_message_count = ?,
                        conversation_updated_at = ?,
                        content = CASE WHEN state = 'automatic' THEN ? ELSE content END,
                        updated_at = CASE WHEN source_hash <> ? THEN ? ELSE updated_at END
                    WHERE memory_id = ?
                    """,
                    (
                        source.conversation_title,
                        source.project_id,
                        source.summary,
                        source_hash,
                        source.summary_message_count,
                        source.conversation_updated_at,
                        source.summary,
                        source_hash,
                        now,
                        memory_id,
                    ),
                )
                if old_hash != source_hash or state == "automatic":
                    changed += int(old_hash != source_hash)
        return changed

    def list(
        self,
        *,
        project_id: str | None = None,
        include_deleted: bool = False,
        query: str = "",
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        self.sync()
        conditions: list[str] = []
        parameters: list[Any] = []
        if not include_deleted:
            conditions.append("state <> 'deleted'")
        if project_id is not None:
            conditions.append("project_id = ?")
            parameters.append(str(project_id or ""))
        clean_query = str(query or "").strip()
        if clean_query:
            conditions.append("(content LIKE ? OR conversation_title LIKE ?)")
            like = f"%{clean_query}%"
            parameters.extend([like, like])
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        parameters.append(max(1, min(int(limit), 500)))
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM cross_chat_memories
                {where}
                ORDER BY conversation_updated_at DESC, updated_at DESC
                LIMIT ?
                """,
                parameters,
            ).fetchall()
        return [memory_payload(row) for row in rows]

    def update(self, memory_id: str, content: str) -> dict[str, Any]:
        clean_content = normalize_memory_content(content)
        if not clean_content:
            raise ValueError("记忆内容不能为空。")
        now = int(time.time())
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE cross_chat_memories
                SET content = ?, state = 'corrected', updated_at = ?
                WHERE memory_id = ? AND state <> 'deleted'
                """,
                (clean_content, now, str(memory_id or "").strip()),
            )
            if cursor.rowcount != 1:
                raise ValueError("没有找到这条可纠正的记忆。")
            row = connection.execute(
                "SELECT * FROM cross_chat_memories WHERE memory_id = ?",
                (str(memory_id or "").strip(),),
            ).fetchone()
        assert row is not None
        return memory_payload(row)

    def delete(self, memory_id: str) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE cross_chat_memories
                SET state = 'deleted', updated_at = ?
                WHERE memory_id = ? AND state <> 'deleted'
                """,
                (int(time.time()), str(memory_id or "").strip()),
            )
            if cursor.rowcount != 1:
                raise ValueError("没有找到这条可删除的记忆。")

    def active_for_scope(self, *, scope: str, project_id: str = "") -> list[dict[str, Any]]:
        resolved_project_id: str | None
        if scope == "project":
            resolved_project_id = str(project_id or "")
        elif scope == "account":
            resolved_project_id = ""
        else:
            return []
        return self.list(project_id=resolved_project_id, limit=500)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=8)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA busy_timeout=8000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS cross_chat_memories (
                    memory_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL UNIQUE,
                    conversation_title TEXT NOT NULL DEFAULT '',
                    project_id TEXT NOT NULL DEFAULT '',
                    content TEXT NOT NULL,
                    source_summary TEXT NOT NULL,
                    source_hash TEXT NOT NULL,
                    summary_message_count INTEGER NOT NULL DEFAULT 0,
                    conversation_updated_at INTEGER NOT NULL DEFAULT 0,
                    state TEXT NOT NULL DEFAULT 'automatic'
                        CHECK(state IN ('automatic', 'corrected', 'deleted')),
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS cross_chat_memories_project_idx
                    ON cross_chat_memories(project_id, state, conversation_updated_at DESC);
                """
            )
            connection.execute(f"PRAGMA user_version = {MEMORY_SCHEMA_VERSION}")


def iter_memory_sources(session_store: SessionStore) -> Iterable[MemorySource]:
    archive = load_archive(session_store)
    session_ids = {
        path.stem
        for path in session_store.session_dir.glob("*.json")
        if sanitize_conversation_id(path.stem)
    }
    session_ids.update(archive)
    for conversation_id in sorted(session_ids):
        session = session_store.load(conversation_id)
        item = archive.get(conversation_id, {})
        summary = normalize_memory_content(
            str(session.summary or item.get("contextSummary") or "")
        )
        if not summary:
            continue
        yield MemorySource(
            conversation_id=conversation_id,
            conversation_title=str(
                item.get("title") or session.metadata.get("title") or conversation_id
            ).strip(),
            project_id=str(
                item.get("projectId")
                or item.get("project_id")
                or session.metadata.get("project_id")
                or ""
            ).strip(),
            summary=summary,
            summary_message_count=max(
                0,
                int(session.summary_message_count or item.get("contextSummaryMessageCount") or 0),
            ),
            conversation_updated_at=max(
                int(session.updated_at or 0),
                int(item.get("updatedAt") or item.get("updated_at") or 0),
            ),
        )


def load_archive(session_store: SessionStore) -> dict[str, dict[str, Any]]:
    path = session_store.session_dir.parent / "conversations.json"
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    items = loaded.get("items") if isinstance(loaded, dict) else loaded
    if not isinstance(items, list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        conversation_id = sanitize_conversation_id(item.get("id"))
        if conversation_id:
            result[conversation_id] = item
    return result


def memory_id_for(conversation_id: str) -> str:
    digest = sha256(str(conversation_id).encode("utf-8")).hexdigest()[:16]
    return f"memory-{digest}"


def summary_hash(summary: str) -> str:
    return sha256(str(summary).encode("utf-8")).hexdigest()


def normalize_memory_content(content: str) -> str:
    text = str(content or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    return text[:MAX_MEMORY_CONTENT_CHARS]


def memory_payload(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": str(row["memory_id"]),
        "conversation_id": str(row["conversation_id"]),
        "conversation_title": str(row["conversation_title"]),
        "project_id": str(row["project_id"]),
        "content": str(row["content"]),
        "source_summary": str(row["source_summary"]),
        "summary_message_count": int(row["summary_message_count"]),
        "conversation_updated_at": int(row["conversation_updated_at"]),
        "state": str(row["state"]),
        "created_at": int(row["created_at"]),
        "updated_at": int(row["updated_at"]),
    }
