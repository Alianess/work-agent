"""Account-scoped persistent memory: facts, preferences, and a derived profile.

This deliberately differs from a conversation summary.  A summary is working
memory for one long chat; records here are small, source-backed items that can
be retrieved before a later chat starts.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any
import json
import re
import sqlite3
import time
import uuid

from .session_store import SessionStore


MEMORY_SCHEMA_VERSION = 2
MAX_MEMORY_CONTENT_CHARS = 1_200
MAX_PROFILE_CHARS = 8_000
MEMORY_KINDS = {"preference", "identity", "goal", "project", "fact"}
MEMORY_STATES = {"automatic", "explicit", "corrected", "deleted"}


class CrossChatMemoryStore:
    """Account-local memory records, separate from raw chats and summaries."""

    def __init__(self, session_store: SessionStore, database_path: str | Path | None = None) -> None:
        self.session_store = session_store
        self.database_path = Path(
            database_path or session_store.session_dir.parent / "cross_chat_memories.sqlite3"
        ).resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def list(
        self,
        *,
        project_id: str | None = None,
        include_deleted: bool = False,
        query: str = "",
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        conditions: list[str] = []
        parameters: list[Any] = []
        if not include_deleted:
            conditions.append("state <> 'deleted'")
        if project_id is not None:
            conditions.append("project_id = ?")
            parameters.append(str(project_id or ""))
        if str(query or "").strip():
            conditions.append("(content LIKE ? OR kind LIKE ?)")
            like = f"%{str(query).strip()}%"
            parameters.extend([like, like])
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        parameters.append(max(1, min(int(limit), 500)))
        with self._connect() as connection:
            rows = connection.execute(
                f"""SELECT * FROM memory_items {where}
                ORDER BY updated_at DESC, created_at DESC LIMIT ?""",
                parameters,
            ).fetchall()
        return [memory_payload(row) for row in rows]

    def relevant_for_scope(self, *, query: str, scope: str, project_id: str = "", limit: int = 8) -> list[dict[str, Any]]:
        selected_project = str(project_id or "") if scope == "project" else ""
        items = self.list(project_id=selected_project, limit=500)
        terms = query_terms(query)
        ranked = sorted(
            items,
            key=lambda item: (memory_relevance(item["content"], terms), item["updated_at"]),
            reverse=True,
        )
        return [item for item in ranked if memory_relevance(item["content"], terms) > 0][: max(1, min(limit, 12))]

    def profile_for_scope(self, *, scope: str, project_id: str = "") -> dict[str, Any] | None:
        selected_project = str(project_id or "") if scope == "project" else ""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM memory_profiles WHERE project_id = ?", (selected_project,)
            ).fetchone()
        return profile_payload(row) if row else None

    def active_for_scope(self, *, scope: str, project_id: str = "") -> list[dict[str, Any]]:
        """Compatibility helper for the raw-history recall tool."""
        if scope == "project":
            return self.list(project_id=str(project_id or ""), limit=500)
        if scope == "account":
            return self.list(project_id="", limit=500)
        return []

    def upsert_many(
        self,
        records: list[dict[str, Any]],
        *,
        conversation_id: str,
        conversation_title: str,
        project_id: str = "",
        source_excerpt: str = "",
        state: str = "automatic",
    ) -> list[dict[str, Any]]:
        if state not in MEMORY_STATES - {"deleted"}:
            raise ValueError("无效的记忆状态。")
        now = int(time.time())
        saved: list[dict[str, Any]] = []
        with self._connect() as connection:
            for raw in records[:30]:
                content = normalize_memory_content(str(raw.get("content") or ""))
                kind = str(raw.get("kind") or "fact").strip().lower()
                if not content or kind not in MEMORY_KINDS:
                    continue
                fingerprint = memory_fingerprint(kind, content, project_id)
                existing = connection.execute(
                    "SELECT * FROM memory_items WHERE fingerprint = ?", (fingerprint,)
                ).fetchone()
                if existing and str(existing["state"]) in {"explicit", "corrected"}:
                    saved.append(memory_payload(existing))
                    continue
                if existing:
                    connection.execute(
                        """UPDATE memory_items SET content=?, conversation_id=?, conversation_title=?,
                        project_id=?, source_excerpt=?, kind=?, state=?, updated_at=? WHERE id=?""",
                        (content, conversation_id, conversation_title, project_id, source_excerpt[:1600], kind, state, now, existing["id"]),
                    )
                    row = connection.execute("SELECT * FROM memory_items WHERE id = ?", (existing["id"],)).fetchone()
                else:
                    memory_id = f"memory-{uuid.uuid4().hex[:16]}"
                    connection.execute(
                        """INSERT INTO memory_items(id, fingerprint, kind, content, conversation_id,
                        conversation_title, project_id, source_excerpt, state, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (memory_id, fingerprint, kind, content, conversation_id, conversation_title, project_id, source_excerpt[:1600], state, now, now),
                    )
                    row = connection.execute("SELECT * FROM memory_items WHERE id = ?", (memory_id,)).fetchone()
                if row:
                    saved.append(memory_payload(row))
        return saved

    def set_profile(self, content: str, *, project_id: str = "") -> dict[str, Any] | None:
        clean = normalize_profile(content)
        if not clean:
            return None
        now = int(time.time())
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO memory_profiles(project_id, content, updated_at)
                VALUES (?, ?, ?) ON CONFLICT(project_id) DO UPDATE SET content=excluded.content, updated_at=excluded.updated_at""",
                (str(project_id or ""), clean, now),
            )
            row = connection.execute("SELECT * FROM memory_profiles WHERE project_id = ?", (str(project_id or ""),)).fetchone()
        return profile_payload(row) if row else None

    def update(self, memory_id: str, content: str) -> dict[str, Any]:
        clean = normalize_memory_content(content)
        if not clean:
            raise ValueError("记忆内容不能为空。")
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE memory_items SET content=?, state='corrected', updated_at=? WHERE id=? AND state <> 'deleted'",
                (clean, int(time.time()), str(memory_id or "").strip()),
            )
            if cursor.rowcount != 1:
                raise ValueError("没有找到这条可纠正的记忆。")
            row = connection.execute("SELECT * FROM memory_items WHERE id=?", (str(memory_id).strip(),)).fetchone()
        assert row is not None
        return memory_payload(row)

    def delete(self, memory_id: str) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE memory_items SET state='deleted', updated_at=? WHERE id=? AND state <> 'deleted'",
                (int(time.time()), str(memory_id or "").strip()),
            )
            if cursor.rowcount != 1:
                raise ValueError("没有找到这条可删除的记忆。")

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
                CREATE TABLE IF NOT EXISTS memory_items (
                    id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL UNIQUE, kind TEXT NOT NULL,
                    content TEXT NOT NULL, conversation_id TEXT NOT NULL, conversation_title TEXT NOT NULL,
                    project_id TEXT NOT NULL DEFAULT '', source_excerpt TEXT NOT NULL DEFAULT '',
                    state TEXT NOT NULL CHECK(state IN ('automatic','explicit','corrected','deleted')),
                    created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS memory_items_scope_idx ON memory_items(project_id, state, updated_at DESC);
                CREATE TABLE IF NOT EXISTS memory_profiles (
                    project_id TEXT PRIMARY KEY, content TEXT NOT NULL, updated_at INTEGER NOT NULL
                );
                """
            )
            connection.execute(f"PRAGMA user_version = {MEMORY_SCHEMA_VERSION}")


def normalize_memory_content(content: str) -> str:
    return re.sub(r"\s+", " ", str(content or "").strip())[:MAX_MEMORY_CONTENT_CHARS]


def normalize_profile(content: str) -> str:
    return str(content or "").strip()[:MAX_PROFILE_CHARS]


def memory_fingerprint(kind: str, content: str, project_id: str) -> str:
    return sha256(f"{project_id}\n{kind}\n{content.casefold()}".encode("utf-8")).hexdigest()


def query_terms(value: str) -> set[str]:
    compact = re.sub(r"\s+", "", str(value or "").casefold())
    terms = {compact[index:index + 2] for index in range(max(0, len(compact) - 1))}
    terms.update(re.findall(r"[a-z0-9_.+-]{2,}", str(value or "").casefold()))
    return terms


def memory_relevance(content: str, terms: set[str]) -> int:
    if not terms:
        return 0
    haystack = query_terms(content)
    return len(terms & haystack)


def memory_payload(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": str(row["id"]), "kind": str(row["kind"]), "content": str(row["content"]),
        "conversation_id": str(row["conversation_id"]), "conversation_title": str(row["conversation_title"]),
        "project_id": str(row["project_id"]), "source_excerpt": str(row["source_excerpt"]),
        "state": str(row["state"]), "created_at": int(row["created_at"]), "updated_at": int(row["updated_at"]),
    }


def profile_payload(row: sqlite3.Row) -> dict[str, Any]:
    return {"project_id": str(row["project_id"]), "content": str(row["content"]), "updated_at": int(row["updated_at"])}
