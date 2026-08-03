"""Account-scoped persistent memory: facts, preferences, and a derived profile.

This deliberately differs from a conversation summary.  A summary is working
memory for one long chat; records here are small, source-backed items that can
be retrieved before a later chat starts.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any
from difflib import SequenceMatcher
import json
import re
import sqlite3
import time
import uuid

from .session_store import SessionStore


MEMORY_SCHEMA_VERSION = 3
MAX_MEMORY_CONTENT_CHARS = 280
MAX_PROFILE_CHARS = 1_200
MAX_ACCOUNT_CORE_MEMORIES = 32
MAX_PROJECT_CORE_MEMORIES = 24
MAX_ACCOUNT_AUTOMATIC_MEMORIES = 12
MAX_PROJECT_AUTOMATIC_MEMORIES = 10
AUTOMATIC_MEMORY_MIN_IMPORTANCE = 0.90
AUTOMATIC_MEMORY_MIN_CONFIDENCE = 0.90
MEMORY_KINDS = {"preference", "identity", "goal", "project", "fact"}
MEMORY_STATES = {"automatic", "explicit", "corrected", "deleted"}
ACCOUNT_AUTOMATIC_KINDS = {"preference", "identity", "goal"}
PROJECT_AUTOMATIC_KINDS = {"preference", "goal", "project"}
TRANSIENT_MEMORY_MARKERS = {
    "当前", "正在", "已完成", "下一步", "本轮", "本次", "这次", "今天", "明天",
    "本周", "初稿", "转写", "文件路径", "归档目录", "输出目录", "已经生成",
}
FILE_DETAIL_PATTERN = re.compile(
    r"(?:meet_files|work_agent_core|web_frontend|/Users/|\\\\Users\\\\|"
    r"\.(?:docx|xlsx|pptx|pdf|md|txt|json|m4a|wav)\b)",
    flags=re.IGNORECASE,
)


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
                ORDER BY
                    CASE state WHEN 'corrected' THEN 3 WHEN 'explicit' THEN 2 ELSE 1 END DESC,
                    importance DESC, updated_at DESC, created_at DESC LIMIT ?""",
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
        if row and str(row["content"] or "").startswith("历史聊天摘要（导入，建议后续核验）："):
            return None
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
            for raw in records[:8]:
                content = normalize_memory_content(str(raw.get("content") or ""))
                kind = str(raw.get("kind") or "fact").strip().lower()
                if not content or kind not in MEMORY_KINDS:
                    continue
                importance = normalize_score(raw.get("importance"), default=1.0 if state != "automatic" else 0.0)
                confidence = normalize_score(raw.get("confidence"), default=1.0 if state != "automatic" else 0.0)
                if state == "automatic" and not qualifies_as_automatic_core_memory(
                    raw,
                    content=content,
                    kind=kind,
                    project_id=project_id,
                    importance=importance,
                    confidence=confidence,
                ):
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
                        project_id=?, source_excerpt=?, kind=?, state=?, importance=?, confidence=?,
                        updated_at=? WHERE id=?""",
                        (
                            content, conversation_id, conversation_title, project_id,
                            source_excerpt[:1600], kind, state, importance, confidence, now,
                            existing["id"],
                        ),
                    )
                    row = connection.execute("SELECT * FROM memory_items WHERE id = ?", (existing["id"],)).fetchone()
                else:
                    similar = find_similar_automatic_memory(
                        connection,
                        kind=kind,
                        content=content,
                        project_id=project_id,
                    ) if state == "automatic" else None
                    if similar is not None:
                        if importance + 0.02 < float(similar["importance"] or 0):
                            saved.append(memory_payload(similar))
                            continue
                        connection.execute(
                            """UPDATE memory_items SET content=?, conversation_id=?,
                            conversation_title=?, source_excerpt=?, importance=?, confidence=?,
                            updated_at=? WHERE id=?""",
                            (
                                content, conversation_id, conversation_title,
                                source_excerpt[:1600], importance, confidence, now,
                                similar["id"],
                            ),
                        )
                        row = connection.execute(
                            "SELECT * FROM memory_items WHERE id = ?", (similar["id"],)
                        ).fetchone()
                        if row:
                            saved.append(memory_payload(row))
                        continue
                    if not make_room_for_core_memory(
                        connection,
                        project_id=project_id,
                        candidate_importance=importance,
                        candidate_state=state,
                    ):
                        continue
                    memory_id = f"memory-{uuid.uuid4().hex[:16]}"
                    connection.execute(
                        """INSERT INTO memory_items(id, fingerprint, kind, content, conversation_id,
                        conversation_title, project_id, source_excerpt, state, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            memory_id, fingerprint, kind, content, conversation_id,
                            conversation_title, project_id, source_excerpt[:1600], state, now, now,
                        ),
                    )
                    connection.execute(
                        "UPDATE memory_items SET importance=?, confidence=? WHERE id=?",
                        (importance, confidence, memory_id),
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
                """UPDATE memory_items SET content=?, state='corrected', importance=1,
                confidence=1, updated_at=? WHERE id=? AND state <> 'deleted'""",
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
                    created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL,
                    importance REAL NOT NULL DEFAULT 0.5,
                    confidence REAL NOT NULL DEFAULT 0.5
                );
                CREATE INDEX IF NOT EXISTS memory_items_scope_idx ON memory_items(project_id, state, updated_at DESC);
                CREATE TABLE IF NOT EXISTS memory_profiles (
                    project_id TEXT PRIMARY KEY, content TEXT NOT NULL, updated_at INTEGER NOT NULL
                );
                """
            )
            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(memory_items)").fetchall()
            }
            if "importance" not in columns:
                connection.execute(
                    "ALTER TABLE memory_items ADD COLUMN importance REAL NOT NULL DEFAULT 0.5"
                )
            if "confidence" not in columns:
                connection.execute(
                    "ALTER TABLE memory_items ADD COLUMN confidence REAL NOT NULL DEFAULT 0.5"
                )
            previous_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            # Earlier releases stored one long session summary per chat. Those
            # raw records remain available through chat-history recall, but are
            # intentionally never exposed or injected as the new memory profile.
            connection.execute(
                "DELETE FROM memory_profiles WHERE content LIKE '历史聊天摘要（导入，建议后续核验）：%'"
            )
            if previous_version < MEMORY_SCHEMA_VERSION:
                # V2 automatic records were produced after only three user
                # messages with no importance threshold or hard cap. They mix
                # salaries, paths and one-off task progress into global memory
                # and are unsafe to carry forward. Raw chats remain intact and
                # searchable through history recall.
                connection.execute("UPDATE memory_items SET state='deleted' WHERE state='automatic'")
                connection.execute("DELETE FROM memory_profiles")
            connection.execute(f"PRAGMA user_version = {MEMORY_SCHEMA_VERSION}")


def normalize_memory_content(content: str) -> str:
    return re.sub(r"\s+", " ", str(content or "").strip())[:MAX_MEMORY_CONTENT_CHARS]


def normalize_profile(content: str) -> str:
    return str(content or "").strip()[:MAX_PROFILE_CHARS]


def normalize_score(value: Any, *, default: float) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return default


def qualifies_as_automatic_core_memory(
    raw: dict[str, Any],
    *,
    content: str,
    kind: str,
    project_id: str,
    importance: float,
    confidence: float,
) -> bool:
    allowed_kinds = PROJECT_AUTOMATIC_KINDS if project_id else ACCOUNT_AUTOMATIC_KINDS
    if kind not in allowed_kinds:
        return False
    if importance < AUTOMATIC_MEMORY_MIN_IMPORTANCE:
        return False
    if confidence < AUTOMATIC_MEMORY_MIN_CONFIDENCE:
        return False
    durability = str(raw.get("durability") or "").strip().lower()
    evidence = str(raw.get("evidence") or "").strip().lower()
    if durability not in {"long_term", "permanent"}:
        return False
    if evidence not in {"explicit", "repeated"}:
        return False
    if len(content) < 8 or FILE_DETAIL_PATTERN.search(content):
        return False
    if any(marker in content for marker in TRANSIENT_MEMORY_MARKERS):
        return False
    return True


def find_similar_automatic_memory(
    connection: sqlite3.Connection,
    *,
    kind: str,
    content: str,
    project_id: str,
) -> sqlite3.Row | None:
    rows = connection.execute(
        """SELECT * FROM memory_items
        WHERE project_id=? AND kind=? AND state='automatic'""",
        (str(project_id or ""), kind),
    ).fetchall()
    normalized = content.casefold()
    best: tuple[float, sqlite3.Row] | None = None
    for row in rows:
        score = SequenceMatcher(
            None,
            normalized,
            str(row["content"] or "").casefold(),
        ).ratio()
        if score >= 0.72 and (best is None or score > best[0]):
            best = (score, row)
    return best[1] if best else None


def make_room_for_core_memory(
    connection: sqlite3.Connection,
    *,
    project_id: str,
    candidate_importance: float,
    candidate_state: str,
) -> bool:
    total_limit = MAX_PROJECT_CORE_MEMORIES if project_id else MAX_ACCOUNT_CORE_MEMORIES
    automatic_limit = MAX_PROJECT_AUTOMATIC_MEMORIES if project_id else MAX_ACCOUNT_AUTOMATIC_MEMORIES
    active_count = int(connection.execute(
        """SELECT COUNT(*) FROM memory_items
        WHERE project_id=? AND state <> 'deleted'""",
        (str(project_id or ""),),
    ).fetchone()[0])
    automatic_count = int(connection.execute(
        """SELECT COUNT(*) FROM memory_items
        WHERE project_id=? AND state='automatic'""",
        (str(project_id or ""),),
    ).fetchone()[0])
    if candidate_state != "automatic" and active_count < total_limit:
        return True
    if candidate_state == "automatic" and active_count < total_limit and automatic_count < automatic_limit:
        return True
    automatic_rows = connection.execute(
        """SELECT * FROM memory_items
        WHERE project_id=? AND state='automatic'
        ORDER BY importance ASC, updated_at ASC""",
        (str(project_id or ""),),
    ).fetchall()
    if not automatic_rows:
        return False
    weakest = automatic_rows[0]
    if (
        candidate_state == "automatic"
        and candidate_importance <= float(weakest["importance"] or 0) + 0.02
    ):
        return False
    connection.execute(
        "UPDATE memory_items SET state='deleted', updated_at=? WHERE id=?",
        (int(time.time()), weakest["id"]),
    )
    return True


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
        "importance": float(row["importance"]), "confidence": float(row["confidence"]),
    }


def profile_payload(row: sqlite3.Row) -> dict[str, Any]:
    return {"project_id": str(row["project_id"]), "content": str(row["content"]), "updated_at": int(row["updated_at"])}
