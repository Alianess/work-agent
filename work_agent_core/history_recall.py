from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable
import json
import math
import re
import sqlite3
import time

import numpy as np

from .cross_chat_memory import CrossChatMemoryStore
from .retrieval_core import (
    MlxRetrievalBackend,
    RetrievalBackend,
    RetrievalBackendError,
)
from .recall_archive import (
    RECALL_ARCHIVE_VERSION,
    approx_units,
    build_recall_episodes,
    render_episode_text,
)
from .session_store import (
    ConversationSession,
    SessionStore,
    repair_runtime_message_sequence,
    sanitize_conversation_id,
    sanitize_runtime_message,
)
from .tools import Tool, ToolRegistry


DEFAULT_LIMIT = 5
MAX_LIMIT = 8
MAX_QUERY_TERMS = 18
HISTORY_INDEX_FORMAT_VERSION = 5
CHUNK_TARGET_UNITS = 160
CHUNK_OVERLAP_UNITS = 24
CHUNK_MIN_UNITS = 70
CHUNK_MAX_UNITS = 240
VECTOR_BATCH_SIZE = 8
MODEL_INPUT_CHARS = 460
MODEL_INPUT_OVERLAP = 40
LEXICAL_CANDIDATE_LIMIT = 60
DENSE_CANDIDATE_LIMIT = 60
RRF_CANDIDATE_LIMIT = 40
RRF_K = 60

TOKEN_PATTERN = re.compile(
    r"[A-Za-z][A-Za-z0-9_.+-]*|[0-9]+(?:\.[0-9]+)*|[\u3400-\u4dbf\u4e00-\u9fff]+"
)
CJK_PATTERN = re.compile(r"^[\u3400-\u4dbf\u4e00-\u9fff]+$")
CJK_STOP_TERMS = {
    "之前", "我们", "你们", "他们", "这个", "那个", "一下", "什么", "怎么",
    "聊天", "历史", "提到", "说过", "讨论", "回想", "记得", "内容", "相关", "当时",
}


@dataclass(frozen=True)
class HistoryChunk:
    message_index: int
    chunk_index: int
    role: str
    content: str
    search_text: str
    parent_id: str
    parent_content: str
    source_kind: str


class ChatHistoryRecall:
    """Hybrid retrieval over one account's raw conversation sessions."""

    def __init__(
        self,
        session_store: SessionStore,
        conversation_id: str,
        *,
        project_id: str = "",
        retrieval_backend: RetrievalBackend | None = None,
    ) -> None:
        clean_id = sanitize_conversation_id(conversation_id)
        if not clean_id:
            raise ValueError("conversation_id is required")
        self.session_store = session_store
        self.conversation_id = clean_id
        self.project_id = str(project_id or "").strip()
        self.database_path = session_store.session_dir.parent / "history_search.sqlite3"
        self.retrieval_backend = retrieval_backend or MlxRetrievalBackend.from_env()

    def search(self, args: dict[str, Any]) -> str:
        query = str(args.get("query") or "").strip()
        keywords = normalize_keywords(args.get("keywords"))
        if not query and not keywords:
            raise ValueError("query 或 keywords 至少提供一项。")
        limit = max(1, min(int(args.get("limit") or DEFAULT_LIMIT), MAX_LIMIT))
        requested_scope = str(args.get("scope") or "auto").strip().lower()
        if requested_scope not in {"auto", "compressed", "all", "current", "project", "account"}:
            raise ValueError("scope 只能是 auto、compressed、current、all、project 或 account。")
        scope = resolve_search_scope(requested_scope, self.project_id)

        session = self.session_store.load(self.conversation_id)
        current_message_limit = current_searchable_message_limit(session, scope=scope)
        # Explicit keywords are model-selected anchors, so keep them ahead of
        # incidental words from the natural-language query when the cap is hit.
        terms = dedupe(
            [*extract_query_terms(" ".join(keywords)), *extract_query_terms(query)]
        )[:MAX_QUERY_TERMS]
        if scope in {"compressed", "current"} and current_message_limit <= 0:
            return json.dumps(
                {
                    "ok": True,
                    "conversation_id": self.conversation_id,
                    "requested_scope": requested_scope,
                    "scope": scope,
                    "query": query,
                    "query_terms": terms,
                    "results": [],
                    "note": (
                        "当前聊天还没有被压缩的历史原文；可改用 scope=current 搜索当前聊天全部已保存消息。"
                        if scope == "compressed"
                        else "当前聊天还没有可检索的历史消息。"
                    ),
                    "retrieval": "hybrid-history-rag",
                    "retrieval_status": {
                        "mode": "empty",
                        "bm25": True,
                        "dense": False,
                    },
                    "model_used": False,
                },
                ensure_ascii=False,
                indent=2,
            )

        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.database_path, timeout=8) as connection:
            connection.row_factory = sqlite3.Row
            configure_database(connection)
            ensure_schema(connection)
            archive = load_conversation_archive(self.session_store)
            scoped_sessions = load_scoped_sessions(
                self.session_store,
                current_session=session,
                archive=archive,
                scope=scope,
                project_id=self.project_id,
            )
            for scoped_session, metadata in scoped_sessions:
                ensure_conversation_index(connection, scoped_session, metadata=metadata)
            allowed_conversation_ids = [item.id for item, _metadata in scoped_sessions]
            lexical_rows = (
                fetch_candidates(
                    connection,
                    conversation_id=self.conversation_id,
                    scope=scope,
                    terms=terms,
                    current_message_limit=current_message_limit,
                    candidate_limit=max(LEXICAL_CANDIDATE_LIMIT, limit * 12),
                    allowed_conversation_ids=allowed_conversation_ids,
                )
                if terms
                else []
            )
            lexical_ranked = rank_candidates(
                lexical_rows,
                query=query,
                keywords=keywords,
                terms=terms,
                total_messages=max(1, len(session.messages)),
                current_conversation_id=self.conversation_id,
                current_project_id=self.project_id,
            )

            degraded_reasons: list[str] = []
            dense_ranked: list[dict[str, Any]] = []
            vectors_indexed = 0
            backend = self.retrieval_backend
            if backend.enabled:
                try:
                    vectors_indexed = ensure_vector_indices(
                        connection,
                        scoped_sessions,
                        backend=backend,
                    )
                except Exception as error:
                    degraded_reasons.append(format_backend_error("向量索引", error))
                try:
                    semantic_query = query or " ".join(keywords)
                    query_vector = normalize_vector(backend.embed_texts([semantic_query])[0])
                    dense_ranked = fetch_dense_candidates(
                        connection,
                        query_vector=query_vector,
                        model=backend.embedding_model,
                        conversation_id=self.conversation_id,
                        scope=scope,
                        current_message_limit=current_message_limit,
                        candidate_limit=max(DENSE_CANDIDATE_LIMIT, limit * 12),
                        allowed_conversation_ids=allowed_conversation_ids,
                    )
                except Exception as error:
                    degraded_reasons.append(format_backend_error("向量召回", error))
            else:
                degraded_reasons.append("向量召回已禁用")

            hybrid_ranked = rrf_merge_results(
                [lexical_ranked, dense_ranked],
                top_n=max(RRF_CANDIDATE_LIMIT, limit * 6),
            )
            ranked = dedupe_message_results(hybrid_ranked)[:limit]

        memory_results = rank_memory_candidates(
            CrossChatMemoryStore(self.session_store).active_for_scope(
                scope=scope,
                project_id=self.project_id,
            ),
            query=query,
            keywords=keywords,
            terms=terms,
        )[:limit]
        return json.dumps(
            {
                "ok": True,
                "conversation_id": self.conversation_id,
                "requested_scope": requested_scope,
                "scope": scope,
                "query": query,
                "query_terms": terms,
                "indexed_conversation_count": len(scoped_sessions),
                "indexed_vector_count": vectors_indexed,
                "memory_results": memory_results,
                "results": ranked,
                "note": (
                    "memory_results 是少量核心记忆；results 是混合检索得到的可核对聊天原文。两者都保留来源聊天。"
                    if memory_results or ranked
                    else "没有命中。可换用当时出现过的专名、数字、文件名或另一种描述重试。"
                ),
                "retrieval": "hybrid-history-rag",
                "retrieval_status": {
                    "mode": (
                        "hybrid"
                        if dense_ranked
                        else "bm25_fallback"
                    ),
                    "bm25": True,
                    "dense": bool(dense_ranked),
                    "embedding_model": (
                        self.retrieval_backend.embedding_model
                        if dense_ranked
                        else None
                    ),
                    "degraded": bool(degraded_reasons),
                    "degraded_reasons": dedupe(degraded_reasons),
                },
                "model_used": bool(dense_ranked),
            },
            ensure_ascii=False,
            indent=2,
        )


def register_history_recall_tool(
    registry: ToolRegistry,
    session_store: SessionStore,
    conversation_id: str,
    *,
    project_id: str = "",
) -> None:
    recall = ChatHistoryRecall(session_store, conversation_id, project_id=project_id)
    registry.register(
        Tool(
            name="recall_chat_history",
            description=(
                "Search bounded core memories and raw passages across the current account's saved chats using hybrid BM25, MLX semantic retrieval, and RRF fusion. "
                "Inside a project, auto scope searches chats from that project only; outside projects, auto scope searches non-project chats in the account. "
                "Use when the user refers to something discussed earlier, the compressed summary lacks a detail, "
                "or an exact name, number, decision, wording, path, or prior correction must be recovered. "
                "The natural-language query can retrieve paraphrases; keywords remain useful for exact names and numbers. "
                "Results include the source conversation title and project. Account boundaries are always enforced."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language recall query containing distinctive names, numbers, phrases, or topics.",
                    },
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional extra exact keywords or aliases to broaden recall.",
                    },
                    "scope": {
                        "type": "string",
                        "enum": ["auto", "compressed", "current", "all", "project", "account"],
                        "default": "auto",
                        "description": "auto searches the current project when present, otherwise all chats in the account. compressed searches summarized-away messages in the current chat. current/all search the current chat. project searches the current project's chats. account searches all chats, unless project-only isolation forces project scope.",
                    },
                    "limit": {
                        "type": "integer", "default": DEFAULT_LIMIT, "minimum": 1, "maximum": MAX_LIMIT,
                    },
                },
                "required": ["query"],
            },
            handler=recall.search,
            metadata={"layer": "core", "read_only": True, "scope": "account_conversations"},
        )
    )


def render_history_recall_system_context(summary_message_count: int, *, project_id: str = "") -> str:
    covered = max(0, int(summary_message_count or 0))
    return (
        "\n\n跨聊天记忆与原文回想能力：recall_chat_history 是只读 core 工具，"
        "内部使用 BM25、MLX 语义向量和 RRF，并在模型不可用时自动退回 BM25。"
        "它会返回少量核心 memory_results，并同时返回带来源、可核对的原文 results。"
        "当用户说‘之前、当时、你还记得、我们讨论过’或需要核对较早的名称、数字、决定、原话、文件路径、纠错时，"
        "如果最近 messages 或压缩摘要不能可靠回答，必须先调用该工具。"
        "通常使用 scope=auto：项目聊天只检索同一项目，普通聊天检索账号内非项目聊天；项目专属记忆不会泄漏到项目外。"
        "需要限定当前聊天时使用 scope=current，需要专门找被摘要覆盖的原文时使用 scope=compressed。"
        "query 应写完整问题；核对名称、数字和原话时可补充 keywords。无结果时换同义词或别名重试一次。"
        "不得把未命中解释为用户从未说过。"
        "日报、周报、双周报及其补写纠错必须优先使用 work-reports 技能的 collect_work_report_evidence 按日期取证；"
        "不得用 scope=compressed 代替账户级日期证据检索，也不得把当前聊天未命中表述为全账户未命中。"
        "memory_results 如为 corrected 表示用户已纠正，应优先于原摘要；回答关键数字和原话时仍应用 results 核对。"
        f"当前已有 {covered} 条较早 runtime messages 被压缩摘要覆盖，可用 scope=compressed 找回其原文。"
        f"当前项目：{project_id or '无；auto 将检索账号全部聊天'}。"
    )


def resolve_search_scope(requested_scope: str, project_id: str) -> str:
    if requested_scope == "auto":
        return "project" if project_id else "account"
    if requested_scope == "all":
        return "current"
    if project_id and requested_scope == "account":
        # Every local project currently uses project-only memory.
        return "project"
    return requested_scope


def configure_database(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA busy_timeout=8000")


def ensure_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS history_conversation_meta (
            conversation_id TEXT PRIMARY KEY,
            title TEXT NOT NULL DEFAULT '',
            project_id TEXT NOT NULL DEFAULT '',
            updated_at INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS history_chunk_parent (
            conversation_id TEXT NOT NULL,
            message_index INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            parent_id TEXT NOT NULL,
            parent_content TEXT NOT NULL,
            source_kind TEXT NOT NULL,
            PRIMARY KEY(conversation_id, message_index, chunk_index)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS history_index_meta (
            conversation_id TEXT PRIMARY KEY,
            signature TEXT NOT NULL,
            message_count INTEGER NOT NULL,
            updated_at INTEGER NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS chat_history_fts USING fts5(
            conversation_id UNINDEXED,
            message_index UNINDEXED,
            chunk_index UNINDEXED,
            role UNINDEXED,
            content UNINDEXED,
            search_text,
            tokenize='unicode61 remove_diacritics 2'
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS history_chunk_vectors (
            conversation_id TEXT NOT NULL,
            message_index INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            content_signature TEXT NOT NULL,
            model TEXT NOT NULL,
            dimensions INTEGER NOT NULL,
            embedding BLOB NOT NULL,
            updated_at INTEGER NOT NULL,
            PRIMARY KEY(conversation_id, message_index, chunk_index)
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS history_chunk_vectors_model_idx
        ON history_chunk_vectors(model)
        """
    )


def ensure_conversation_index(
    connection: sqlite3.Connection,
    session: ConversationSession,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    metadata = metadata or {}
    project_id = str(metadata.get("project_id") or session.metadata.get("project_id") or "").strip()
    title = str(metadata.get("title") or session.metadata.get("title") or session.id).strip()
    updated_at = max(int(metadata.get("updated_at") or 0), int(session.updated_at or 0))
    with connection:
        connection.execute(
            """
            INSERT INTO history_conversation_meta(conversation_id, title, project_id, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(conversation_id) DO UPDATE SET
                title=excluded.title,
                project_id=excluded.project_id,
                updated_at=excluded.updated_at
            """,
            (session.id, title, project_id, updated_at),
        )
    signature = session_signature(session, title=title)
    row = connection.execute(
        "SELECT signature, message_count FROM history_index_meta WHERE conversation_id = ?", (session.id,)
    ).fetchone()
    if row is not None and str(row[0]) == signature:
        return
    # Parent episodes can merge across the append boundary, so a changed
    # conversation is projected atomically. The local corpus is small and this
    # avoids maintaining two subtly different chunking paths.
    chunks = list(iter_session_chunks(session, title=title))
    with connection:
        connection.execute("DELETE FROM chat_history_fts WHERE conversation_id = ?", (session.id,))
        connection.execute(
            "DELETE FROM history_chunk_parent WHERE conversation_id = ?",
            (session.id,),
        )
        connection.executemany(
            """
            INSERT INTO chat_history_fts(
                conversation_id, message_index, chunk_index, role, content, search_text
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (session.id, chunk.message_index, chunk.chunk_index, chunk.role, chunk.content, chunk.search_text)
                for chunk in chunks
            ],
        )
        connection.executemany(
            """
            INSERT INTO history_chunk_parent(
                conversation_id, message_index, chunk_index,
                parent_id, parent_content, source_kind
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    session.id,
                    chunk.message_index,
                    chunk.chunk_index,
                    chunk.parent_id,
                    chunk.parent_content,
                    chunk.source_kind,
                )
                for chunk in chunks
            ],
        )
        connection.execute(
            """
            INSERT INTO history_index_meta(conversation_id, signature, message_count, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(conversation_id) DO UPDATE SET
                signature=excluded.signature,
                message_count=excluded.message_count,
                updated_at=excluded.updated_at
            """,
            (session.id, signature, len(session.messages), int(session.updated_at)),
        )


def session_signature(session: ConversationSession, *, title: str = "") -> str:
    digest = sha256()
    digest.update(str(HISTORY_INDEX_FORMAT_VERSION).encode("ascii"))
    digest.update(b"\0")
    digest.update(messages_signature(session.messages, title=title).encode("ascii"))
    digest.update(b"\0")
    digest.update(
        json.dumps(session.recall_episodes, ensure_ascii=False, sort_keys=True).encode(
            "utf-8", errors="replace"
        )
    )
    return digest.hexdigest()


def messages_signature(
    messages: Iterable[dict[str, Any]],
    *,
    title: str = "",
) -> str:
    digest = sha256()
    digest.update(str(HISTORY_INDEX_FORMAT_VERSION).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(title or "").encode("utf-8", errors="replace"))
    digest.update(b"\0")
    for message in messages:
        digest.update(str(message.get("role") or "").encode("utf-8"))
        digest.update(b"\0")
        digest.update(message_text(message).encode("utf-8", errors="replace"))
        digest.update(b"\0")
    return digest.hexdigest()


def current_searchable_message_limit(session: ConversationSession, *, scope: str) -> int:
    end = len(session.messages)
    if end and session.messages[-1].get("role") == "user":
        end -= 1
    if scope == "compressed":
        end = min(end, max(0, int(session.summary_message_count or 0)))
    return max(0, end)


def load_conversation_archive(session_store: SessionStore) -> dict[str, dict[str, Any]]:
    path = session_store.session_dir.parent / "conversations.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    items = data.get("items") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return {}
    if isinstance(data, dict) and data.get("storage") == "per_item":
        raw_order = data.get("order") if isinstance(data.get("order"), list) else []
        order = [str(value or "").strip() for value in raw_order]
        if not order:
            order = [str(item.get("id") or "").strip() for item in items if isinstance(item, dict)]
        loaded_items: list[dict[str, Any]] = []
        for conversation_id in order:
            if not conversation_id:
                continue
            item_path = path.parent / "archive_items" / f"{conversation_id}.json"
            try:
                item = json.loads(item_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(item, dict) and sanitize_conversation_id(item.get("id")) == conversation_id:
                loaded_items.append(item)
        items = loaded_items
    archive: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        conversation_id = sanitize_conversation_id(item.get("id"))
        if not conversation_id:
            continue
        archive[conversation_id] = item
    return archive


def load_scoped_sessions(
    session_store: SessionStore,
    *,
    current_session: ConversationSession,
    archive: dict[str, dict[str, Any]],
    scope: str,
    project_id: str,
) -> list[tuple[ConversationSession, dict[str, Any]]]:
    if scope in {"compressed", "current"}:
        return [(current_session, archive_metadata(archive.get(current_session.id), current_session))]

    session_ids = {
        path.stem
        for path in session_store.session_dir.glob("*.json")
        if sanitize_conversation_id(path.stem)
    }
    session_ids.update(archive)
    session_ids.add(current_session.id)
    scoped: list[tuple[ConversationSession, dict[str, Any]]] = []
    for conversation_id in sorted(session_ids):
        session = current_session if conversation_id == current_session.id else session_store.load(conversation_id)
        archive_item = archive.get(conversation_id)
        if not session.messages and archive_item is not None:
            session = session_from_archive_item(archive_item, conversation_id)
        metadata = archive_metadata(archive_item, session)
        if scope == "project" and str(metadata.get("project_id") or "") != project_id:
            continue
        if scope == "account" and str(metadata.get("project_id") or ""):
            continue
        if session.messages:
            scoped.append((session, metadata))
    return scoped


def archive_metadata(
    item: dict[str, Any] | None,
    session: ConversationSession,
) -> dict[str, Any]:
    item = item or {}
    if "project_id" in session.metadata:
        project_id = str(session.metadata.get("project_id") or "")
    else:
        project_id = str(item.get("projectId") or item.get("project_id") or "")
    return {
        "title": str(item.get("title") or session.metadata.get("title") or session.id),
        "project_id": project_id,
        "updated_at": int(session.updated_at or 0),
    }


def session_from_archive_item(item: dict[str, Any], conversation_id: str) -> ConversationSession:
    raw_messages = item.get("messages") if isinstance(item.get("messages"), list) else []
    messages = [message for message in (sanitize_runtime_message(raw) for raw in raw_messages) if message]
    return ConversationSession(
        id=conversation_id,
        messages=repair_runtime_message_sequence(messages),
        summary=str(item.get("contextSummary") or ""),
        summary_message_count=max(0, int(item.get("contextSummaryMessageCount") or 0)),
        metadata={"project_id": str(item.get("projectId") or ""), "title": str(item.get("title") or "")},
    )


def iter_session_chunks(
    session: ConversationSession,
    *,
    start_index: int = 0,
    title: str = "",
) -> Iterable[HistoryChunk]:
    del start_index  # Semantic parents may merge across append boundaries.
    covered_count = min(
        max(0, int(session.summary_message_count or 0)),
        len(session.messages),
    )
    archived = [
        item for item in session.recall_episodes if isinstance(item, dict)
    ] if covered_count else []
    archive_is_stale = any(
        int(item.get("archive_version") or 0) != RECALL_ARCHIVE_VERSION
        for item in archived
    )
    if covered_count and (not archived or archive_is_stale):
        archived = build_recall_episodes(session.messages[:covered_count])
    live = build_recall_episodes(
        session.messages[covered_count:],
        start_message_index=covered_count,
    )
    episodes = [*archived, *live]
    episode_covered_indexes: set[int] = set()
    for episode in episodes:
        start = max(0, int(episode.get("start_message_index") or 0))
        end = max(start, int(episode.get("end_message_index") or start))
        episode_covered_indexes.update(range(start, end))
        parent_content = render_episode_text(episode)
        parent_id = str(episode.get("id") or f"episode-{start}-{end}")
        context_prefix = f"{title}\n" if title else ""
        for chunk_index, chunk in enumerate(chunk_text(parent_content)):
            search_text = " ".join(index_terms(f"{context_prefix}{chunk}"))
            if search_text:
                yield HistoryChunk(
                    start,
                    chunk_index,
                    "episode",
                    chunk,
                    search_text,
                    parent_id,
                    parent_content,
                    "recall_episode",
                )

    # Keep legacy/imported assistant-only records searchable. Normal completed
    # user turns have already been projected into episodes above.
    for message_index in range(len(session.messages)):
        if message_index in episode_covered_indexes:
            continue
        message = session.messages[message_index]
        content = message_text(message).strip()
        if not content:
            continue
        role = str(message.get("role") or "unknown")
        for chunk_index, chunk in enumerate(chunk_text(content)):
            search_text = " ".join(index_terms(f"{title}\n{chunk}" if title else chunk))
            if search_text:
                parent_id = f"message-{message_index}"
                yield HistoryChunk(
                    message_index,
                    chunk_index,
                    role,
                    chunk,
                    search_text,
                    parent_id,
                    content,
                    "legacy_message",
                )


def message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = "\n".join(
            str(item.get("text") or "")
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        )
    else:
        text = ""
    if message.get("role") == "tool":
        name = str(message.get("name") or "").strip()
        return f"工具结果 {name}：\n{text}" if name else text
    return text


def chunk_text(
    text: str,
    *,
    target_units: int = CHUNK_TARGET_UNITS,
    overlap_units: int = CHUNK_OVERLAP_UNITS,
    min_units: int = CHUNK_MIN_UNITS,
    max_units: int = CHUNK_MAX_UNITS,
) -> list[str]:
    value = str(text or "").strip()
    if not value:
        return []
    if approx_units(value) <= max_units:
        return [value]
    blocks = semantic_blocks(value, max_units=max_units)
    chunks: list[str] = []
    start = 0
    while start < len(blocks):
        end = start
        units = 0
        while end < len(blocks):
            next_units = approx_units(blocks[end])
            if end > start and units + next_units > target_units:
                break
            units += next_units
            end += 1
            if units >= target_units:
                break
        if end == start:
            end += 1
        chunk = "\n\n".join(blocks[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(blocks):
            break
        overlap = 0
        next_start = end
        while next_start > start + 1 and overlap < overlap_units:
            next_start -= 1
            overlap += approx_units(blocks[next_start])
        start = next_start

    if len(chunks) >= 2 and approx_units(chunks[-1]) < min_units:
        merged = f"{chunks[-2]}\n\n{chunks[-1]}".strip()
        if approx_units(merged) <= max_units:
            chunks[-2:] = [merged]
    return chunks


def split_model_input(
    text: str,
    *,
    size: int = MODEL_INPUT_CHARS,
    overlap: int = MODEL_INPUT_OVERLAP,
) -> list[str]:
    """Split one FTS passage for MLX, then mean-pool its segment vectors."""
    return split_text_by_chars(text, size=size, overlap=overlap)


def semantic_blocks(text: str, *, max_units: int) -> list[str]:
    """Prefer Markdown/paragraph/sentence boundaries without dropping tails."""
    paragraphs = [
        part.strip()
        for part in re.split(r"\n\s*\n+", str(text or "").replace("\r\n", "\n"))
        if part.strip()
    ]
    blocks: list[str] = []
    for paragraph in paragraphs:
        if approx_units(paragraph) <= max_units:
            blocks.append(paragraph)
            continue
        sentences = [
            item.strip()
            for item in re.split(r"(?<=[。！？!?；;])\s*|(?<=\.)\s+(?=[A-Z0-9])", paragraph)
            if item.strip()
        ]
        if len(sentences) <= 1:
            blocks.extend(split_text_by_units(paragraph, max_units=max_units))
            continue
        current: list[str] = []
        current_units = 0
        for sentence in sentences:
            sentence_units = approx_units(sentence)
            if sentence_units > max_units:
                if current:
                    blocks.append(" ".join(current).strip())
                    current = []
                    current_units = 0
                blocks.extend(split_text_by_units(sentence, max_units=max_units))
                continue
            if current and current_units + sentence_units > max_units:
                blocks.append(" ".join(current).strip())
                current = []
                current_units = 0
            current.append(sentence)
            current_units += sentence_units
        if current:
            blocks.append(" ".join(current).strip())
    return [block for block in blocks if block]


def split_text_by_units(text: str, *, max_units: int) -> list[str]:
    tokens = re.findall(
        r"[\u3400-\u9fff]|[A-Za-z0-9_][A-Za-z0-9_.+-]*|\s+|.",
        str(text or ""),
        flags=re.DOTALL,
    )
    parts: list[str] = []
    current: list[str] = []
    units = 0
    for token in tokens:
        token_units = approx_units(token)
        if current and units + token_units > max_units:
            parts.append("".join(current).strip())
            current = []
            units = 0
        current.append(token)
        units += token_units
    if current:
        parts.append("".join(current).strip())
    return [part for part in parts if part]


def split_text_by_chars(text: str, *, size: int, overlap: int) -> list[str]:
    value = str(text or "").strip()
    if not value:
        return []
    if len(value) <= size:
        return [value]
    parts: list[str] = []
    start = 0
    while start < len(value):
        hard_end = min(len(value), start + size)
        end = hard_end
        if hard_end < len(value):
            boundary = max(
                value.rfind("\n", start + size // 2, hard_end),
                value.rfind("。", start + size // 2, hard_end),
                value.rfind("！", start + size // 2, hard_end),
                value.rfind("？", start + size // 2, hard_end),
                value.rfind(". ", start + size // 2, hard_end),
            )
            if boundary > start:
                end = boundary + 1
        parts.append(value[start:end].strip())
        if end >= len(value):
            break
        start = max(start + 1, end - overlap)
    return [part for part in parts if part]


def normalize_keywords(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text and text not in result:
            result.append(text)
    return result[:12]


def index_terms(text: str) -> list[str]:
    terms: list[str] = []
    for raw in TOKEN_PATTERN.findall(str(text or "")):
        token = raw.lower()
        if CJK_PATTERN.fullmatch(token):
            terms.extend(cjk_ngrams(token, include_unigrams=True))
        elif len(token) >= 2 or token.isdigit():
            terms.append(token)
    return dedupe(terms)


def extract_query_terms(text: str) -> list[str]:
    terms: list[str] = []
    cleaned = str(text or "")
    for stop_term in sorted(CJK_STOP_TERMS, key=len, reverse=True):
        cleaned = cleaned.replace(stop_term, " ")
    for raw in TOKEN_PATTERN.findall(cleaned):
        token = raw.lower()
        if CJK_PATTERN.fullmatch(token):
            grams = cjk_ngrams(token, include_unigrams=len(token) == 1)
            terms.extend(term for term in grams if term not in CJK_STOP_TERMS)
        elif len(token) >= 2 or token.isdigit():
            terms.append(token)
    unique = dedupe(terms)
    if len(unique) <= MAX_QUERY_TERMS:
        return unique
    ranked = sorted(enumerate(unique), key=lambda item: (-len(item[1]), item[0]))[:MAX_QUERY_TERMS]
    keep = {index for index, _ in ranked}
    return [term for index, term in enumerate(unique) if index in keep]


def cjk_ngrams(value: str, *, include_unigrams: bool) -> list[str]:
    if len(value) == 1:
        return [value]
    terms = [value[index : index + 2] for index in range(len(value) - 1)]
    if include_unigrams:
        terms.extend(value)
    return terms


def dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def chunk_signature(content: str) -> str:
    return sha256(str(content or "").encode("utf-8", errors="replace")).hexdigest()


def normalize_vector(vector: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(vector), dtype=np.float32)
    if array.ndim != 1 or array.size <= 0:
        raise RetrievalBackendError("embedding vector is empty")
    norm = float(np.linalg.norm(array))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise RetrievalBackendError("embedding vector norm is invalid")
    return array / norm


def ensure_vector_indices(
    connection: sqlite3.Connection,
    scoped_sessions: list[tuple[ConversationSession, dict[str, Any]]],
    *,
    backend: RetrievalBackend,
) -> int:
    """Incrementally backfill normalized vectors for the currently allowed scope."""
    indexed = 0
    for session, metadata in scoped_sessions:
        # Tool payloads are retained in FTS for exact path/value recovery, but
        # excluded from semantic indexing: they dominate chat volume and often
        # contain protocol/debug noise rather than user-visible conversation.
        chunks = [
            chunk
            for chunk in iter_session_chunks(
                session,
                title=str(metadata.get("title") or ""),
            )
            if chunk.role in {"user", "assistant", "episode"}
        ]
        valid = {
            (chunk.message_index, chunk.chunk_index): chunk_signature(chunk.content)
            for chunk in chunks
        }
        rows = connection.execute(
            """
            SELECT message_index, chunk_index, content_signature, model
            FROM history_chunk_vectors
            WHERE conversation_id = ?
            """,
            (session.id,),
        ).fetchall()
        existing = {
            (int(row["message_index"]), int(row["chunk_index"])): (
                str(row["content_signature"] or ""),
                str(row["model"] or ""),
            )
            for row in rows
        }
        stale_keys = [
            key
            for key, (signature, model) in existing.items()
            if key not in valid
            or signature != valid[key]
            or model != backend.embedding_model
        ]
        with connection:
            for message_index, chunk_index in stale_keys:
                connection.execute(
                    """
                    DELETE FROM history_chunk_vectors
                    WHERE conversation_id = ? AND message_index = ? AND chunk_index = ?
                    """,
                    (session.id, message_index, chunk_index),
                )

        pending = [
            chunk
            for chunk in chunks
            if existing.get((chunk.message_index, chunk.chunk_index))
            != (chunk_signature(chunk.content), backend.embedding_model)
        ]
        for start in range(0, len(pending), VECTOR_BATCH_SIZE):
            batch = pending[start : start + VECTOR_BATCH_SIZE]
            segments: list[str] = []
            owners: list[int] = []
            for owner, chunk in enumerate(batch):
                for segment in split_model_input(chunk.content):
                    segments.append(segment)
                    owners.append(owner)
            segment_vectors = backend.embed_texts(segments)
            if len(segment_vectors) != len(segments):
                raise RetrievalBackendError(
                    "embedding segment count mismatch: "
                    f"expected={len(segments)} actual={len(segment_vectors)}"
                )
            grouped: list[list[np.ndarray]] = [[] for _chunk in batch]
            for owner, vector in zip(owners, segment_vectors):
                grouped[owner].append(normalize_vector(vector))
            records: list[tuple[Any, ...]] = []
            for chunk, group in zip(batch, grouped):
                if not group:
                    raise RetrievalBackendError("embedding chunk produced no segments")
                normalized = normalize_vector(np.mean(np.vstack(group), axis=0))
                records.append(
                    (
                        session.id,
                        chunk.message_index,
                        chunk.chunk_index,
                        chunk_signature(chunk.content),
                        backend.embedding_model,
                        int(normalized.size),
                        normalized.astype("<f4", copy=False).tobytes(),
                        int(time.time()),
                    )
                )
            with connection:
                connection.executemany(
                    """
                    INSERT INTO history_chunk_vectors(
                        conversation_id, message_index, chunk_index,
                        content_signature, model, dimensions, embedding, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(conversation_id, message_index, chunk_index) DO UPDATE SET
                        content_signature=excluded.content_signature,
                        model=excluded.model,
                        dimensions=excluded.dimensions,
                        embedding=excluded.embedding,
                        updated_at=excluded.updated_at
                    """,
                    records,
                )
            indexed += len(records)
    return indexed


def fetch_dense_candidates(
    connection: sqlite3.Connection,
    *,
    query_vector: np.ndarray,
    model: str,
    conversation_id: str,
    scope: str,
    current_message_limit: int,
    candidate_limit: int,
    allowed_conversation_ids: list[str],
) -> list[dict[str, Any]]:
    if not allowed_conversation_ids:
        return []
    placeholders = ", ".join("?" for _item in allowed_conversation_ids)
    conditions = [
        "vectors.model = ?",
        "vectors.dimensions = ?",
        f"vectors.conversation_id IN ({placeholders})",
    ]
    parameters: list[Any] = [model, int(query_vector.size), *allowed_conversation_ids]
    if scope in {"compressed", "current"}:
        conditions.extend(
            [
                "vectors.conversation_id = ?",
                "vectors.message_index < ?",
            ]
        )
        parameters.extend([conversation_id, current_message_limit])
    else:
        conditions.append(
            "(vectors.conversation_id <> ? OR vectors.message_index < ?)"
        )
        parameters.extend([conversation_id, current_message_limit])
    rows = connection.execute(
        f"""
        SELECT vectors.conversation_id,
               vectors.message_index,
               vectors.chunk_index,
               vectors.dimensions,
               vectors.embedding,
               chat_history_fts.role,
               chat_history_fts.content,
               parent.parent_id,
               parent.parent_content,
               parent.source_kind,
               meta.title AS conversation_title,
               meta.project_id,
               meta.updated_at
        FROM history_chunk_vectors AS vectors
        JOIN chat_history_fts
          ON chat_history_fts.conversation_id = vectors.conversation_id
         AND CAST(chat_history_fts.message_index AS INTEGER) = vectors.message_index
         AND CAST(chat_history_fts.chunk_index AS INTEGER) = vectors.chunk_index
        JOIN history_conversation_meta AS meta
          ON meta.conversation_id = vectors.conversation_id
        JOIN history_chunk_parent AS parent
          ON parent.conversation_id = vectors.conversation_id
         AND parent.message_index = vectors.message_index
         AND parent.chunk_index = vectors.chunk_index
        WHERE {" AND ".join(conditions)}
        """,
        parameters,
    ).fetchall()
    valid_rows: list[sqlite3.Row] = []
    vectors: list[np.ndarray] = []
    for row in rows:
        vector = np.frombuffer(row["embedding"], dtype="<f4")
        if vector.size != query_vector.size:
            continue
        valid_rows.append(row)
        vectors.append(vector)
    if not vectors:
        return []
    matrix = np.vstack(vectors)
    scores = matrix @ query_vector
    order = np.argsort(scores)[::-1][:candidate_limit]
    results: list[dict[str, Any]] = []
    for index in order:
        row = valid_rows[int(index)]
        similarity = float(scores[int(index)])
        results.append(
            {
                "conversation_id": str(row["conversation_id"] or ""),
                "conversation_title": str(
                    row["conversation_title"] or row["conversation_id"] or ""
                ),
                "project_id": str(row["project_id"] or ""),
                "conversation_updated_at": int(row["updated_at"] or 0),
                "message_ordinal": int(row["message_index"]) + 1,
                "chunk_ordinal": int(row["chunk_index"]) + 1,
                "role": str(row["role"] or "unknown"),
                "score": round(similarity, 6),
                "matched_terms": [],
                "matched_by": ["dense"],
                "retrieval_scores": {"dense": round(similarity, 6)},
                "content": str(row["parent_content"] or row["content"] or ""),
                "matched_content": str(row["content"] or ""),
                "parent_id": str(row["parent_id"] or ""),
                "source_kind": str(row["source_kind"] or ""),
                "return_mode": "parent_episode",
            }
        )
    return results


def result_key(result: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(result.get("conversation_id") or ""),
        int(result.get("message_ordinal") or 0),
        int(result.get("chunk_ordinal") or 0),
    )


def rrf_merge_results(
    ranked_lists: list[list[dict[str, Any]]],
    *,
    top_n: int,
    k: int = RRF_K,
) -> list[dict[str, Any]]:
    scores: dict[tuple[str, int, int], float] = {}
    payloads: dict[tuple[str, int, int], dict[str, Any]] = {}
    first_seen: dict[tuple[str, int, int], tuple[int, int]] = {}
    channel_names = ("bm25", "dense")
    for list_index, ranked in enumerate(ranked_lists):
        channel = channel_names[list_index] if list_index < len(channel_names) else f"rank_{list_index}"
        for rank, raw in enumerate(ranked):
            key = result_key(raw)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            first_seen.setdefault(key, (list_index, rank))
            payload = payloads.setdefault(key, dict(raw))
            payload["matched_terms"] = dedupe(
                [
                    *list(payload.get("matched_terms") or []),
                    *list(raw.get("matched_terms") or []),
                ]
            )
            payload["matched_by"] = dedupe(
                [
                    *list(payload.get("matched_by") or []),
                    channel,
                ]
            )
            retrieval_scores = dict(payload.get("retrieval_scores") or {})
            if channel == "bm25":
                retrieval_scores.setdefault("bm25", raw.get("score"))
            elif channel == "dense":
                retrieval_scores.setdefault("dense", raw.get("score"))
            payload["retrieval_scores"] = retrieval_scores
    ordered = sorted(
        scores,
        key=lambda key: (
            -scores[key],
            first_seen[key][0],
            first_seen[key][1],
        ),
    )
    results: list[dict[str, Any]] = []
    for key in ordered[:top_n]:
        payload = payloads[key]
        payload["rrf_score"] = round(scores[key], 8)
        payload["score"] = round(scores[key], 8)
        results.append(payload)
    return results


def dedupe_message_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    deduped: list[dict[str, Any]] = []
    for result in results:
        parent_key = (
            str(result.get("conversation_id") or ""),
            str(result.get("parent_id") or f"message-{result.get('message_ordinal')}"),
        )
        if parent_key in seen:
            existing = seen[parent_key]
            passages = list(existing.get("matched_passages") or [])
            matched = str(result.get("matched_content") or "").strip()
            if matched and matched not in passages:
                passages.append(matched)
            existing["matched_passages"] = passages[:3]
            existing["matched_terms"] = dedupe(
                [*list(existing.get("matched_terms") or []), *list(result.get("matched_terms") or [])]
            )
            existing["matched_by"] = dedupe(
                [*list(existing.get("matched_by") or []), *list(result.get("matched_by") or [])]
            )
            continue
        item = dict(result)
        matched = str(item.get("matched_content") or "").strip()
        item["matched_passages"] = [matched] if matched else []
        seen[parent_key] = item
        deduped.append(item)
    return deduped


def format_backend_error(stage: str, error: Exception) -> str:
    text = re.sub(r"\s+", " ", str(error or "")).strip()
    if len(text) > 240:
        text = text[:237].rstrip() + "..."
    return f"{stage}不可用：{text or type(error).__name__}"


def fetch_candidates(
    connection: sqlite3.Connection,
    *,
    conversation_id: str,
    scope: str,
    terms: list[str],
    current_message_limit: int,
    candidate_limit: int,
    allowed_conversation_ids: list[str],
) -> list[sqlite3.Row]:
    if not terms or not allowed_conversation_ids:
        return []
    expression = " OR ".join(fts_quote(term) for term in terms)
    placeholders = ", ".join("?" for _item in allowed_conversation_ids)
    conditions = [
        "chat_history_fts MATCH ?",
        f"chat_history_fts.conversation_id IN ({placeholders})",
    ]
    parameters: list[Any] = [expression]
    parameters.extend(allowed_conversation_ids)
    if scope in {"compressed", "current"}:
        conditions.extend(
            [
                "chat_history_fts.conversation_id = ?",
                "CAST(chat_history_fts.message_index AS INTEGER) < ?",
            ]
        )
        parameters.extend([conversation_id, current_message_limit])
    else:
        conditions.append(
            "(chat_history_fts.conversation_id <> ? OR CAST(chat_history_fts.message_index AS INTEGER) < ?)"
        )
        parameters.extend([conversation_id, current_message_limit])
    parameters.append(candidate_limit)
    where_clause = " AND ".join(conditions)
    return list(
        connection.execute(
            f"""
            SELECT chat_history_fts.conversation_id,
                   chat_history_fts.message_index,
                   chat_history_fts.chunk_index,
                   chat_history_fts.role,
                   chat_history_fts.content,
                   chat_history_fts.search_text,
                   parent.parent_id,
                   parent.parent_content,
                   parent.source_kind,
                   meta.title AS conversation_title,
                   meta.project_id,
                   meta.updated_at,
                   bm25(chat_history_fts) AS bm25_score
            FROM chat_history_fts
            JOIN history_conversation_meta AS meta
              ON meta.conversation_id = chat_history_fts.conversation_id
            JOIN history_chunk_parent AS parent
              ON parent.conversation_id = chat_history_fts.conversation_id
             AND parent.message_index = CAST(chat_history_fts.message_index AS INTEGER)
             AND parent.chunk_index = CAST(chat_history_fts.chunk_index AS INTEGER)
            WHERE {where_clause}
            ORDER BY bm25_score ASC
            LIMIT ?
            """,
            parameters,
        )
    )


def fts_quote(term: str) -> str:
    return '"' + str(term).replace('"', '""') + '"'


def rank_candidates(
    rows: list[sqlite3.Row],
    *,
    query: str,
    keywords: list[str],
    terms: list[str],
    total_messages: int,
    current_conversation_id: str,
    current_project_id: str,
) -> list[dict[str, Any]]:
    normalized_phrases = [normalize_for_exact(value) for value in [query, *keywords] if value.strip()]
    ranked: list[tuple[float, dict[str, Any]]] = []
    per_message: dict[tuple[str, int], int] = {}
    for row in rows:
        source_conversation_id = str(row["conversation_id"] or "")
        message_index = int(row["message_index"])
        message_key = (source_conversation_id, message_index)
        if per_message.get(message_key, 0) >= 2:
            continue
        search_tokens = set(str(row["search_text"] or "").split())
        matched_terms = [term for term in terms if term in search_tokens]
        if not matched_terms:
            continue
        coverage = len(matched_terms) / max(1, len(terms))
        matched_content = str(row["content"] or "")
        parent_content = str(row["parent_content"] or matched_content)
        normalized_content = normalize_for_exact(parent_content)
        exact_hits = sum(1 for phrase in normalized_phrases if len(phrase) >= 2 and phrase in normalized_content)
        bm25_score = float(row["bm25_score"] or 0.0)
        lexical = math.log1p(max(0.0, -bm25_score) * 1000.0)
        recency = min(1.0, message_index / max(1, total_messages - 1))
        current_boost = 0.35 if source_conversation_id == current_conversation_id else 0.0
        project_boost = (
            0.25
            if current_project_id and str(row["project_id"] or "") == current_project_id
            else 0.0
        )
        role_penalty = -1.5 if str(row["role"] or "") == "tool" else 0.0
        score = (
            coverage * 8.0
            + exact_hits * 5.0
            + lexical
            + recency * 0.2
            + current_boost
            + project_boost
            + role_penalty
        )
        payload = {
            "conversation_id": source_conversation_id,
            "conversation_title": str(row["conversation_title"] or source_conversation_id),
            "project_id": str(row["project_id"] or ""),
            "conversation_updated_at": int(row["updated_at"] or 0),
            "message_ordinal": message_index + 1,
            "chunk_ordinal": int(row["chunk_index"]) + 1,
            "role": str(row["role"] or "unknown"),
            "score": round(score, 4),
            "matched_terms": matched_terms,
            "matched_by": ["bm25"],
            "retrieval_scores": {"bm25": round(score, 4)},
            "content": parent_content,
            "matched_content": matched_content,
            "parent_id": str(row["parent_id"] or ""),
            "source_kind": str(row["source_kind"] or ""),
            "return_mode": "parent_episode",
        }
        ranked.append((score, payload))
        per_message[message_key] = per_message.get(message_key, 0) + 1
    ranked.sort(
        key=lambda item: (
            -item[0],
            -int(item[1]["conversation_updated_at"]),
            -int(item[1]["message_ordinal"]),
        )
    )
    return [payload for _, payload in ranked]


def rank_memory_candidates(
    memories: list[dict[str, Any]],
    *,
    query: str,
    keywords: list[str],
    terms: list[str],
) -> list[dict[str, Any]]:
    normalized_phrases = [normalize_for_exact(value) for value in [query, *keywords] if value.strip()]
    ranked: list[tuple[float, dict[str, Any]]] = []
    for memory in memories:
        content = str(memory.get("content") or "")
        tokens = set(index_terms(content))
        matched_terms = [term for term in terms if term in tokens]
        if not matched_terms:
            continue
        coverage = len(matched_terms) / max(1, len(terms))
        normalized_content = normalize_for_exact(content)
        exact_hits = sum(
            1 for phrase in normalized_phrases if len(phrase) >= 2 and phrase in normalized_content
        )
        corrected_boost = 0.75 if str(memory.get("state") or "") == "corrected" else 0.0
        score = coverage * 8.0 + exact_hits * 5.0 + corrected_boost
        payload = {
            "memory_id": str(memory.get("id") or ""),
            "conversation_id": str(memory.get("conversation_id") or ""),
            "conversation_title": str(memory.get("conversation_title") or ""),
            "project_id": str(memory.get("project_id") or ""),
            "conversation_updated_at": int(memory.get("conversation_updated_at") or 0),
            "summary_message_count": int(memory.get("summary_message_count") or 0),
            "state": str(memory.get("state") or "automatic"),
            "score": round(score, 4),
            "matched_terms": matched_terms,
            "content": content,
        }
        ranked.append((score, payload))
    ranked.sort(
        key=lambda item: (
            -item[0],
            -int(item[1]["conversation_updated_at"]),
        )
    )
    return [payload for _, payload in ranked]


def normalize_for_exact(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "")).lower()
