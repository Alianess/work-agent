from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable
import json
import math
import re
import sqlite3

from .cross_chat_memory import CrossChatMemoryStore
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
CHUNK_CHARS = 1800
CHUNK_OVERLAP = 180

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


class ChatHistoryRecall:
    """SQLite FTS5/BM25 retrieval over one account's raw conversation sessions."""

    def __init__(
        self,
        session_store: SessionStore,
        conversation_id: str,
        *,
        project_id: str = "",
    ) -> None:
        clean_id = sanitize_conversation_id(conversation_id)
        if not clean_id:
            raise ValueError("conversation_id is required")
        self.session_store = session_store
        self.conversation_id = clean_id
        self.project_id = str(project_id or "").strip()
        self.database_path = session_store.session_dir.parent / "history_search.sqlite3"

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
        if not terms:
            raise ValueError("没有提取到可检索的关键词，请换用更具体的名称、数字或短语。")
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
                    "retrieval": "sqlite-fts5-bm25",
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
            candidates = fetch_candidates(
                connection,
                conversation_id=self.conversation_id,
                project_id=self.project_id,
                scope=scope,
                terms=terms,
                current_message_limit=current_message_limit,
                candidate_limit=max(40, limit * 12),
            )

        ranked = rank_candidates(
            candidates,
            query=query,
            keywords=keywords,
            terms=terms,
            total_messages=max(1, len(session.messages)),
            current_conversation_id=self.conversation_id,
            current_project_id=self.project_id,
        )[:limit]
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
                "memory_results": memory_results,
                "results": ranked,
                "note": (
                    "memory_results 是可查看、纠正和删除的聊天摘要记忆；results 是可核对的聊天原文片段。两者都保留来源聊天。"
                    if memory_results or ranked
                    else "没有命中。请改用当时出现过的专名、数字、文件名或同义关键词重试。"
                ),
                "retrieval": "sqlite-fts5-bm25",
                "model_used": False,
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
                "Search manageable cross-chat summary memories and exact raw passages across the current account's saved chats without using another model. "
                "Inside a project, auto scope searches chats from that project only; outside projects, auto scope searches non-project chats in the account. "
                "Use when the user refers to something discussed earlier, the compressed summary lacks a detail, "
                "or an exact name, number, decision, wording, path, or prior correction must be recovered. "
                "Search with several distinctive keywords; retry with aliases or exact names when needed. "
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
        "\n\n跨聊天记忆与原文回想能力：recall_chat_history 是只读 core 工具，不调用额外模型，"
        "会先返回由历史聊天摘要同步而来的可管理 memory_results，并同时返回可核对的原文 results。"
        "当用户说‘之前、当时、你还记得、我们讨论过’或需要核对较早的名称、数字、决定、原话、文件路径、纠错时，"
        "如果最近 messages 或压缩摘要不能可靠回答，必须先调用该工具。"
        "通常使用 scope=auto：项目聊天只检索同一项目，普通聊天检索账号内非项目聊天；项目专属记忆不会泄漏到项目外。"
        "需要限定当前聊天时使用 scope=current，需要专门找被摘要覆盖的原文时使用 scope=compressed。"
        "优先用多个有区分度的专名、数字和短语检索；无结果时换同义词或别名重试一次。"
        "不得把未命中解释为用户从未说过。"
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
    signature = session_signature(session)
    row = connection.execute(
        "SELECT signature, message_count FROM history_index_meta WHERE conversation_id = ?", (session.id,)
    ).fetchone()
    if row is not None and str(row[0]) == signature:
        return
    indexed_count = int(row[1]) if row is not None else 0
    can_append = (
        row is not None
        and 0 <= indexed_count < len(session.messages)
        and str(row[0]) == messages_signature(session.messages[:indexed_count])
    )
    start_index = indexed_count if can_append else 0
    chunks = list(iter_session_chunks(session, start_index=start_index))
    with connection:
        if not can_append:
            connection.execute("DELETE FROM chat_history_fts WHERE conversation_id = ?", (session.id,))
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


def session_signature(session: ConversationSession) -> str:
    return messages_signature(session.messages)


def messages_signature(messages: Iterable[dict[str, Any]]) -> str:
    digest = sha256()
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
    return {
        "title": str(item.get("title") or session.metadata.get("title") or session.id),
        "project_id": str(
            item.get("projectId")
            or item.get("project_id")
            or session.metadata.get("project_id")
            or ""
        ),
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
) -> Iterable[HistoryChunk]:
    for message_index in range(max(0, start_index), len(session.messages)):
        message = session.messages[message_index]
        content = message_text(message).strip()
        if not content:
            continue
        role = str(message.get("role") or "unknown")
        for chunk_index, chunk in enumerate(chunk_text(content)):
            search_text = " ".join(index_terms(chunk))
            if search_text:
                yield HistoryChunk(message_index, chunk_index, role, chunk, search_text)


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


def chunk_text(text: str, *, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    value = str(text or "").strip()
    if not value:
        return []
    if len(value) <= size:
        return [value]
    chunks: list[str] = []
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
        chunks.append(value[start:end].strip())
        if end >= len(value):
            break
        start = max(start + 1, end - overlap)
    return [chunk for chunk in chunks if chunk]


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


def fetch_candidates(
    connection: sqlite3.Connection,
    *,
    conversation_id: str,
    project_id: str,
    scope: str,
    terms: list[str],
    current_message_limit: int,
    candidate_limit: int,
) -> list[sqlite3.Row]:
    expression = " OR ".join(fts_quote(term) for term in terms)
    conditions = ["chat_history_fts MATCH ?"]
    parameters: list[Any] = [expression]
    if scope in {"compressed", "current"}:
        conditions.extend(
            [
                "chat_history_fts.conversation_id = ?",
                "CAST(chat_history_fts.message_index AS INTEGER) < ?",
            ]
        )
        parameters.extend([conversation_id, current_message_limit])
    else:
        if scope == "project":
            conditions.append("meta.project_id = ?")
            parameters.append(project_id)
        elif scope == "account":
            conditions.append("meta.project_id = ''")
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
                   meta.title AS conversation_title,
                   meta.project_id,
                   meta.updated_at,
                   bm25(chat_history_fts) AS bm25_score
            FROM chat_history_fts
            JOIN history_conversation_meta AS meta
              ON meta.conversation_id = chat_history_fts.conversation_id
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
        content = str(row["content"] or "")
        normalized_content = normalize_for_exact(content)
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
        score = coverage * 8.0 + exact_hits * 5.0 + lexical + recency * 0.2 + current_boost + project_boost
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
            "content": content,
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
