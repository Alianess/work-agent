"""把检索挂成工具，并在轮末让索引跟上。

检索是工具而不是每轮预处理：大多数轮次用不到它，每轮无条件跑一次就是每轮
无条件付费。模型自己判断要不要找——和技能懒加载是同一个道理。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
import json
import os
import threading

from ..tools import Tool, ToolRegistry
from .backends import SiliconFlowEmbedding, SiliconFlowRerank
from .index import NodeFilter, RecallIndex
from .search import DEFAULT_TOP_K, RecallDeps, expand, search
from .sync import RecallSync


RECALL_DATABASE_NAME = "recall.sqlite3"
RECALL_RESULT_MAX_TEXT_CHARS = 700
RECALL_TOOL_MAX_RESULTS = 6
RECALL_TOOL_MAX_CHARS = 12_000
_INDEX_CACHE: dict[str, RecallIndex] = {}
_INDEX_LOCK = threading.Lock()


def recall_index_for(data_root: str | Path) -> RecallIndex:
    """每个账户一个索引文件。隔离靠路径，和会话存储一致。"""

    path = Path(data_root) / "recall" / RECALL_DATABASE_NAME
    key = str(path)
    with _INDEX_LOCK:
        index = _INDEX_CACHE.get(key)
        if index is None:
            index = RecallIndex(path)
            _INDEX_CACHE[key] = index
        return index


def embedding_backend() -> SiliconFlowEmbedding | None:
    backend = SiliconFlowEmbedding.from_env()
    return backend if backend.available else None


def rerank_backend() -> SiliconFlowRerank | None:
    backend = SiliconFlowRerank.from_env()
    return backend if backend.available else None


def build_deps(data_root: str | Path, *, graph: Any | None = None) -> RecallDeps:
    deps = RecallDeps(
        index=recall_index_for(data_root),
        embedding=embedding_backend(),
        rerank=rerank_backend(),
    )
    if graph is not None:
        deps.graph = graph
    return deps


def _parse_since(value: Any) -> int:
    """接受毫秒时间戳或 'YYYY-MM-DD'。时间是结构化条件，不该让模型描述它。"""

    text = str(value or "").strip()
    if not text:
        return 0
    if text.isdigit():
        return int(text)
    from datetime import datetime

    for pattern in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%d %H:%M"):
        try:
            return int(datetime.strptime(text, pattern).timestamp() * 1000)
        except ValueError:
            continue
    return 0


def register_recall_tools(
    registry: ToolRegistry,
    data_root: str | Path,
    *,
    graph: Any | None = None,
    project_id: str = "",
    session_store: Any | None = None,
    conversation_id: str = "",
) -> None:
    def _current_conversation_is_fully_replayed() -> bool:
        if session_store is None or not conversation_id:
            return False
        try:
            session = session_store.load(conversation_id)
        except Exception:
            return False
        return not str(getattr(session, "summary", "") or "").strip() and int(
            getattr(session, "summary_message_count", 0) or 0
        ) == 0

    def _compact_outcome(outcome: dict[str, Any]) -> dict[str, Any]:
        compacted = dict(outcome)
        compact_results: list[dict[str, Any]] = []
        for raw_item in list(outcome.get("results") or [])[:RECALL_TOOL_MAX_RESULTS]:
            if not isinstance(raw_item, dict):
                continue
            item = dict(raw_item)
            text = str(item.get("text") or "")
            item["text_chars"] = len(text)
            if len(text) > RECALL_RESULT_MAX_TEXT_CHARS:
                item["text"] = text[:RECALL_RESULT_MAX_TEXT_CHARS].rstrip() + "…"
                item["text_truncated"] = True
            compact_results.append(item)
        compacted["results"] = compact_results
        if isinstance(compacted.get("memory_results"), list):
            compacted["memory_results"] = compacted["memory_results"][:5]
        compacted["note"] = (
            "results 是短片段，已在工具边界限长；不够用时用 recall_expand "
            "按 expand 里的 id 展开，tokens 是展开代价。"
        )
        encoded = json.dumps(compacted, ensure_ascii=False, indent=2)
        while len(encoded) > RECALL_TOOL_MAX_CHARS and len(compact_results) > 1:
            compact_results.pop()
            retrieval = compacted.setdefault("retrieval", {})
            if isinstance(retrieval, dict):
                retrieval["results_trimmed_for_output_budget"] = True
            encoded = json.dumps(compacted, ensure_ascii=False, indent=2)
        return compacted

    def _recall(args: dict[str, Any]) -> str:
        query = str(args.get("query") or "").strip()
        if not query:
            raise ValueError("query 不能为空。")
        scope = str(args.get("scope") or "all").strip()
        scope_note = ""
        if scope == "project" and not project_id:
            # 08-25~28 连续踩了 11 次：会话不在项目里时 scope=project 直接报错，
            # 模型每轮都要重选。降级到账户级并在结果里说明，比抛错有用。
            scope = "all"
            scope_note = "当前会话不在任何项目里，scope=project 已自动降级为 scope=all。"
        kinds = args.get("source_kinds") or []
        filters = NodeFilter(
            source_kinds=tuple(str(item) for item in kinds if str(item or "").strip()),
            excluded_source_ids=(f"chat:{conversation_id}",)
            if _current_conversation_is_fully_replayed()
            else (),
            since_ms=_parse_since(args.get("since")),
            until_ms=_parse_since(args.get("until")),
            project_ids=(project_id,) if scope == "project" else (),
            include_superseded=bool(args.get("include_superseded")),
        )
        outcome = search(
            build_deps(data_root, graph=graph),
            query,
            filters=filters,
            entity_names=[str(item) for item in (args.get("entities") or []) if str(item or "").strip()],
            top_k=max(1, min(int(args.get("limit") or DEFAULT_TOP_K), RECALL_TOOL_MAX_RESULTS)),
            recency_weight=float(args.get("recency_weight", 0.3)),
        )
        if scope == "project":
            outcome["scope"] = f"project:{project_id}"
        if scope_note:
            outcome["scope_note"] = scope_note
        # 核心记忆跟原文一起给：模型既然主动检索了，相关的记忆条目就不该
        # 再等它另开一问。账户级始终在场；scope=project 时叠加项目级。
        if session_store is not None:
            try:
                from ..cross_chat_memory import CrossChatMemoryStore, rank_memories_for_query

                memory_store = CrossChatMemoryStore(session_store)
                memories = memory_store.list(project_id="", limit=100)
                if project_id:
                    memories += memory_store.list(project_id=project_id, limit=100)
                memory_results = rank_memories_for_query(memories, query)
                if memory_results:
                    outcome["memory_results"] = memory_results
            except Exception:
                pass  # 记忆附带是加菜，失败不该殃及检索本身
        return json.dumps(_compact_outcome(outcome), ensure_ascii=False, indent=2)

    def _expand(args: dict[str, Any]) -> str:
        node_id_value = str(args.get("id") or "").strip()
        if not node_id_value:
            raise ValueError("id 不能为空，用 recall 结果里 expand 给出的 id。")
        outcome = expand(
            recall_index_for(data_root),
            node_id_value,
            scope=str(args.get("scope") or "").strip(),
            max_tokens=int(args.get("max_tokens") or 0),
            seen_ids=[str(item) for item in (args.get("seen_ids") or [])],
        )
        return json.dumps(outcome, ensure_ascii=False, indent=2)

    registry.register(
        Tool(
            name="recall",
            description=(
                "Search indexed documents, meeting transcripts, and past chats. "
                "Returns matching passages with ids for recall_expand."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Question or remembered keywords.",
                    },
                    "source_kinds": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["document", "chat", "transcript"]},
                    },
                    "scope": {
                        "type": "string",
                        "enum": ["all", "project"] if project_id else ["all"],
                        "default": "all",
                        "description": (
                            "project=current project; all=account."
                            if project_id
                            else "This conversation has no project; scope=project is auto-downgraded to all."
                        ),
                    },
                    "include_superseded": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include older document versions.",
                    },
                    "since": {"type": "string", "description": "YYYY-MM-DD or Unix milliseconds."},
                    "until": {"type": "string", "description": "YYYY-MM-DD or Unix milliseconds."},
                    "entities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "People, companies, or projects.",
                    },
                    "limit": {"type": "integer", "default": DEFAULT_TOP_K},
                    "recency_weight": {
                        "type": "number",
                        "default": 0.3,
                        "description": "Higher favors recent results.",
                    },
                },
                "required": ["query"],
            },
            handler=_recall,
        )
    )
    registry.register(
        Tool(
            name="recall_expand",
            description="Expand a recall result by hierarchy, neighbor, or token budget.",
            parameters={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Id from a recall result."},
                    "scope": {
                        "type": "string",
                        "enum": ["up1", "up2", "up3", "file", "prev", "next"],
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Largest complete unit when scope is omitted.",
                    },
                    "seen_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Ids already shown in context.",
                    },
                },
                "required": ["id"],
            },
            handler=_expand,
        )
    )


# ---------------------------------------------------------------------------
# 索引维护
# ---------------------------------------------------------------------------


def sync_for(data_root: str | Path, *, workspace_root: str | Path | None = None) -> RecallSync:
    """索引住在账户数据目录，语料来自工作区——两个根不是一回事。

    混同的代价很实在：admin 的 account_workspace_root() 是仓库根，索引就被建到
    了仓库根下的一个空库里，而真正的语料库在账户目录下。工具查的和写的不是同
    一个索引，检索永远是空的。
    """

    return RecallSync(
        recall_index_for(data_root), workspace_root=workspace_root or data_root
    )


def index_conversation_async(
    data_root: str | Path,
    conversation_id: str,
    log: Any,
    *,
    title: str = "",
    project_id: str = "",
    on_error: Callable[[Exception], None] | None = None,
) -> threading.Thread:
    """轮末索引这次会话。

    放后台线程是因为**索引写入不许挡住回复**。词法索引本身只有几毫秒，但一份
    大会话的重新切片可能到几十毫秒，没有理由让用户等它。
    """

    def _run() -> None:
        try:
            report = sync_for(data_root).index_conversation(
                conversation_id, log, title=title, project_id=project_id
            )
            # 词法索引完成后顺手把这一轮的新窗口补上向量。一轮只新增几个窗口，
            # 一次调用就够；等调度线程 60 秒后再来，这段时间稠密召回是缺的。
            if report.touched:
                backfill_vectors_once(data_root, budget=32)
        except Exception as error:  # pragma: no cover - 后台维护不得影响主路径
            if on_error is not None:
                on_error(error)

    thread = threading.Thread(target=_run, name="recall-index", daemon=True)
    thread.start()
    return thread


def index_file_async(
    data_root: str | Path,
    path: str | Path,
    *,
    workspace_root: str | Path | None = None,
    on_error: Callable[[Exception], None] | None = None,
) -> threading.Thread | None:
    """文件写入或更新后立刻入索引。同样放后台，不挡住写文件的那一步。"""

    def _run() -> None:
        try:
            report = sync_for(data_root, workspace_root=workspace_root).index_file(path)
            if report.touched:
                backfill_vectors_once(data_root, budget=32)
        except Exception as error:  # pragma: no cover - 后台维护不得影响主路径
            if on_error is not None:
                on_error(error)

    thread = threading.Thread(target=_run, name="recall-index-file", daemon=True)
    thread.start()
    return thread


def backfill_vectors_once(data_root: str | Path, *, budget: int = 64) -> dict[str, Any]:
    """补算一批向量。没有配置 embedding 后端时什么都不做。"""

    backend = embedding_backend()
    if backend is None:
        return {"skipped": "未配置 SILICONFLOW_API_KEY"}
    return sync_for(data_root).backfill_vectors(backend, budget=budget)


def recall_status(data_root: str | Path) -> dict[str, Any]:
    index = recall_index_for(data_root)
    backend = embedding_backend()
    status: dict[str, Any] = {"index_path": str(index.database_path), **index.stats()}
    status["embedding"] = backend.model if backend is not None else ""
    status["rerank"] = (rerank_backend() or SiliconFlowRerank()).model if rerank_backend() else ""
    if backend is not None:
        status["vector_debt"] = sync_for(data_root).vector_debt(backend.model)
    return status
