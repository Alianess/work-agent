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
) -> None:
    def _recall(args: dict[str, Any]) -> str:
        query = str(args.get("query") or "").strip()
        if not query:
            raise ValueError("query 不能为空。")
        kinds = args.get("source_kinds") or []
        filters = NodeFilter(
            source_kinds=tuple(str(item) for item in kinds if str(item or "").strip()),
            since_ms=_parse_since(args.get("since")),
            until_ms=_parse_since(args.get("until")),
        )
        outcome = search(
            build_deps(data_root, graph=graph),
            query,
            filters=filters,
            entity_names=[str(item) for item in (args.get("entities") or []) if str(item or "").strip()],
            top_k=max(1, min(int(args.get("limit") or DEFAULT_TOP_K), 10)),
            recency_weight=float(args.get("recency_weight", 0.3)),
        )
        return json.dumps(outcome, ensure_ascii=False, indent=2)

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
                "Search everything this account has seen: workspace documents (Word, PDF, "
                "spreadsheets, Markdown), meeting transcripts, and past conversations. "
                "Use it when the answer depends on something said or written earlier and you "
                "do not already have it. Returns the smallest readable passage that matched, "
                "each with an 'expand' map naming the section, the parent section and the whole "
                "file together with the token cost of opening each — open one with recall_expand "
                "only when the passage is not enough. "
                "Not for questions about counts or current state (how many revisions, which "
                "version is latest): those are ledger questions, not search questions."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "自然语言问题，或当时出现过的专名、文件名、数字。",
                    },
                    "source_kinds": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["document", "chat", "transcript"]},
                        "description": "只搜某几类语料。留空搜全部。",
                    },
                    "since": {"type": "string", "description": "起始时间，YYYY-MM-DD 或毫秒时间戳。"},
                    "until": {"type": "string", "description": "截止时间，同上。"},
                    "entities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "限定到某些人、公司或项目。知识图谱未建时忽略。",
                    },
                    "limit": {"type": "integer", "default": DEFAULT_TOP_K},
                    "recency_weight": {
                        "type": "number",
                        "default": 0.3,
                        "description": "问“最新/上一版”时调高，问定义或背景时调低。",
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
            description=(
                "Open more context around a passage recall returned. Pass scope=up1/up2/file to "
                "climb the document structure, prev/next for the neighbouring passage, or "
                "max_tokens to get the largest whole section that fits a budget. The token cost "
                "of each option is already in the recall result, so choose before spending. "
                "A section over the limit comes back as an outline instead of text — pick a "
                "smaller one from it."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "recall 结果 expand 里的 id。"},
                    "scope": {
                        "type": "string",
                        "enum": ["up1", "up2", "up3", "file", "prev", "next"],
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "不给 scope 时按预算取最大的完整单元。",
                    },
                    "seen_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "已经在上文见过的片段 id，返回时会标出来。",
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
    on_error: Callable[[Exception], None] | None = None,
) -> threading.Thread:
    """轮末索引这次会话。

    放后台线程是因为**索引写入不许挡住回复**。词法索引本身只有几毫秒，但一份
    大会话的重新切片可能到几十毫秒，没有理由让用户等它。
    """

    def _run() -> None:
        try:
            report = sync_for(data_root).index_conversation(conversation_id, log, title=title)
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
