"""让索引跟上语料的增长。

两条不变量决定了这里的形状：

1. **索引写入不许挡住回复。** 词法索引是本地的、几毫秒的事，可以在一轮结束时
   同步做；向量要走网络，必须分开成后台补算。所以新内容立刻能被 BM25 找到，
   稠密召回随后追上——而端点抽风时，最坏结果只是"这段暂时没有向量"。
2. **增量到节点。** 一次对话每加一轮就整份重算向量，一天下来是几十次重复调用。
   `upsert_tree` 按节点求差，没变的节点连同它的向量原地保留。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence
import time

from ..session_log import (
    ASSISTANT_MESSAGE,
    SessionLog,
    TURN_START,
    USER_MESSAGE,
)
from .backends import RecallBackendError, normalize
from .chunking import INDEXABLE_SUFFIXES, build_chat_tree, build_file_tree
from .index import RecallIndex, UpsertReport
from .nodes import SOURCE_CHAT


VECTOR_BATCH = 64
DEFAULT_SKIP_DIRECTORIES = frozenset({".git", ".venv", "node_modules", "__pycache__", "tmp"})
MAX_FILE_BYTES = 20 * 1024 * 1024


def turns_from_log(log: SessionLog) -> list[dict[str, Any]]:
    """把事件日志摊成"轮"。

    聊天块不自足，所以叶子的父必须是完整一轮——这里的边界就是 turn/start。
    """

    turns: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for event in log.events:
        if event.type == TURN_START:
            current = {
                "title": str(dict(event.data).get("turn_id") or f"第 {len(turns) + 1} 轮"),
                "occurred_at": int(event.ts_ms or 0),
                "messages": [],
            }
            turns.append(current)
            continue
        if event.type not in {USER_MESSAGE, ASSISTANT_MESSAGE}:
            continue
        if current is None:
            current = {"title": "第 1 轮", "occurred_at": int(event.ts_ms or 0), "messages": []}
            turns.append(current)
        content = str(dict(event.data).get("content") or "").strip()
        if content:
            current["messages"].append(
                {
                    "role": "user" if event.type == USER_MESSAGE else "assistant",
                    "content": content,
                }
            )
    return [turn for turn in turns if turn["messages"]]


@dataclass
class SyncReport:
    sources: int = 0
    added: int = 0
    updated: int = 0
    removed: int = 0
    unchanged: int = 0
    skipped: int = 0
    failures: list[str] = field(default_factory=list)

    def absorb(self, report: UpsertReport) -> None:
        if report.skipped:
            self.skipped += 1
            return
        self.sources += 1
        self.added += report.added
        self.updated += report.updated
        self.removed += report.removed
        self.unchanged += report.unchanged


class RecallSync:
    """索引维护：会话、文件、向量补算。"""

    def __init__(self, index: RecallIndex, *, aliases_for: Any | None = None) -> None:
        self.index = index
        self.aliases_for = aliases_for

    # ------------------------------------------------------------------
    # 会话
    # ------------------------------------------------------------------

    def index_conversation(
        self,
        conversation_id: str,
        log: SessionLog,
        *,
        title: str = "",
    ) -> UpsertReport:
        """一轮结束时调用。只做本地词法索引，不碰网络。"""

        turns = turns_from_log(log)
        if not turns:
            return UpsertReport(skipped=True)
        tree = build_chat_tree(
            source_id=f"chat:{conversation_id}",
            title=title or conversation_id,
            turns=turns,
            source_kind=SOURCE_CHAT,
        )
        root = tree.root()
        if root is not None and not root.occurred_at:
            root.occurred_at = max(int(turn["occurred_at"] or 0) for turn in turns)
        return self.index.upsert_tree(
            tree, uri=f"conversation/{conversation_id}", aliases_for=self.aliases_for
        )

    # ------------------------------------------------------------------
    # 文件
    # ------------------------------------------------------------------

    def index_file(self, path: str | Path, *, source_id: str = "") -> UpsertReport:
        resolved = Path(path)
        if resolved.suffix.lower() not in INDEXABLE_SUFFIXES:
            return UpsertReport(skipped=True)
        try:
            if resolved.stat().st_size > MAX_FILE_BYTES:
                return UpsertReport(skipped=True)
        except OSError:
            return UpsertReport(skipped=True)
        tree = build_file_tree(resolved, source_id=source_id)
        return self.index.upsert_tree(
            tree, uri=resolved.as_posix(), aliases_for=self.aliases_for
        )

    def index_directory(
        self,
        root: str | Path,
        *,
        skip_directories: Iterable[str] = DEFAULT_SKIP_DIRECTORIES,
        limit: int = 0,
    ) -> SyncReport:
        """把一个目录整体纳入索引。首次建库用，之后靠文件变更回调增量维护。"""

        report = SyncReport()
        skip = set(skip_directories)
        base = Path(root)
        seen = 0
        for path in sorted(base.rglob("*")):
            if limit and seen >= limit:
                break
            if not path.is_file() or path.suffix.lower() not in INDEXABLE_SUFFIXES:
                continue
            if any(part in skip or part.startswith(".") for part in path.relative_to(base).parts[:-1]):
                continue
            seen += 1
            try:
                report.absorb(self.index_file(path))
            except Exception as error:
                report.failures.append(f"{path.name}: {type(error).__name__}: {error}")
        return report

    # ------------------------------------------------------------------
    # 向量：走网络，所以永远是后台的第二步
    # ------------------------------------------------------------------

    def backfill_vectors(self, embedding: Any, *, budget: int = VECTOR_BATCH) -> dict[str, Any]:
        """给还没有向量的叶子补算。

        失败不抛：这一批没算成，下一次再来。词法召回一直是可用的，所以补算落后
        只是稠密召回暂时少几段，不是检索坏了。
        """

        pending = self.index.leaves_without_vectors(embedding.model, limit=budget)
        if not pending:
            return {"pending": 0, "written": 0}
        texts = [_leaf_text(row) for row in pending]
        try:
            vectors = embedding.embed(texts)
        except RecallBackendError as error:
            return {"pending": len(pending), "written": 0, "error": str(error)}
        except Exception as error:  # pragma: no cover - 后端异常形态不可穷举
            return {"pending": len(pending), "written": 0, "error": f"{type(error).__name__}: {error}"}
        written = self.index.store_vectors(
            embedding.model,
            [
                (str(row["id"]), normalize(vector))
                for row, vector in zip(pending, vectors)
            ],
        )
        remaining = len(self.index.leaves_without_vectors(embedding.model, limit=1))
        return {"pending": len(pending), "written": written, "more": bool(remaining)}

    def vector_debt(self, model: str) -> int:
        """还有多少叶子没有向量。用来判断补算是否跟得上。"""

        return len(self.index.leaves_without_vectors(model, limit=100_000))


def _leaf_text(row: dict[str, Any]) -> str:
    """喂给 embedding 的文本：路径 + 表头 + 正文。

    路径要进去——"（二）建设中试基地"这样的标题本身就是强信号，而正文里
    往往不再重复它。
    """

    from .nodes import split_path

    parts = [" / ".join(split_path(str(row.get("path") or ""))), str(row.get("header") or ""), str(row.get("text") or "")]
    return "\n".join(part for part in parts if part).strip() or " "
