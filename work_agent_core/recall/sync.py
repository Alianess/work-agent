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
DEFAULT_SKIP_DIRECTORIES = frozenset(
    {
        ".git",
        ".venv",
        "node_modules",
        "__pycache__",
        "tmp",
        # 以下是**本系统自己生成**的目录，不是用户的材料。排除它们不是猜测，
        # 是我们知道自己往哪里写东西——就像 AGENTS.md 里写 .venv 一样。
        # 实测不排除时，沙箱快照占了索引 82% 的节点：每次 shell_exec 复制一份
        # 工作区，同一段内容出现 882 次。
        "execution",       # 沙箱执行快照：工作区的逐次副本
        "file_previews",   # 已索引文件的生成预览
        "office_extracts", # 已索引 docx 的生成 Markdown
        "qwen3_denoise_trials",  # 降噪试验输出
        "users",           # 账户数据目录，索引本身就在里面
        "model_cache",
    }
)
"""不进索引的目录。

判据是"这里的东西是本系统写出来的，还是用户给的"。
`asr_full` 虽然也是我们写的，但里面是真实会议转写——那是内容，不是产物。
"""
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
                # 标题先留空，等第一条用户消息到了再取它的开头当标题。
                # turn id 当标题等于没有标题：一串 turn-1787123863871-cff1671e
                # 排在一起，目录就退化成一列看不懂的编号。
                "title": "",
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
        if not content:
            continue
        role = "user" if event.type == USER_MESSAGE else "assistant"
        if role == "user" and not current["title"]:
            current["title"] = turn_title(content)
        current["messages"].append({"role": role, "content": content})
    kept = [turn for turn in turns if turn["messages"]]
    for position, turn in enumerate(kept, start=1):
        if not turn["title"]:
            turn["title"] = f"第 {position} 轮"
    return kept


def turn_title(content: str) -> str:
    """用用户那句话的开头当轮标题。

    目录要一眼看得懂才叫目录。用户说的第一句话是这一轮最短、最准的概括，
    比任何自动生成的标题都便宜。
    """

    text = " ".join(str(content or "").split())
    text = text.split("【附件】")[0].strip()
    return text[:24] + ("…" if len(text) > 24 else "")


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

    def __init__(
        self,
        index: RecallIndex,
        *,
        aliases_for: Any | None = None,
        workspace_root: str | Path | None = None,
    ) -> None:
        self.index = index
        self.aliases_for = aliases_for
        # 来源标识相对它计算，这样同名文件不会互相覆盖，路径也保持可读。
        self.workspace_root = Path(workspace_root) if workspace_root else None

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
        tree = build_file_tree(resolved, source_id=source_id, relative_to=self.workspace_root)
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
        if self.workspace_root is None:
            self.workspace_root = base
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
                (str(row["text_hash"]), normalize(vector))
                for row, vector in zip(pending, vectors)
            ],
        )
        remaining = len(self.index.leaves_without_vectors(embedding.model, limit=1))
        return {"pending": len(pending), "written": written, "more": bool(remaining)}

    def vector_debt(self, model: str) -> int:
        """还有多少种不同正文没有向量。用来判断补算是否跟得上。"""

        return self.index.vector_coverage(model)["remaining"]


def _leaf_text(row: dict[str, Any]) -> str:
    """喂给 embedding 的文本：路径 + 表头 + 正文。

    路径要进去——"（二）建设中试基地"这样的标题本身就是强信号，而正文里
    往往不再重复它。
    """

    from .nodes import split_path

    parts = [" / ".join(split_path(str(row.get("path") or ""))), str(row.get("header") or ""), str(row.get("text") or "")]
    return "\n".join(part for part in parts if part).strip() or " "
