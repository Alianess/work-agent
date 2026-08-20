"""把一份材料、一段对话切成节点树。

切片的目标不是"块大小均匀"，而是**每个叶子单独拿出来仍然读得懂**。所以边界跟着
结构走：文档跟标题，表格跟行（且每层继承表头），对话跟轮次。按字数硬切会把一句话
劈成两半，命中之后模型看到半句，比没命中还糟。

标题识别包含中文公文的"一、/（一）/1./（1）"。这是**结构解析**，和把 "# " 认成标题
是同一件事，不是拿关键词猜语义。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence
import re

from .nodes import (
    SOURCE_CHAT,
    SOURCE_DOCUMENT,
    SOURCE_TRANSCRIPT,
    MemoryNode,
    MemoryTree,
    estimate_tokens,
    node_id,
)


LEAF_TARGET_TOKENS = 260
LEAF_MAX_TOKENS = 420
LEAF_MIN_TOKENS = 60
TABLE_ROWS_PER_LEAF = 12

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
# 中文公文的层级编号。顺序即层级。
_CN_HEADING_LEVELS: tuple[tuple[int, re.Pattern[str]], ...] = (
    (1, re.compile(r"^\s*([一二三四五六七八九十百]+)\s*[、．.]\s*(\S.*)$")),
    (2, re.compile(r"^\s*[（(]\s*([一二三四五六七八九十百]+)\s*[)）]\s*(\S.*)$")),
    (3, re.compile(r"^\s*(\d{1,2})\s*[、．.]\s*(\S.*)$")),
    (4, re.compile(r"^\s*[（(]\s*(\d{1,2})\s*[)）]\s*(\S.*)$")),
)
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_DIVIDER = re.compile(r"^\s*\|[\s:|-]+\|\s*$")


def detect_heading(line: str) -> tuple[int, str] | None:
    """返回 (层级, 标题)。Markdown 井号优先，其次中文编号。"""

    match = _MD_HEADING.match(line)
    if match:
        return len(match.group(1)), match.group(2).strip()
    for level, pattern in _CN_HEADING_LEVELS:
        found = pattern.match(line)
        if found:
            # 编号开头但正文很长的，是正文，不是标题。
            if estimate_tokens(found.group(2).strip()) <= 40:
                return level, line.strip()
    return None


def split_paragraphs(text: str) -> list[str]:
    """按空行分段，再把过长的段按句号切开，但绝不切在句子中间。"""

    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    parts: list[str] = []
    for block in blocks:
        if estimate_tokens(block) <= LEAF_MAX_TOKENS:
            parts.append(block)
            continue
        sentences = re.split(r"(?<=[。！？；!?;])\s*", block)
        buffer = ""
        for sentence in sentences:
            if not sentence.strip():
                continue
            candidate = f"{buffer}{sentence}"
            if buffer and estimate_tokens(candidate) > LEAF_TARGET_TOKENS:
                parts.append(buffer.strip())
                buffer = sentence
            else:
                buffer = candidate
        if buffer.strip():
            parts.append(buffer.strip())
    return parts


def _merge_small(parts: Sequence[str]) -> list[str]:
    """把过短的相邻段合并。一个二十字的叶子命中了也说明不了什么。"""

    merged: list[str] = []
    for part in parts:
        if merged and estimate_tokens(merged[-1]) < LEAF_MIN_TOKENS:
            candidate = f"{merged[-1]}\n\n{part}"
            if estimate_tokens(candidate) <= LEAF_MAX_TOKENS:
                merged[-1] = candidate
                continue
        merged.append(part)
    return merged


def split_segments(text: str) -> list[tuple[str, str, str]]:
    """按原文顺序切成 (类型, 表头, 正文) 段：table 或 text。

    顺序要保住——展开到父节点时读到的应当是原文的样子，而不是先表格后正文的
    重排版本。
    """

    segments: list[tuple[str, str, str]] = []
    lines = text.split("\n")
    buffer: list[str] = []
    index = 0

    def flush_text() -> None:
        body = "\n".join(buffer).strip()
        buffer.clear()
        if body:
            segments.append(("text", "", body))

    while index < len(lines):
        is_table_head = (
            _TABLE_ROW.match(lines[index])
            and index + 1 < len(lines)
            and _TABLE_DIVIDER.match(lines[index + 1])
        )
        if is_table_head:
            flush_text()
            header = f"{lines[index]}\n{lines[index + 1]}"
            rows: list[str] = []
            cursor = index + 2
            while cursor < len(lines) and _TABLE_ROW.match(lines[cursor]):
                rows.append(lines[cursor])
                cursor += 1
            segments.append(("table", header, "\n".join(rows)))
            index = cursor
            continue
        buffer.append(lines[index])
        index += 1
    flush_text()
    return segments


def build_document_tree(
    *,
    source_id: str,
    title: str,
    text: str,
    occurred_at: int = 0,
    meta: dict[str, Any] | None = None,
) -> MemoryTree:
    """文件 → 一级标题 → …… → 段落（叶子）。表格另切成行组，逐层带表头。"""

    tree = MemoryTree(source_id=source_id, source_kind=SOURCE_DOCUMENT)
    root = tree.add(
        MemoryNode(
            id=node_id(source_id, [], 0),
            source_id=source_id,
            source_kind=SOURCE_DOCUMENT,
            title=title,
            path=(title,),
            occurred_at=occurred_at,
            meta=dict(meta or {}),
        )
    )

    # (level, node)；栈顶是当前所处的最深标题
    stack: list[tuple[int, MemoryNode]] = [(0, root)]
    buffer: list[str] = []
    ordinal = 0

    def flush() -> None:
        nonlocal buffer, ordinal
        body = "\n".join(buffer).strip()
        buffer = []
        if not body:
            return
        parent = stack[-1][1]
        for kind, header, segment in split_segments(body):
            if kind == "table":
                rows = segment.split("\n")
                for start in range(0, len(rows), TABLE_ROWS_PER_LEAF):
                    ordinal += 1
                    tree.add(
                        MemoryNode(
                            id=node_id(source_id, parent.path, ordinal),
                            source_id=source_id,
                            source_kind=SOURCE_DOCUMENT,
                            parent_id=parent.id,
                            path=parent.path,
                            text="\n".join(rows[start : start + TABLE_ROWS_PER_LEAF]),
                            header=header,
                            occurred_at=occurred_at,
                            is_leaf=True,
                        )
                    )
                continue
            for part in _merge_small(split_paragraphs(segment)):
                ordinal += 1
                tree.add(
                    MemoryNode(
                        id=node_id(source_id, parent.path, ordinal),
                        source_id=source_id,
                        source_kind=SOURCE_DOCUMENT,
                        parent_id=parent.id,
                        path=parent.path,
                        text=part,
                        occurred_at=occurred_at,
                        is_leaf=True,
                    )
                )

    for line in text.split("\n"):
        heading = detect_heading(line)
        if heading is None:
            buffer.append(line)
            continue
        flush()
        level, heading_title = heading
        # 文件里的第一个标题若就是文件标题，它就是根，不再多造一层同名节点。
        if len(tree.nodes) == 1 and heading_title == title:
            stack = [(level, root)]
            continue
        while len(stack) > 1 and stack[-1][0] >= level:
            stack.pop()
        parent = stack[-1][1]
        path = (*parent.path, heading_title)
        ordinal += 1
        section = tree.add(
            MemoryNode(
                id=node_id(source_id, path, 0),
                source_id=source_id,
                source_kind=SOURCE_DOCUMENT,
                parent_id=parent.id,
                title=heading_title,
                path=path,
                occurred_at=occurred_at,
            )
        )
        stack.append((level, section))
    flush()

    tree.fill_internal_text()
    return tree


def build_chat_tree(
    *,
    source_id: str,
    title: str,
    turns: Iterable[dict[str, Any]],
    source_kind: str = SOURCE_CHAT,
) -> MemoryTree:
    """会话 → 轮 → 消息块（叶子）。

    聊天的块不自足——满篇"那个文件""他说"。所以叶子的父亲必须是完整的一轮，
    展开一层就能拿回上下文，而不是拿到另外半句。
    """

    tree = MemoryTree(source_id=source_id, source_kind=source_kind)
    root = tree.add(
        MemoryNode(
            id=node_id(source_id, [], 0),
            source_id=source_id,
            source_kind=source_kind,
            title=title,
            path=(title,),
        )
    )
    ordinal = 0
    for turn_index, turn in enumerate(turns, start=1):
        turn_title = str(turn.get("title") or f"第 {turn_index} 轮")
        occurred_at = int(turn.get("occurred_at") or 0)
        path = (title, turn_title)
        ordinal += 1
        turn_node = tree.add(
            MemoryNode(
                id=node_id(source_id, path, 0),
                source_id=source_id,
                source_kind=source_kind,
                parent_id=root.id,
                title=turn_title,
                path=path,
                occurred_at=occurred_at,
            )
        )
        for message in turn.get("messages") or []:
            role = str(message.get("role") or "")
            content = str(message.get("content") or "").strip()
            if not content:
                continue
            speaker = {"user": "用户", "assistant": "助手"}.get(role, role or "系统")
            for part in _merge_small(split_paragraphs(content)):
                ordinal += 1
                tree.add(
                    MemoryNode(
                        id=node_id(source_id, path, ordinal),
                        source_id=source_id,
                        source_kind=source_kind,
                        parent_id=turn_node.id,
                        path=path,
                        text=f"{speaker}：{part}",
                        occurred_at=occurred_at,
                        is_leaf=True,
                        meta={"role": role},
                    )
                )
    tree.fill_internal_text()
    return tree


TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".log"}
OFFICE_SUFFIXES = {".docx", ".xlsx", ".xlsm", ".csv", ".tsv", ".pptx", ".pdf"}
INDEXABLE_SUFFIXES = TEXT_SUFFIXES | OFFICE_SUFFIXES


def file_to_text(path: Path) -> str:
    """把一份文件读成 Markdown。Office 与 PDF 复用既有的抽取器。"""

    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return path.read_text(encoding="utf-8", errors="replace")
    from ..office_processor import build_markdown, extract_document

    result = extract_document(path)
    return build_markdown(path, result, "index")


def build_file_tree(path: Path, *, source_id: str = "", occurred_at: int = 0) -> MemoryTree:
    resolved = Path(path)
    identifier = source_id or f"doc:{resolved.name}"
    stamp = occurred_at
    if not stamp:
        try:
            stamp = int(resolved.stat().st_mtime * 1000)
        except OSError:
            stamp = 0
    return build_document_tree(
        source_id=identifier,
        title=resolved.name,
        text=file_to_text(resolved),
        occurred_at=stamp,
        meta={"path": resolved.as_posix(), "suffix": resolved.suffix.lower()},
    )
