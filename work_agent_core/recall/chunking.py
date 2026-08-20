"""把一份材料、一段对话切成节点树。

切片的目标不是"块大小均匀"，而是**每个叶子单独拿出来仍然读得懂**。所以边界跟着
结构走：文档跟标题，表格跟行（且每层继承表头），对话跟轮次。按字数硬切会把一句话
劈成两半，命中之后模型看到半句，比没命中还糟。

标题识别包含中文公文的"一、/（一）/1./（1）"。这是**结构解析**，和把 "# " 认成标题
是同一件事，不是拿关键词猜语义。
"""

from __future__ import annotations

from dataclasses import dataclass
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


# 三层，各管各的事：
#
#   章节 section   跟着标题走，可能上万 token —— 用来定位"在文件的哪一部分"
#   展示段 passage 约 500t，不重叠     —— 命中后返回给模型的就是它
#   匹配窗口 window 约 200t，15% 重叠  —— 只进索引，从不直接返回
#
# 窗口切小是为了命中准，展示段切大是为了读得完整。两件事分开，谁也不必迁就谁。
# 参数对齐 V5_dev 的实战值（其检索拿过研点赛全国第一），单位换成 token：
# 目标 160 units ≈ 中文 160 字，实测子块中位 223 字符。
WINDOW_TARGET_TOKENS = 200
WINDOW_OVERLAP_TOKENS = 30
"""相邻叶子重叠约 15%。

不重叠的代价很具体：答案跨在两段之间时会被切成两半，命中哪一半都不完整。
有重叠则至少有一块完整包含它。这是滑动窗口存在的唯一理由，也是它值得付出的
索引膨胀（约 15%）。
"""

WINDOW_MIN_TOKENS = 70
"""短尾并入前一块。一个二十字的窗口命中了也说明不了什么。"""

WINDOW_MAX_TOKENS = 300
WINDOW_WHOLE_MAX_TOKENS = 200
"""整段不超过目标就整块保留，不为了切而切。"""

PASSAGE_TARGET_TOKENS = 500
PASSAGE_MAX_TOKENS = 700
"""展示单元的大小。

匹配和展示是两件事：窗口切小是为了命中准，返回给模型的却该是一段读得完整的
正文。所以命中发生在窗口上，返回的是包住它的这一段——V5_dev 的
RETURN_PARENT_SECTION 就是这个意思。
"""

TABLE_TOKENS_PER_WINDOW = 260
"""表格按 token 预算切，不按行数。一行长短差几十倍，按行数切必然爆块。"""

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


# 切分点的优先级：先在语义最强的边界上切，退无可退才动子句。
# 空行 > 句末标点 > 冒号 > 单换行 > 逗号（只有超长无标点句才用到最后一档）。
_BOUNDARIES = (
    re.compile(r"\n\s*\n"),
    re.compile(r"(?<=[。！？!?；;])\s*"),
    re.compile(r"(?<=[：:])\s+"),
    re.compile(r"\n"),
)
_CLAUSE_BOUNDARY = re.compile(r"(?<=[，,、])\s*")


def _spans(text: str) -> list[tuple[int, int]]:
    """把文本切成"最小不可分单元"的区间。

    保留下标而不是切成字符串：叶子要记住自己在原文的位置，邻居、高亮和以后的
    引用定位都要用它。
    """

    units: list[tuple[int, int]] = [(0, len(text))]
    for pattern in _BOUNDARIES:
        nxt: list[tuple[int, int]] = []
        for start, end in units:
            if estimate_tokens(text[start:end]) <= WINDOW_MAX_TOKENS:
                nxt.append((start, end))
                continue
            cursor = start
            for match in pattern.finditer(text, start, end):
                if match.end() <= cursor:
                    continue
                nxt.append((cursor, match.end()))
                cursor = match.end()
            if cursor < end:
                nxt.append((cursor, end))
        units = nxt

    # 仍然超长的（没有标点的长句）按子句切，再不行按字符硬切——但这已是最后一档。
    final: list[tuple[int, int]] = []
    for start, end in units:
        if estimate_tokens(text[start:end]) <= WINDOW_MAX_TOKENS:
            final.append((start, end))
            continue
        cursor = start
        for match in _CLAUSE_BOUNDARY.finditer(text, start, end):
            if match.end() > cursor:
                final.append((cursor, match.end()))
                cursor = match.end()
        while end - cursor > WINDOW_MAX_TOKENS:
            final.append((cursor, cursor + WINDOW_MAX_TOKENS))
            cursor += WINDOW_MAX_TOKENS
        if cursor < end:
            final.append((cursor, end))
    return [(start, end) for start, end in final if text[start:end].strip()]


@dataclass(frozen=True)
class Window:
    start: int
    end: int
    text: str


def _table_groups(rows: list[str]) -> list[list[str]]:
    """按 token 预算分组表格行，不按行数。

    一行的长短能差几十倍：按固定行数切，遇到宽表必然切出几百 token 的巨块。
    单行本身超预算时独占一组——表格的行是不可分的最小语义单位。
    """

    groups: list[list[str]] = []
    current: list[str] = []
    used = 0
    for row in rows:
        size = estimate_tokens(row)
        if current and used + size > TABLE_TOKENS_PER_WINDOW:
            groups.append(current)
            current, used = [], 0
        current.append(row)
        used += size
    if current:
        groups.append(current)
    return groups


def split_passages(text: str) -> list[Window]:
    """把一段正文切成**展示单元**：不重叠、边界对齐、约 500 token。

    这一层是返回给模型的东西，所以要求是"读得完整"而不是"匹配得准"。
    不重叠，否则展开时会重复。
    """

    body = text.strip()
    if not body:
        return []
    if estimate_tokens(body) <= PASSAGE_MAX_TOKENS:
        offset = text.index(body)
        return [Window(offset, offset + len(body), body)]

    spans = _spans(text)
    passages: list[Window] = []
    index = 0
    total = len(spans)
    sizes = [estimate_tokens(text[start:end]) for start, end in spans]
    while index < total:
        used = 0
        cursor = index
        while cursor < total and (used < PASSAGE_TARGET_TOKENS or cursor == index):
            used += sizes[cursor]
            cursor += 1
        if 0 < sum(sizes[cursor:]) < WINDOW_MIN_TOKENS:
            cursor = total
        start, end = spans[index][0], spans[cursor - 1][1]
        chunk = text[start:end].strip()
        if chunk:
            offset = start + (len(text[start:end]) - len(text[start:end].lstrip()))
            passages.append(Window(offset, offset + len(chunk), chunk))
        index = cursor
    return passages


def sliding_windows(text: str) -> list[Window]:
    """句子感知的滑动窗口：组装到目标大小，相邻窗口重叠，切点只落在边界上。

    和"按段落切"的区别是主动性：段落多长块就多长，块大小完全随原文摆布；
    滑动窗口把大小控制在目标附近，稠密召回的向量质量才稳定。
    """

    body = text.strip()
    if not body:
        return []
    if estimate_tokens(body) <= WINDOW_WHOLE_MAX_TOKENS:
        # 整节不超目标就不切。为了切而切只会让每一块都读不完整。
        offset = text.index(body)
        return [Window(offset, offset + len(body), body)]

    spans = _spans(text)
    if not spans:
        return []
    sizes = [estimate_tokens(text[start:end]) for start, end in spans]

    windows: list[Window] = []
    index = 0
    total = len(spans)
    while index < total:
        used = 0
        cursor = index
        while cursor < total and (used < WINDOW_TARGET_TOKENS or cursor == index):
            used += sizes[cursor]
            cursor += 1
        # 防短尾：剩下的凑不满一个最小块就并进来，不留孤立短句。
        if 0 < sum(sizes[cursor:]) < WINDOW_MIN_TOKENS:
            cursor = total
        start = spans[index][0]
        end = spans[cursor - 1][1]
        chunk = text[start:end].strip()
        if chunk:
            offset = start + (len(text[start:end]) - len(text[start:end].lstrip()))
            windows.append(Window(offset, offset + len(chunk), chunk))
        if cursor >= total:
            break
        # 从块尾回退 overlap 个 token 作为下一块起点。
        back = 0
        step = cursor
        while step - 1 > index and back < WINDOW_OVERLAP_TOKENS:
            step -= 1
            back += sizes[step]
        index = step
    return windows


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
                for group in _table_groups(segment.split("\n")):
                    ordinal += 1
                    tree.add(
                        MemoryNode(
                            id=node_id(source_id, parent.path, ordinal),
                            source_id=source_id,
                            source_kind=SOURCE_DOCUMENT,
                            parent_id=parent.id,
                            path=parent.path,
                            text="\n".join(group),
                            header=header,
                            occurred_at=occurred_at,
                            is_leaf=True,
                        )
                    )
                continue
            for passage in split_passages(segment):
                ordinal += 1
                passage_node = tree.add(
                    MemoryNode(
                        id=node_id(source_id, parent.path, ordinal),
                        source_id=source_id,
                        source_kind=SOURCE_DOCUMENT,
                        parent_id=parent.id,
                        path=parent.path,
                        text=passage.text,
                        occurred_at=occurred_at,
                        is_leaf=False,
                        meta={"char_start": passage.start, "char_end": passage.end},
                    )
                )
                for window in sliding_windows(passage.text):
                    ordinal += 1
                    tree.add(
                        MemoryNode(
                            id=node_id(source_id, parent.path, ordinal),
                            source_id=source_id,
                            source_kind=SOURCE_DOCUMENT,
                            parent_id=passage_node.id,
                            path=parent.path,
                            text=window.text,
                            occurred_at=occurred_at,
                            is_leaf=True,
                            meta={
                                "char_start": passage.start + window.start,
                                "char_end": passage.start + window.end,
                            },
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
            for passage in split_passages(content):
                ordinal += 1
                passage_node = tree.add(
                    MemoryNode(
                        id=node_id(source_id, path, ordinal),
                        source_id=source_id,
                        source_kind=source_kind,
                        parent_id=turn_node.id,
                        path=path,
                        title=f"{speaker}发言",
                        text=f"{speaker}：{passage.text}",
                        occurred_at=occurred_at,
                        is_leaf=False,
                        meta={"role": role},
                    )
                )
                for window in sliding_windows(passage.text):
                    ordinal += 1
                    tree.add(
                        MemoryNode(
                            id=node_id(source_id, path, ordinal),
                            source_id=source_id,
                            source_kind=source_kind,
                            parent_id=passage_node.id,
                            path=path,
                            text=f"{speaker}：{window.text}",
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


def file_source_key(path: Path, relative_to: Path | None = None) -> str:
    """来源标识用**路径**，不是文件名。

    一个工作区里同名文件遍地都是（"会议纪要.docx" 每个目录一份）。用文件名当
    来源，它们会互相覆盖：实测 5002 份材料入库后只剩 559 份。
    """

    resolved = Path(path)
    if relative_to is not None:
        try:
            return resolved.resolve().relative_to(Path(relative_to).resolve()).as_posix()
        except ValueError:
            pass
    return resolved.resolve().as_posix()


def file_to_text(path: Path) -> str:
    """把一份文件读成 Markdown。Office 与 PDF 复用既有的抽取器。

    不走 `build_markdown`：它会在正文前加一段"文件类型/处理方式/paragraphs/tables"
    的元信息，那是给人看转换结果用的。索引进去会变成一个纯噪音的叶子，还会在
    检索里跟真正的内容抢名次。
    """

    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return path.read_text(encoding="utf-8", errors="replace")
    from ..office_processor import extract_document

    result = extract_document(path)
    sections = [
        str(section).strip()
        for section in (result.get("sections") or [])
        if str(section).strip()
    ]
    return "\n\n".join(sections)


def build_file_tree(
    path: Path,
    *,
    source_id: str = "",
    occurred_at: int = 0,
    relative_to: Path | None = None,
) -> MemoryTree:
    resolved = Path(path)
    identifier = source_id or f"doc:{file_source_key(resolved, relative_to)}"
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
