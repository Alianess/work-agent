"""记忆语料的层级节点：只索引叶子，靠展开补语境。

一段材料、一次会议、一段对话，都是树。命中发生在叶子上——那是能被精确匹配的
最小单位；语境完整性靠向上展开取得。两件事分开，检索精度和证据完整就不必互相
让步。

节点 id 由来源加路径的内容哈希构成，不用行号或自增序号：重建一次索引之后，
上一轮交给模型的 id 仍然有效，否则展开会指向别的东西。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator
import hashlib
import re


# 语料类型。不同语料的树形不同，但都是树。
SOURCE_DOCUMENT = "document"
SOURCE_CHAT = "chat"
SOURCE_TRANSCRIPT = "transcript"
SOURCE_KINDS = frozenset({SOURCE_DOCUMENT, SOURCE_CHAT, SOURCE_TRANSCRIPT})

# 路径在库里用单元分隔符拼接：标题里可以有空格、斜杠、竖线，不能有它。
PATH_SEPARATOR = "\x1f"


def join_path(path: "Iterable[str]") -> str:
    return PATH_SEPARATOR.join(str(item) for item in path)


def split_path(value: str) -> list[str]:
    return [part for part in str(value or "").split(PATH_SEPARATOR) if part]

_CJK = re.compile("[　-〿㐀-䶿一-鿿＀-￯]")


def estimate_tokens(text: str) -> int:
    """给展开选项标价用的粗估。

    模型要用它决定"值不值得往上开一层"，所以宁可粗糙也要稳定：中文按一字一
    token，其余按四字符一 token。偏差在同一量级内，不影响那个决定。
    """

    if not text:
        return 0
    cjk = len(_CJK.findall(text))
    return max(1, cjk + (len(text) - cjk + 3) // 4)


def node_id(source_id: str, path: Iterable[str], ordinal: int = 0) -> str:
    """来源 + 路径 + 序号的内容哈希。重建索引后仍然指向同一处。"""

    material = " ".join([source_id, *[str(item) for item in path], str(ordinal)])
    digest = hashlib.blake2b(material.encode("utf-8"), digest_size=8).hexdigest()
    return f"{source_id}#{digest}"


@dataclass
class MemoryNode:
    id: str
    source_id: str
    source_kind: str
    parent_id: str = ""
    title: str = ""
    path: tuple[str, ...] = ()
    """祖先标题，从根到自己。进检索文本，也用来给模型看清位置。"""

    text: str = ""
    """本节点自身的正文。内部节点的正文由其后代拼接而成。"""

    occurred_at: int = 0
    """业务发生时间（毫秒）。排序必须能用上它——记忆里"最近的"几乎总是更相关。"""

    is_leaf: bool = False
    header: str = ""
    """表格类节点继承的表头。一行数据脱开表头没有意义，所以每层都带着它。"""

    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.searchable_body())

    def searchable_body(self) -> str:
        return f"{self.header}\n{self.text}".strip() if self.header else self.text

    def to_row(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "source_kind": self.source_kind,
            "parent_id": self.parent_id,
            "title": self.title,
            "path": join_path(self.path),
            "text": self.text,
            "header": self.header,
            "occurred_at": self.occurred_at,
            "is_leaf": 1 if self.is_leaf else 0,
            "tokens": self.tokens,
        }


@dataclass
class MemoryTree:
    """一个来源的完整节点树。索引与展开都以它为单位。"""

    source_id: str
    source_kind: str
    nodes: list[MemoryNode] = field(default_factory=list)

    def add(self, node: MemoryNode) -> MemoryNode:
        self.nodes.append(node)
        return node

    def by_id(self) -> dict[str, MemoryNode]:
        return {node.id: node for node in self.nodes}

    def leaves(self) -> list[MemoryNode]:
        return [node for node in self.nodes if node.is_leaf]

    def root(self) -> MemoryNode | None:
        return next((node for node in self.nodes if not node.parent_id), None)

    def children_of(self, node_id_value: str) -> list[MemoryNode]:
        return [node for node in self.nodes if node.parent_id == node_id_value]

    def ancestors_of(self, node_id_value: str) -> list[MemoryNode]:
        """从父到根。展开地图按这个顺序给出 up1 / up2 / … / file。"""

        index = self.by_id()
        chain: list[MemoryNode] = []
        current = index.get(node_id_value)
        while current is not None and current.parent_id:
            current = index.get(current.parent_id)
            if current is None:
                break
            chain.append(current)
        return chain

    def _children_map(self) -> dict[str, list[MemoryNode]]:
        order = {node.id: index for index, node in enumerate(self.nodes)}
        children: dict[str, list[MemoryNode]] = {}
        for node in self.nodes:
            children.setdefault(node.parent_id, []).append(node)
        for bucket in children.values():
            bucket.sort(key=lambda item: order[item.id])
        return children

    def fill_internal_text(self) -> None:
        """内部节点的正文 = 其后代按顺序拼接。

        展开到父节点时要返回可读的一整段，而不是把叶子丢给模型自己缝。
        """

        children = self._children_map()

        def compose(node: MemoryNode) -> str:
            if node.is_leaf:
                return node.text
            parts: list[str] = []
            if node.title:
                parts.append(node.title)
            for child in children.get(node.id, []):
                body = compose(child)
                if body.strip():
                    parts.append(body)
            return "\n\n".join(parts)

        for node in self.nodes:
            if not node.is_leaf:
                node.text = compose(node)

    def iter_depth_first(self) -> Iterator[MemoryNode]:
        children = self._children_map()

        def walk(node: MemoryNode) -> Iterator[MemoryNode]:
            yield node
            for child in children.get(node.id, []):
                yield from walk(child)

        root = self.root()
        if root is not None:
            yield from walk(root)
