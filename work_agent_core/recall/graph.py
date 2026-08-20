"""知识图谱接缝。

图本身还没建，但**位置现在就留出来**：等实体、关系、时效和证据真正建起来时，
检索侧不需要改结构，只需要换一个 `GraphStore` 实现。

留接缝而不留 TODO 的原因很实在：一个没人调用的协议会烂掉，所以默认实现
`NullGraphStore` 是真的接在检索管线里的——它返回空集，检索照常工作。
接缝因此始终被执行到，不会在建图那天才发现签名不对。

图要承载的东西（详见 TODO.md Phase 2）：
- 规范实体：人物、公司/组织、项目、地点、会议、材料
- 关系：主体、客体、方向、类型、业务时间、有效期、来源证据、
  候选/已确认/已否定/已失效、置信度、被谁纠正
- 每条关系可回到 `recall_nodes` 里的具体节点作为证据
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence


@dataclass(frozen=True)
class EntityRef:
    """一次实体指称：图里的实体 id，加上它在文本里的原始写法。"""

    entity_id: str
    mention: str = ""
    confidence: float = 1.0


@dataclass(frozen=True)
class EntityMatch:
    entity_id: str
    canonical: str
    aliases: tuple[str, ...] = ()
    kind: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


class GraphStore(Protocol):
    """检索侧需要图提供的三件事，仅此三件。

    刻意小：图可以长得很复杂，但检索只需要"这个名字指谁""跟它相关的还有谁"
    "这条关系的证据在哪"。接口小，换实现才不会牵动检索。
    """

    def resolve(self, name: str, *, scope: str = "") -> list[EntityMatch]:
        """把一个写法解析成实体候选。别名、历史名称、同名消歧都在这后面。"""

    def neighbors(self, entity_id: str, *, hops: int = 1, kinds: Sequence[str] = ()) -> list[EntityMatch]:
        """沿已确认关系扩展一到两跳。用于"跟零次方相关的还有哪些项目"。"""

    def evidence_nodes(self, entity_id: str) -> list[str]:
        """该实体的证据落在哪些 recall 节点上。让每条断言都能回到原文。"""


class NullGraphStore:
    """图还没建时的默认实现：什么都不知道，但一切照常工作。

    它不是占位符——检索管线真的调它。所以接缝天天在跑，不会烂。
    """

    def resolve(self, name: str, *, scope: str = "") -> list[EntityMatch]:
        return []

    def neighbors(self, entity_id: str, *, hops: int = 1, kinds: Sequence[str] = ()) -> list[EntityMatch]:
        return []

    def evidence_nodes(self, entity_id: str) -> list[str]:
        return []


def expand_entity_filter(
    graph: GraphStore,
    names: Sequence[str],
    *,
    hops: int = 0,
    scope: str = "",
) -> list[str]:
    """把用户/模型给的名字解析成实体 id 集合，可选地扩一跳。

    没有图时返回空列表，调用方据此退回"不做实体过滤"，而不是搜不到东西。
    """

    resolved: list[str] = []
    seen: set[str] = set()
    for name in names:
        for match in graph.resolve(str(name or "").strip(), scope=scope):
            if match.entity_id not in seen:
                seen.add(match.entity_id)
                resolved.append(match.entity_id)
    if hops <= 0:
        return resolved
    for entity_id in list(resolved):
        for neighbor in graph.neighbors(entity_id, hops=hops):
            if neighbor.entity_id not in seen:
                seen.add(neighbor.entity_id)
                resolved.append(neighbor.entity_id)
    return resolved
