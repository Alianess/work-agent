"""检索管线与展开地图。

返回叶子，外加一张**标好 token 价**的展开地图。决定"够不够"的只有模型自己，
所以把选择权交给它——但必须同时把账本交给它，否则它只能瞎开。

排序含时间维度。手册检索不需要，记忆检索需要：这里"最近的"几乎总是更相关，
而"上次""最新一版"是最高频的问法。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence
import math
import time

from .backends import RecallBackendError, cosine, normalize
from .graph import GraphStore, NullGraphStore
from .index import NodeFilter, RecallIndex
from .nodes import split_path


RECALL_CANDIDATES = 20
RRF_K = 60
DEFAULT_TOP_K = 6
DEFAULT_RECENCY_WEIGHT = 0.3
RECENCY_HALF_LIFE_DAYS = 45.0
SNIPPET_CHARS = 220
EXPAND_MAX_TOKENS = 6000


@dataclass
class RecallDeps:
    """检索需要的外部件。缺哪个就降级哪一段，不让一次检索整体失败。"""

    index: RecallIndex
    embedding: Any | None = None
    rerank: Any | None = None
    graph: GraphStore = field(default_factory=NullGraphStore)


def _rrf(ranked_lists: Sequence[Sequence[str]], *, k: int = RRF_K) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for position, node_id_value in enumerate(ranked, start=1):
            scores[node_id_value] = scores.get(node_id_value, 0.0) + 1.0 / (k + position)
    return scores


def recency_factor(occurred_at: int, now_ms: int) -> float:
    """越近越接近 1，越远越接近 0。半衰期以天计。"""

    if occurred_at <= 0 or now_ms <= 0:
        return 0.0
    age_days = max(0.0, (now_ms - occurred_at) / 86_400_000.0)
    return math.exp(-age_days / RECENCY_HALF_LIFE_DAYS)


def snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    body = " ".join(str(text or "").split())
    return body if len(body) <= limit else body[:limit].rstrip() + "…"


def _expand_map(index: RecallIndex, node_id_value: str) -> dict[str, dict[str, Any]]:
    """祖先链转成标价的展开选项：up1 / up2 / … / file。

    每项带 tokens（要花多少）和 summary（里面是什么）。只有价格没有货物，
    选择仍然是盲的——摘要补的正是这一半。
    """

    chain = index.ancestors(node_id_value)
    options: dict[str, dict[str, Any]] = {}
    for depth, ancestor in enumerate(chain, start=1):
        key = "file" if depth == len(chain) else f"up{depth}"
        option = {
            "id": str(ancestor["id"]),
            "title": str(ancestor["title"] or ""),
            "tokens": int(ancestor["tokens"]),
        }
        summary = str(ancestor["summary"] or "")
        if summary:
            option["summary"] = summary
        options[key] = option
    return options


def search(
    deps: RecallDeps,
    query: str,
    *,
    filters: NodeFilter | None = None,
    entity_names: Sequence[str] = (),
    top_k: int = DEFAULT_TOP_K,
    recency_weight: float = DEFAULT_RECENCY_WEIGHT,
    now_ms: int = 0,
) -> dict[str, Any]:
    index = deps.index
    filters = filters or NodeFilter()
    degraded: list[str] = []

    # 实体过滤先于语义：名字是结构化条件，不该丢给向量去猜。
    if entity_names:
        from .graph import expand_entity_filter

        entity_ids = tuple(expand_entity_filter(deps.graph, entity_names))
        if entity_ids:
            filters = NodeFilter(
                source_kinds=filters.source_kinds,
                source_ids=filters.source_ids,
                since_ms=filters.since_ms,
                until_ms=filters.until_ms,
                entity_ids=entity_ids,
            )
        else:
            degraded.append("知识图谱未建，实体过滤跳过")

    lexical = index.lexical_candidates(query, limit=RECALL_CANDIDATES, filters=filters)

    dense: list[str] = []
    if deps.embedding is not None:
        try:
            query_vector = normalize(deps.embedding.embed([query])[0])
            # 在内容层打分，再映射回节点：向量按正文共享，节点层打分等于把同一条
            # 向量算很多遍。多取一些正文，因为过滤之后可能有的正文一个节点都不剩。
            stored = index.vectors_by_text(deps.embedding.model)
            scored = sorted(
                ((text_hash, cosine(query_vector, vector)) for text_hash, vector in stored),
                key=lambda item: item[1],
                reverse=True,
            )[: RECALL_CANDIDATES * 3]
            mapping = index.nodes_for_texts([text_hash for text_hash, _ in scored], filters=filters)
            dense = []
            for text_hash, _ in scored:
                dense.extend(mapping.get(text_hash, [])[:1])
                if len(dense) >= RECALL_CANDIDATES:
                    break
        except RecallBackendError as error:
            degraded.append(f"向量召回不可用：{error}")
        except Exception as error:  # pragma: no cover - 后端异常形态不可穷举
            degraded.append(f"向量召回失败：{type(error).__name__}")
    else:
        degraded.append("未配置向量后端，仅词法召回")

    fused = _rrf([lexical, dense] if dense else [lexical])
    candidates = sorted(fused, key=lambda node_id_value: -fused[node_id_value])[:80]
    if not candidates:
        return {
            "ok": True,
            "query": query,
            "results": [],
            "retrieval": {
                "lexical": 0,
                "dense": len(dense),
                "reranked": False,
                "degraded": degraded,
            },
            "note": "没有命中。换用当时出现过的专名、文件名或另一种说法再试。",
        }

    # 命中发生在窗口上，返回的是包住它的展示单元。窗口切小是为了匹配准，
    # 给模型看的却该是一段读得完整的正文——两件事分开，谁也不必迁就谁。
    rows: dict[str, Any] = {}
    for node_id_value in candidates:
        row = index.display_node(node_id_value)
        if row is not None:
            rows.setdefault(str(row["id"]), row)

    relevance: list[str]
    reranked = False
    if deps.rerank is not None:
        try:
            documents = [
                f"{rows[node_id_value]['path']}\n{rows[node_id_value]['header']}\n{rows[node_id_value]['text']}"
                for node_id_value in rows
            ]
            order = deps.rerank.rank(query, documents, top_n=min(len(documents), 20))
            keys = list(rows)
            relevance = [keys[position] for position, _ in order if 0 <= position < len(keys)]
            # 精排只返回 top_n，没被返回的候选仍要有名次——否则后面按名次排序会漏掉它们。
            # 排在精排结果之后，保留融合阶段的相对顺序。
            ranked = set(relevance)
            relevance.extend(key for key in keys if key not in ranked)
            reranked = True
        except RecallBackendError as error:
            degraded.append(f"精排不可用：{error}")
            relevance = list(rows)
        except Exception as error:  # pragma: no cover
            degraded.append(f"精排失败：{type(error).__name__}")
            relevance = list(rows)
    else:
        degraded.append("未配置精排后端，按融合名次排序")
        relevance = list(rows)

    # 时间维度：用连续衰减做乘子，而不是再融合一次名次。
    # 名次融合在 k=60 下把差异压得太平，一个 0.3 权重的第二信号永远翻不动
    # 相邻名次；而两条同样相关的证据里，新的那条本来就该赢。
    relevance_rank = {node_id_value: position for position, node_id_value in enumerate(relevance, 1)}
    stamp = now_ms or int(time.time() * 1000)

    def blended(node_id_value: str) -> float:
        base = 1.0 / (RRF_K + relevance_rank[node_id_value])
        freshness = recency_factor(int(rows[node_id_value]["occurred_at"] or 0), stamp)
        return base * (1.0 + recency_weight * freshness)

    final = sorted(rows, key=lambda node_id_value: -blended(node_id_value))

    results: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    # 同一份材料常常同时存在 docx / md / pdf 三种渲染，还会散在多个归档目录里。
    # 按正文哈希去重，只给模型看一份，其余折算成一个计数——重复不该占预算。
    seen_text: dict[str, int] = {}
    duplicates = 0
    for node_id_value in final:
        row = rows[node_id_value]
        text_key = str(row["text_hash"] or "")
        if text_key and text_key in seen_text:
            kept = results[seen_text[text_key]]
            kept["duplicates"] += 1
            duplicates += 1
            # 同样的内容出现在多处时，摆出来的应当是**最新的那份**。
            # 谁排名靠前是检索的偶然，谁更新是事实。
            if int(row["occurred_at"] or 0) > kept["occurred_at"]:
                kept["also_in"].append({"id": kept["id"], "path": kept["path"]})
                kept["id"] = node_id_value
                kept["path"] = split_path(str(row["path"]))
                kept["source_id"] = str(row["source_id"])
                kept["occurred_at"] = int(row["occurred_at"] or 0)
                kept["expand"] = _expand_map(index, node_id_value)
                kept["neighbors"] = index.neighbors(node_id_value)
            else:
                kept["also_in"].append(
                    {"id": node_id_value, "path": split_path(str(row["path"]))}
                )
            continue
        path_key = f"{row['source_id']}|{row['path']}"
        if path_key in seen_paths:
            continue
        seen_paths.add(path_key)
        if text_key:
            seen_text[text_key] = len(results)
        results.append(
            {
                "id": node_id_value,
                "source_id": str(row["source_id"]),
                "source_kind": str(row["source_kind"]),
                "path": split_path(str(row["path"])),
                "text": str(row["text"]),
                "header": str(row["header"] or ""),
                "tokens": int(row["tokens"]),
                "occurred_at": int(row["occurred_at"] or 0),
                "expand": _expand_map(index, node_id_value),
                "neighbors": index.neighbors(node_id_value),
                "duplicates": 0,
                "also_in": [],
            }
        )
        if len(results) >= top_k:
            break

    # 展开地图里没有摘要的章节记一笔：被问到过才值得为它生成摘要。
    index.want_summaries(
        option["id"]
        for item in results
        for option in item["expand"].values()
        if not option.get("summary")
    )

    return {
        "ok": True,
        "query": query,
        "results": results,
        "retrieval": {
            "lexical": len(lexical),
            "dense": len(dense),
            "reranked": reranked,
            "recency_weight": recency_weight,
            "duplicates_folded": duplicates,
            "degraded": degraded,
        },
        "note": (
            "results 是命中的最小片段。不够用时用 recall_expand 按 expand 里的 id 展开，"
            "tokens 是展开的代价。"
        ),
    }


def expand(
    index: RecallIndex,
    node_id_value: str,
    *,
    scope: str = "",
    max_tokens: int = 0,
    seen_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """按结构或按预算展开一个节点。

    不切碎父节点——切了反而读不通。已经给过模型的子节点会被标出来，
    它在缓存前缀里，重复的代价很小，读不通的代价大得多。
    """

    target = index.node(node_id_value)
    if target is None:
        return {"ok": False, "error": f"未知节点：{node_id_value}"}

    if scope in {"prev", "next"}:
        neighbor_id = index.neighbors(node_id_value).get(scope)
        if not neighbor_id:
            return {"ok": False, "error": f"没有{scope}邻居"}
        target = index.node(neighbor_id) or target
    elif scope:
        chain = index.ancestors(node_id_value)
        if not chain:
            return {"ok": False, "error": "该节点没有上层"}
        if scope == "file":
            target = chain[-1]
        elif scope.startswith("up"):
            try:
                depth = int(scope[2:] or "1")
            except ValueError:
                return {"ok": False, "error": f"无法识别的 scope：{scope}"}
            if depth < 1 or depth > len(chain):
                return {"ok": False, "error": f"只有 {len(chain)} 层可以往上展开"}
            target = chain[depth - 1]
        else:
            return {"ok": False, "error": f"无法识别的 scope：{scope}"}
    elif max_tokens > 0:
        # 按预算挑能装下的最大完整单元。装不下就退回节点自身。
        chain = index.ancestors(node_id_value)
        for ancestor in chain:
            if int(ancestor["tokens"]) <= max_tokens:
                target = ancestor
            else:
                break

    tokens = int(target["tokens"])
    if tokens > EXPAND_MAX_TOKENS:
        # 超上限不硬灌，降级成目录让模型再挑一次。
        children = [
            {
                "id": str(row["id"]),
                "title": str(row["title"] or snippet(str(row["text"]), 40)),
                "tokens": int(row["tokens"]),
                **({"summary": str(row["summary"])} if row["summary"] else {}),
            }
            for row in _children(index, str(target["id"]))
        ]
        return {
            "ok": True,
            "id": str(target["id"]),
            "too_large": True,
            "tokens": tokens,
            "limit": EXPAND_MAX_TOKENS,
            "outline": children,
            "note": "这一节超过展开上限，先从目录里挑一个更小的再展开。",
        }

    already = [item for item in seen_ids if item and item != str(target["id"])]
    return {
        "ok": True,
        "id": str(target["id"]),
        "path": split_path(str(target["path"])),
        "title": str(target["title"] or ""),
        "tokens": tokens,
        "text": str(target["text"]),
        "expand": _expand_map(index, str(target["id"])),
        "already_shown": already,
    }


def _children(index: RecallIndex, node_id_value: str) -> list[dict[str, Any]]:
    with index._connect() as connection:  # noqa: SLF001 - 同包内部读取
        rows = connection.execute(
            "SELECT id, title, text, tokens, summary FROM recall_nodes"
            " WHERE parent_id = ? ORDER BY ordinal",
            (node_id_value,),
        ).fetchall()
    return [dict(row) for row in rows]
