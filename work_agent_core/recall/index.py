"""记忆索引：节点表 + FTS5 + 向量 + 实体关联。

索引是**投影**，不是第二数据源：随时可以删掉，从原始文件和会话日志完整重建。
所以这里只存能重建的东西，不存任何只在这里存在的事实。

按来源增量：一份材料的内容哈希没变就整份跳过；变了就把该来源的节点整体换掉。
只追加的语料每天在长，全量重建不是选项。
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence
import hashlib
import sqlite3
import struct
import time

from ..retrieval_core import extract_query_terms, index_terms
from .graph import EntityRef
from .nodes import MemoryNode, MemoryTree


SCHEMA_VERSION = 1


@dataclass
class UpsertReport:
    """一次写入动了什么。unchanged 的那部分是省下来的 embedding 调用。"""

    added: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    skipped: bool = False

    @property
    def touched(self) -> bool:
        return bool(self.added or self.updated or self.removed)


@dataclass(frozen=True)
class NodeFilter:
    """元数据过滤先于语义：时间和归属是结构化条件，不该丢给向量去猜。"""

    source_kinds: tuple[str, ...] = ()
    source_ids: tuple[str, ...] = ()
    since_ms: int = 0
    until_ms: int = 0
    entity_ids: tuple[str, ...] = ()

    def where(self) -> tuple[str, list[Any]]:
        clauses: list[str] = ["n.is_leaf = 1"]
        params: list[Any] = []
        if self.source_kinds:
            clauses.append(f"n.source_kind IN ({','.join('?' * len(self.source_kinds))})")
            params.extend(self.source_kinds)
        if self.source_ids:
            clauses.append(f"n.source_id IN ({','.join('?' * len(self.source_ids))})")
            params.extend(self.source_ids)
        if self.since_ms:
            clauses.append("n.occurred_at >= ?")
            params.append(int(self.since_ms))
        if self.until_ms:
            clauses.append("n.occurred_at <= ?")
            params.append(int(self.until_ms))
        if self.entity_ids:
            placeholders = ",".join("?" * len(self.entity_ids))
            clauses.append(
                "EXISTS (SELECT 1 FROM recall_node_entities e"
                f" WHERE e.node_id = n.id AND e.entity_id IN ({placeholders}))"
            )
            params.extend(self.entity_ids)
        return " AND ".join(clauses), params


def content_hash(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def pack_vector(values: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def unpack_vector(blob: bytes) -> list[float]:
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))


class RecallIndex:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            self._ensure_schema(connection)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def _ensure_schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS recall_sources (
                source_id TEXT PRIMARY KEY,
                source_kind TEXT NOT NULL,
                uri TEXT NOT NULL DEFAULT '',
                title TEXT NOT NULL DEFAULT '',
                content_hash TEXT NOT NULL,
                occurred_at INTEGER NOT NULL DEFAULT 0,
                indexed_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS recall_nodes (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                source_kind TEXT NOT NULL,
                parent_id TEXT NOT NULL DEFAULT '',
                title TEXT NOT NULL DEFAULT '',
                path TEXT NOT NULL DEFAULT '',
                text TEXT NOT NULL DEFAULT '',
                header TEXT NOT NULL DEFAULT '',
                occurred_at INTEGER NOT NULL DEFAULT 0,
                is_leaf INTEGER NOT NULL DEFAULT 0,
                tokens INTEGER NOT NULL DEFAULT 0,
                ordinal INTEGER NOT NULL DEFAULT 0,
                node_hash TEXT NOT NULL DEFAULT '',
                text_hash TEXT NOT NULL DEFAULT '',
                summary TEXT NOT NULL DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS recall_nodes_text_idx ON recall_nodes(text_hash);
            CREATE INDEX IF NOT EXISTS recall_nodes_source_idx ON recall_nodes(source_id);
            CREATE INDEX IF NOT EXISTS recall_nodes_parent_idx ON recall_nodes(parent_id);
            CREATE INDEX IF NOT EXISTS recall_nodes_time_idx ON recall_nodes(occurred_at);

            CREATE VIRTUAL TABLE IF NOT EXISTS recall_fts USING fts5(
                node_id UNINDEXED,
                search_text,
                tokenize='unicode61 remove_diacritics 2'
            );

            -- 向量按**正文哈希**存，不按节点存：同一份材料的 docx/md/pdf 三种渲染、
            -- 散在多个归档目录里的同一个文件，正文一样就共用一条向量。
            -- 实测 186480 个叶子去重后只剩 14458 条不同正文，省掉 92% 的 embedding 调用。
            CREATE TABLE IF NOT EXISTS recall_vectors (
                text_hash TEXT NOT NULL,
                model TEXT NOT NULL,
                dim INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                PRIMARY KEY (text_hash, model)
            );

            -- 知识图谱接缝：节点与实体的关联。图还没建，表先在这里，
            -- 检索的实体过滤已经查它，所以建图那天不必改检索。
            CREATE TABLE IF NOT EXISTS recall_node_entities (
                node_id TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                mention TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL DEFAULT 1.0,
                PRIMARY KEY (node_id, entity_id)
            );
            CREATE INDEX IF NOT EXISTS recall_node_entities_entity_idx
                ON recall_node_entities(entity_id);
            """
        )
        # 老库补列：索引是投影，删了能重建，但没必要为一列强迫重建。
        columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(recall_nodes)").fetchall()
        }
        for name in ("node_hash", "text_hash", "summary"):
            if name not in columns:
                connection.execute(
                    f"ALTER TABLE recall_nodes ADD COLUMN {name} TEXT NOT NULL DEFAULT ''"
                )
        if "summary_wanted" not in columns:
            connection.execute(
                "ALTER TABLE recall_nodes ADD COLUMN summary_wanted INTEGER NOT NULL DEFAULT 0"
            )

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    def source_hash(self, source_id: str) -> str:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT content_hash FROM recall_sources WHERE source_id = ?", (source_id,)
            ).fetchone()
        return str(row["content_hash"]) if row else ""

    def upsert_tree(
        self,
        tree: MemoryTree,
        *,
        uri: str = "",
        digest: str = "",
        aliases_for: Any | None = None,
    ) -> "UpsertReport":
        """按节点增量写入。没变的节点原地不动，**它们的向量因此得以保留**。

        一次对话每加一轮就整份重写的话，全库向量作废重算——一天下来光 embedding
        就是几十次重复调用。所以这里按 (id, 内容哈希) 求差：新增的插入、变了的
        更新、没了的删除，其余一律不碰。

        `aliases_for(node)` 可以给节点补一段**只进检索、不进正文**的文本
        （别名、纠正过的写法）。对应 V5_dev 的 caption_aux：让它可被找到，
        但不出现在返回给模型的正文里。
        """

        root = tree.root()
        if root is None:
            return UpsertReport()
        body = "\n".join(node.text for node in tree.iter_depth_first())
        digest = digest or content_hash(body)
        if self.source_hash(tree.source_id) == digest:
            return UpsertReport(skipped=True)

        incoming: dict[str, tuple[MemoryNode, int, str]] = {}
        for ordinal, node in enumerate(tree.iter_depth_first()):
            incoming[node.id] = (node, ordinal, content_hash(f"{ordinal}\x1f{node.searchable_body()}"))

        report = UpsertReport()
        with self._connect() as connection:
            existing = {
                str(row["id"]): str(row["node_hash"])
                for row in connection.execute(
                    "SELECT id, node_hash FROM recall_nodes WHERE source_id = ?",
                    (tree.source_id,),
                ).fetchall()
            }
            for stale_id in set(existing) - set(incoming):
                self._delete_node(connection, stale_id)
                report.removed += 1

            for node_id_value, (node, ordinal, node_hash) in incoming.items():
                if existing.get(node_id_value) == node_hash:
                    report.unchanged += 1
                    continue
                changed = node_id_value in existing
                if changed:
                    # 内容变了，旧向量不再对应这段文本，必须失效。
                    self._delete_node(connection, node_id_value)
                row = node.to_row()
                row["ordinal"] = ordinal
                row["node_hash"] = node_hash
                # 正文哈希单独存：同一份材料的 docx/md/pdf 三种渲染、以及散在多个
                # 归档目录里的同一个文件，内容一样但路径不同，只能靠它去重。
                row["text_hash"] = content_hash(node.searchable_body())
                connection.execute(
                    """
                    INSERT INTO recall_nodes
                        (id, source_id, source_kind, parent_id, title, path, text,
                         header, occurred_at, is_leaf, tokens, ordinal, node_hash,
                         text_hash)
                    VALUES (:id, :source_id, :source_kind, :parent_id, :title, :path,
                            :text, :header, :occurred_at, :is_leaf, :tokens, :ordinal,
                            :node_hash, :text_hash)
                    """,
                    row,
                )
                if node.is_leaf:
                    extra = ""
                    if aliases_for is not None:
                        try:
                            extra = str(aliases_for(node) or "")
                        except Exception:
                            extra = ""
                    connection.execute(
                        "INSERT INTO recall_fts (node_id, search_text) VALUES (?, ?)",
                        (node.id, self.build_search_text(node, extra)),
                    )
                if changed:
                    report.updated += 1
                else:
                    report.added += 1

            connection.execute(
                """
                INSERT INTO recall_sources
                    (source_id, source_kind, uri, title, content_hash, occurred_at, indexed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_id) DO UPDATE SET
                    source_kind = excluded.source_kind,
                    uri = excluded.uri,
                    title = excluded.title,
                    content_hash = excluded.content_hash,
                    occurred_at = excluded.occurred_at,
                    indexed_at = excluded.indexed_at
                """,
                (
                    tree.source_id,
                    tree.source_kind,
                    uri,
                    root.title,
                    digest,
                    root.occurred_at,
                    int(time.time() * 1000),
                ),
            )
        return report

    @staticmethod
    def _delete_node(connection: sqlite3.Connection, node_id_value: str) -> None:
        connection.execute("DELETE FROM recall_fts WHERE node_id = ?", (node_id_value,))
        # 不删向量：它按正文哈希共享，别的节点可能还在用。孤儿向量由 vacuum 清理。
        connection.execute("DELETE FROM recall_node_entities WHERE node_id = ?", (node_id_value,))
        connection.execute("DELETE FROM recall_nodes WHERE id = ?", (node_id_value,))

    @staticmethod
    def build_search_text(node: MemoryNode, extra: str = "") -> str:
        """检索文本 ≠ 返回文本。

        路径、表头和别名都进来，因为它们决定"找不找得到"；返回给模型的仍然
        只有正文。
        """

        parts = [" ".join(node.path), node.header, node.text, extra]
        return " ".join(index_terms(" ".join(part for part in parts if part)))

    @staticmethod
    def _delete_source(connection: sqlite3.Connection, source_id: str) -> None:
        connection.execute(
            "DELETE FROM recall_fts WHERE node_id IN"
            " (SELECT id FROM recall_nodes WHERE source_id = ?)",
            (source_id,),
        )
        connection.execute(
            "DELETE FROM recall_node_entities WHERE node_id IN"
            " (SELECT id FROM recall_nodes WHERE source_id = ?)",
            (source_id,),
        )
        connection.execute("DELETE FROM recall_nodes WHERE source_id = ?", (source_id,))
        connection.execute("DELETE FROM recall_sources WHERE source_id = ?", (source_id,))

    def forget_source(self, source_id: str) -> None:
        with self._connect() as connection:
            self._delete_source(connection, source_id)

    # ------------------------------------------------------------------
    # 读取
    # ------------------------------------------------------------------

    def node(self, node_id_value: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM recall_nodes WHERE id = ?", (node_id_value,)
            ).fetchone()
        return dict(row) if row else None

    def display_node(self, node_id_value: str) -> dict[str, Any] | None:
        """匹配到窗口，返回包住它的展示单元。

        窗口只为命中准而存在，本身读起来是半截的；它的父节点才是完整的一段。
        没有父节点（整节就是一块）时返回它自己。
        """

        node = self.node(node_id_value)
        if node is None:
            return None
        if not node["is_leaf"] or not node["parent_id"]:
            return node
        parent = self.node(str(node["parent_id"]))
        return parent if parent is not None else node

    def ancestors(self, node_id_value: str) -> list[dict[str, Any]]:
        chain: list[dict[str, Any]] = []
        current = self.node(node_id_value)
        while current and current["parent_id"]:
            parent = self.node(str(current["parent_id"]))
            if parent is None:
                break
            chain.append(parent)
            current = parent
        return chain

    def neighbors(self, node_id_value: str) -> dict[str, str]:
        """前一段与后一段——都在**展示单元**这一层。

        往后看是"这件事后来怎么样了"，往前看是"为什么会有这一段"。兄弟窗口是
        半句挨半句，回答不了这两个问题；兄弟展示单元才可以。
        """

        current = self.display_node(node_id_value)
        if current is None:
            return {}
        current_id = str(current["id"])
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id FROM recall_nodes WHERE parent_id = ? ORDER BY ordinal",
                (current["parent_id"],),
            ).fetchall()
        ids = [str(row["id"]) for row in rows]
        if current_id not in ids:
            return {}
        position = ids.index(current_id)
        found: dict[str, str] = {}
        if position > 0:
            found["prev"] = ids[position - 1]
        if position + 1 < len(ids):
            found["next"] = ids[position + 1]
        return found

    def lexical_candidates(
        self,
        query: str,
        *,
        limit: int = 20,
        filters: NodeFilter | None = None,
    ) -> list[str]:
        terms = extract_query_terms(query)
        if not terms:
            return []
        match_expression = " OR ".join(f'"{term}"' for term in terms)
        where, params = (filters or NodeFilter()).where()
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT n.id AS id
                FROM recall_fts
                JOIN recall_nodes n ON n.id = recall_fts.node_id
                WHERE recall_fts MATCH ? AND {where}
                ORDER BY bm25(recall_fts) ASC
                LIMIT ?
                """,
                [match_expression, *params, int(limit)],
            ).fetchall()
        return [str(row["id"]) for row in rows]

    def leaves_without_vectors(self, model: str, *, limit: int = 256) -> list[dict[str, Any]]:
        """每种不同正文只返回一个代表。重复内容不该重复付费。"""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT n.id, n.path, n.text, n.header, n.text_hash, MAX(n.occurred_at) AS occurred_at
                FROM recall_nodes n
                LEFT JOIN recall_vectors v ON v.text_hash = n.text_hash AND v.model = ?
                WHERE n.is_leaf = 1 AND n.text_hash != '' AND v.text_hash IS NULL
                GROUP BY n.text_hash
                ORDER BY occurred_at DESC
                LIMIT ?
                """,
                (model, int(limit)),
            ).fetchall()
        return [dict(row) for row in rows]

    def store_vectors(self, model: str, vectors: Iterable[tuple[str, Sequence[float]]]) -> int:
        """按正文哈希写入。传入的第一项是 text_hash，不是 node_id。"""

        written = 0
        with self._connect() as connection:
            for text_hash, values in vectors:
                if not values or not text_hash:
                    continue
                connection.execute(
                    "INSERT OR REPLACE INTO recall_vectors (text_hash, model, dim, embedding)"
                    " VALUES (?, ?, ?, ?)",
                    (text_hash, model, len(values), pack_vector(values)),
                )
                written += 1
        return written

    def vectors_by_text(self, model: str) -> list[tuple[str, list[float]]]:
        """返回 (text_hash, 向量)，每种正文一条。

        打分要在**内容层**做，不在节点层：同一条向量被多少节点共用不影响它的
        相似度。按节点取会把 1.6 万条向量放大成 17 万行，光加载就要几秒。
        """

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT text_hash, embedding FROM recall_vectors WHERE model = ?",
                (model,),
            ).fetchall()
        return [(str(row["text_hash"]), unpack_vector(row["embedding"])) for row in rows]

    def nodes_for_texts(
        self,
        text_hashes: Sequence[str],
        *,
        filters: NodeFilter | None = None,
    ) -> dict[str, list[str]]:
        """把胜出的正文映射回符合过滤条件的节点。"""

        if not text_hashes:
            return {}
        where, params = (filters or NodeFilter()).where()
        placeholders = ",".join("?" * len(text_hashes))
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT n.id AS id, n.text_hash AS text_hash FROM recall_nodes n
                WHERE n.text_hash IN ({placeholders}) AND {where}
                ORDER BY n.occurred_at DESC
                """,
                [*text_hashes, *params],
            ).fetchall()
        mapping: dict[str, list[str]] = {}
        for row in rows:
            mapping.setdefault(str(row["text_hash"]), []).append(str(row["id"]))
        return mapping

    def vector_coverage(self, model: str) -> dict[str, int]:
        with self._connect() as connection:
            distinct = int(
                connection.execute(
                    "SELECT COUNT(DISTINCT text_hash) AS c FROM recall_nodes"
                    " WHERE is_leaf = 1 AND text_hash != ''"
                ).fetchone()["c"]
            )
            done = int(
                connection.execute(
                    "SELECT COUNT(*) AS c FROM recall_vectors WHERE model = ?", (model,)
                ).fetchone()["c"]
            )
        return {"distinct_texts": distinct, "embedded": done, "remaining": max(0, distinct - done)}

    # ------------------------------------------------------------------
    # 知识图谱接缝
    # ------------------------------------------------------------------

    def link_entities(self, node_id_value: str, refs: Iterable[EntityRef]) -> int:
        """把节点关联到实体。图建起来之后由抽取侧填，检索侧已经在查它了。"""

        written = 0
        with self._connect() as connection:
            for ref in refs:
                connection.execute(
                    "INSERT OR REPLACE INTO recall_node_entities"
                    " (node_id, entity_id, mention, confidence) VALUES (?, ?, ?, ?)",
                    (node_id_value, ref.entity_id, ref.mention, float(ref.confidence)),
                )
                written += 1
        return written

    def entities_for(self, node_id_value: str) -> list[EntityRef]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT entity_id, mention, confidence FROM recall_node_entities"
                " WHERE node_id = ?",
                (node_id_value,),
            ).fetchall()
        return [
            EntityRef(
                entity_id=str(row["entity_id"]),
                mention=str(row["mention"]),
                confidence=float(row["confidence"]),
            )
            for row in rows
        ]

    def want_summaries(self, node_ids: Iterable[str]) -> None:
        """记下"模型问到过这一节但它没有摘要"。

        全量生成不现实：这个库里 ≥1000t 的章节就有 15095 个，按限流要跑 8 小时，
        而其中绝大多数永远不会被打开。所以按需——被展开地图问到过的才排队，
        没人碰过的一分钱不花。
        """

        ids = [str(item) for item in node_ids if item]
        if not ids:
            return
        with self._connect() as connection:
            connection.executemany(
                "UPDATE recall_nodes SET summary_wanted = summary_wanted + 1"
                " WHERE id = ? AND summary = ''",
                [(item,) for item in ids],
            )

    def vacuum_vectors(self) -> int:
        """清掉没有任何节点再引用的向量。

        向量按正文共享，所以删节点时不能顺手删它。孤儿在这里统一回收，
        代价是一次全表扫描，所以不放在写入路径上。
        """

        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM recall_vectors WHERE text_hash NOT IN"
                " (SELECT DISTINCT text_hash FROM recall_nodes WHERE is_leaf = 1)"
            )
            return int(cursor.rowcount or 0)

    def stats(self) -> dict[str, int]:
        with self._connect() as connection:
            def count(table: str) -> int:
                return int(connection.execute(f"SELECT COUNT(*) AS c FROM {table}").fetchone()["c"])

            return {
                "sources": count("recall_sources"),
                "nodes": count("recall_nodes"),
                "leaves": int(
                    connection.execute(
                        "SELECT COUNT(*) AS c FROM recall_nodes WHERE is_leaf = 1"
                    ).fetchone()["c"]
                ),
                "vectors": count("recall_vectors"),
                "entity_links": count("recall_node_entities"),
            }
