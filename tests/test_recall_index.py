"""索引与检索：命中最小片段，展开选项标好价，降级要说出来。"""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from work_agent_core.recall.backends import RecallBackendError, SiliconFlowEmbedding, SiliconFlowRerank
from work_agent_core.recall.chunking import build_chat_tree, build_document_tree
from work_agent_core.recall.graph import EntityMatch, EntityRef, NullGraphStore, expand_entity_filter
from work_agent_core.recall.index import NodeFilter, RecallIndex
from work_agent_core.recall.search import RecallDeps, expand, search


REPORT = """# 报市稿

一、当前存在的问题

（一）本体企业少

我市现有具身智能相关企业十余家，多集中在零部件环节。零次方、逐际动力等头部企业
均未在我市设立生产基地，缺少整机牵引。

二、几点建议

1. 建设中试基地

建议依托现有闲置厂房，建设具身智能中试基地，解决样机验证场地不足的问题，由工信局牵头。

2. 开放应用场景

建议梳理全市公共场景清单，按季度向企业开放，形成常态化的验证机会。
"""


def fresh_index() -> RecallIndex:
    return RecallIndex(Path(tempfile.mkdtemp()) / "recall.sqlite3")


def indexed(now_ms: int = 0) -> tuple[RecallIndex, int]:
    stamp = now_ms or int(time.time() * 1000)
    index = fresh_index()
    index.upsert_tree(
        build_document_tree(
            source_id="doc:报市稿", title="报市稿", text=REPORT, occurred_at=stamp - 86_400_000
        ),
        uri="meet_files/报市稿.md",
    )
    return index, stamp


class IncrementalIndexTests(unittest.TestCase):
    def test_unchanged_content_is_skipped_whole(self) -> None:
        index = fresh_index()
        tree = build_document_tree(source_id="doc:a", title="a", text=REPORT)

        self.assertTrue(index.upsert_tree(tree).added)
        self.assertTrue(index.upsert_tree(tree).skipped)

    def test_changed_content_replaces_the_source_instead_of_piling_up(self) -> None:
        index = fresh_index()
        index.upsert_tree(build_document_tree(source_id="doc:a", title="a", text=REPORT))
        before = index.stats()["nodes"]

        index.upsert_tree(
            build_document_tree(source_id="doc:a", title="a", text="# a\n\n只剩一句话了。\n")
        )

        self.assertLess(index.stats()["nodes"], before)
        self.assertEqual(index.stats()["sources"], 1)

    def test_forgetting_a_source_leaves_nothing_behind(self) -> None:
        index, _ = indexed()
        node_id_value = index.lexical_candidates("中试基地")[0]
        index.link_entities(node_id_value, [EntityRef(entity_id="ent:x")])

        index.forget_source("doc:报市稿")

        stats = index.stats()
        self.assertEqual(
            {key: stats[key] for key in ("sources", "nodes", "leaves", "entity_links")},
            {"sources": 0, "nodes": 0, "leaves": 0, "entity_links": 0},
        )


class LexicalTests(unittest.TestCase):
    def test_chinese_terms_match_without_a_word_segmenter(self) -> None:
        index, _ = indexed()
        self.assertTrue(index.lexical_candidates("中试基地"))
        self.assertTrue(index.lexical_candidates("零次方"))

    def test_the_heading_path_is_searchable_but_not_part_of_the_body(self) -> None:
        index, _ = indexed()
        hits = index.lexical_candidates("当前存在的问题")
        self.assertTrue(hits)
        self.assertNotIn("当前存在的问题", index.node(hits[0])["text"])

    def test_filters_run_before_the_query(self) -> None:
        index, now = indexed()
        self.assertEqual(index.lexical_candidates("基地", filters=NodeFilter(since_ms=now + 1)), [])
        self.assertEqual(
            index.lexical_candidates("基地", filters=NodeFilter(source_kinds=("chat",))), []
        )


class VectorStoreTests(unittest.TestCase):
    def test_vectors_round_trip_and_are_reported_as_missing_until_written(self) -> None:
        index, _ = indexed()
        pending = index.leaves_without_vectors("m1")
        self.assertTrue(pending)

        index.store_vectors("m1", [(pending[0]["text_hash"], [0.1, 0.2, 0.3])])

        stored = dict(index.vectors_by_text("m1"))
        self.assertEqual(
            [round(value, 3) for value in stored[pending[0]["text_hash"]]], [0.1, 0.2, 0.3]
        )
        self.assertLess(len(index.leaves_without_vectors("m1")), len(pending))

    def test_identical_text_in_two_files_shares_one_vector(self) -> None:
        """同一份材料的 docx/md/pdf 三种渲染只该付一次 embedding 的钱。"""

        index = fresh_index()
        body = "一、结论\n\n中试基地的选址已经确定，由工信局牵头推进。\n"
        for name in ("doc:a", "doc:b", "doc:c"):
            index.upsert_tree(build_document_tree(source_id=name, title=name, text=f"# {name}\n\n{body}"))

        pending = index.leaves_without_vectors("m1", limit=100)

        self.assertEqual(len(pending), 1)
        index.store_vectors("m1", [(pending[0]["text_hash"], [0.5, 0.5])])
        # 一条向量，三个节点都能用上——打分只算一次，映射回节点时才展开
        self.assertEqual(len(index.vectors_by_text("m1")), 1)
        self.assertEqual(len(index.nodes_for_texts([pending[0]["text_hash"]])[pending[0]["text_hash"]]), 3)
        self.assertEqual(index.vector_coverage("m1"), {
            "distinct_texts": 1, "embedded": 1, "remaining": 0
        })

    def test_orphan_vectors_are_reclaimed_only_on_vacuum(self) -> None:
        index, _ = indexed()
        pending = index.leaves_without_vectors("m1")
        index.store_vectors("m1", [(row["text_hash"], [0.1, 0.2]) for row in pending])

        index.forget_source("doc:报市稿")

        # 删来源不顺手删向量：它按正文共享，别的节点可能还在用
        self.assertGreater(index.stats()["vectors"], 0)
        self.assertGreater(index.vacuum_vectors(), 0)
        self.assertEqual(index.stats()["vectors"], 0)


class ExpandMapTests(unittest.TestCase):
    def test_every_expansion_option_carries_its_price(self) -> None:
        index, now = indexed()
        result = search(RecallDeps(index=index), "中试基地怎么建", now_ms=now)["results"][0]

        options = result["expand"]
        self.assertEqual(set(options) & {"up1", "file"}, {"up1", "file"})
        for option in options.values():
            self.assertGreater(option["tokens"], 0)
        # 越往上越贵，模型才能据此选择
        self.assertLess(options["up1"]["tokens"], options["file"]["tokens"])
        self.assertLessEqual(result["tokens"], options["up1"]["tokens"])

    def test_a_hit_returns_the_smallest_readable_unit(self) -> None:
        index, now = indexed()
        result = search(RecallDeps(index=index), "中试基地怎么建", now_ms=now)["results"][0]

        self.assertIn("中试基地", result["text"])
        self.assertLess(result["tokens"], 120)
        self.assertEqual(result["path"][0], "报市稿")


class ExpandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.index, self.now = indexed()
        self.leaf = search(RecallDeps(index=self.index), "中试基地", now_ms=self.now)["results"][0]["id"]

    def test_up_one_level_returns_the_section_whole(self) -> None:
        opened = expand(self.index, self.leaf, scope="up1")
        self.assertTrue(opened["ok"])
        self.assertIn("中试基地", opened["text"])
        self.assertIn("建设中试基地", opened["title"])

    def test_file_scope_reaches_the_root(self) -> None:
        opened = expand(self.index, self.leaf, scope="file")
        self.assertEqual(opened["title"], "报市稿")
        self.assertIn("当前存在的问题", opened["text"])

    def test_a_budget_picks_the_largest_unit_that_fits(self) -> None:
        small = expand(self.index, self.leaf, max_tokens=60)
        whole = expand(self.index, self.leaf, max_tokens=100_000)
        self.assertLessEqual(small["tokens"], 60)
        self.assertEqual(whole["title"], "报市稿")

    def test_an_unknown_node_fails_loudly(self) -> None:
        self.assertFalse(expand(self.index, "doc:nope#0000")["ok"])

    def test_an_oversized_expansion_degrades_to_an_outline(self) -> None:
        index = fresh_index()
        big = "# 大文件\n\n" + "\n\n".join(
            f"一、第{n}章\n\n" + "这一段很长。" * 200 for n in range(1, 8)
        )
        index.upsert_tree(build_document_tree(source_id="doc:big", title="大文件", text=big))
        leaf = index.lexical_candidates("这一段很长")[0]

        opened = expand(index, leaf, scope="file")

        self.assertTrue(opened["ok"])
        self.assertTrue(opened["too_large"])
        self.assertTrue(opened["outline"])
        self.assertNotIn("text", opened)


class DegradationTests(unittest.TestCase):
    def test_missing_backends_are_named_not_hidden(self) -> None:
        index, now = indexed()
        report = search(RecallDeps(index=index), "中试基地", now_ms=now)["retrieval"]

        self.assertFalse(report["reranked"])
        self.assertEqual(report["dense"], 0)
        self.assertTrue(any("向量" in reason for reason in report["degraded"]))

    def test_a_failing_backend_degrades_instead_of_failing_the_search(self) -> None:
        class _Broken:
            model = "m1"

            def embed(self, _texts):
                raise RecallBackendError("endpoint down")

        index, now = indexed()
        out = search(RecallDeps(index=index, embedding=_Broken()), "中试基地", now_ms=now)

        self.assertTrue(out["results"])
        self.assertTrue(any("向量召回不可用" in reason for reason in out["retrieval"]["degraded"]))


class RecencyTests(unittest.TestCase):
    def test_the_more_recent_of_two_equal_hits_wins(self) -> None:
        index = fresh_index()
        now = int(time.time() * 1000)
        body = "一、结论\n\n中试基地的选址已经确定。\n"
        index.upsert_tree(
            build_document_tree(
                source_id="doc:old", title="旧稿", text=f"# 旧稿\n\n{body}",
                occurred_at=now - 400 * 86_400_000,
            )
        )
        index.upsert_tree(
            build_document_tree(
                source_id="doc:new", title="新稿", text=f"# 新稿\n\n{body}",
                occurred_at=now,
            )
        )

        results = search(RecallDeps(index=index), "中试基地选址", now_ms=now)["results"]

        self.assertEqual(results[0]["source_id"], "doc:new")

    def test_recency_can_be_turned_off_by_the_caller(self) -> None:
        index, now = indexed()
        out = search(RecallDeps(index=index), "中试基地", recency_weight=0.0, now_ms=now)
        self.assertEqual(out["retrieval"]["recency_weight"], 0.0)


class GraphSeamTests(unittest.TestCase):
    """图还没建，但接缝天天在跑——所以它不会烂。"""

    def test_search_works_with_no_graph_at_all(self) -> None:
        index, now = indexed()
        out = search(
            RecallDeps(index=index, graph=NullGraphStore()),
            "中试基地",
            entity_names=["零次方"],
            now_ms=now,
        )
        self.assertTrue(out["results"])
        self.assertTrue(any("知识图谱未建" in reason for reason in out["retrieval"]["degraded"]))

    def test_a_graph_narrows_the_search_to_its_entities(self) -> None:
        class _Graph:
            def resolve(self, name, *, scope=""):
                return [EntityMatch(entity_id="ent:中试基地", canonical=name)] if name else []

            def neighbors(self, entity_id, *, hops=1, kinds=()):
                return []

            def evidence_nodes(self, entity_id):
                return []

        index, now = indexed()
        target = index.lexical_candidates("中试基地")[0]
        index.link_entities(target, [EntityRef(entity_id="ent:中试基地", mention="中试基地")])

        out = search(
            RecallDeps(index=index, graph=_Graph()), "基地", entity_names=["中试基地"], now_ms=now
        )

        # 命中的是那个窗口，返回的是包住它的展示单元
        self.assertEqual(
            [item["id"] for item in out["results"]], [index.display_node(target)["id"]]
        )

    def test_neighbour_expansion_only_runs_when_asked(self) -> None:
        class _Graph(NullGraphStore):
            def resolve(self, name, *, scope=""):
                return [EntityMatch(entity_id="a", canonical=name)]

            def neighbors(self, entity_id, *, hops=1, kinds=()):
                return [EntityMatch(entity_id="b", canonical="邻居")]

        graph = _Graph()
        self.assertEqual(expand_entity_filter(graph, ["x"]), ["a"])
        self.assertEqual(expand_entity_filter(graph, ["x"], hops=1), ["a", "b"])


class ChatCorpusTests(unittest.TestCase):
    def test_chat_and_documents_share_one_index(self) -> None:
        index, now = indexed()
        index.upsert_tree(
            build_chat_tree(
                source_id="chat:c1",
                title="报市材料讨论",
                turns=[
                    {
                        "title": "第 1 轮",
                        "occurred_at": now,
                        "messages": [{"role": "user", "content": "中试基地那段再补一句资金安排。"}],
                    }
                ],
            )
        )

        kinds = {
            item["source_kind"]
            for item in search(RecallDeps(index=index), "中试基地", top_k=8, now_ms=now)["results"]
        }

        self.assertEqual(kinds, {"document", "chat"})


class BackendConfigTests(unittest.TestCase):
    def test_no_key_means_unavailable_rather_than_a_hardcoded_default(self) -> None:
        embedding = SiliconFlowEmbedding(api_key="")
        rerank = SiliconFlowRerank(api_key="")

        self.assertFalse(embedding.available)
        self.assertFalse(rerank.available)
        with self.assertRaises(RecallBackendError):
            embedding.embed(["x"])
        with self.assertRaises(RecallBackendError):
            rerank.rank("q", ["d"])

    def test_defaults_point_at_siliconflow(self) -> None:
        self.assertIn("siliconflow", SiliconFlowEmbedding().base_url)
        self.assertEqual(SiliconFlowEmbedding().model, "Pro/BAAI/bge-m3")
        self.assertEqual(SiliconFlowRerank().model, "BAAI/bge-reranker-v2-m3")


if __name__ == "__main__":
    unittest.main()
