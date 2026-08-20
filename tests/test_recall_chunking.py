"""切片的判据只有一条：每个叶子单独拿出来仍然读得懂。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from work_agent_core.recall.chunking import (
    build_chat_tree,
    build_document_tree,
    build_file_tree,
    detect_heading,
    split_segments,
)
from work_agent_core.recall.nodes import estimate_tokens, node_id


REPORT = """# 关于我市具身智能产业发展的思考和建议

一、当前存在的问题

（一）本体企业少，链条不完整

我市现有具身智能相关企业十余家，多集中在零部件环节，整机本体企业稀缺。
零次方、逐际动力等头部企业均未在我市设立生产基地。

（二）应用场景开放不足

场景开放缺乏统一入口，企业反映找不到验证机会。

二、几点建议

1. 建设中试基地

建议依托现有闲置厂房，建设具身智能中试基地，解决样机验证场地不足的问题。

| 序号 | 事项 | 责任单位 |
| --- | --- | --- |
| 1 | 场地选址 | 工信局 |
| 2 | 资金安排 | 财政局 |
"""


class HeadingTests(unittest.TestCase):
    def test_markdown_and_chinese_numbering_both_read_as_headings(self) -> None:
        self.assertEqual(detect_heading("## 小节"), (2, "小节"))
        self.assertEqual(detect_heading("一、当前存在的问题")[0], 1)
        self.assertEqual(detect_heading("（二）建设中试基地")[0], 2)
        self.assertEqual(detect_heading("3. 人才引进")[0], 3)
        self.assertEqual(detect_heading("（1）细则")[0], 4)

    def test_a_long_numbered_sentence_is_body_not_a_heading(self) -> None:
        body = (
            "一、二线城市在具身智能产业上普遍面临同样的困难，主要体现在四个方面，"
            "需要逐一分析并给出对策建议，以下展开说明其中的缘由与可行路径。"
        )
        self.assertIsNone(detect_heading(body))

    def test_plain_text_is_not_a_heading(self) -> None:
        self.assertIsNone(detect_heading("这是一段普通正文。"))


class DocumentTreeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tree = build_document_tree(
            source_id="doc:报市稿",
            title="关于我市具身智能产业发展的思考和建议",
            text=REPORT,
            occurred_at=1_755_000_000_000,
        )

    def test_chinese_numbering_nests(self) -> None:
        paths = {" / ".join(node.path[1:]) for node in self.tree.nodes if not node.is_leaf}
        self.assertIn("一、当前存在的问题", paths)
        self.assertIn("一、当前存在的问题 / （一）本体企业少，链条不完整", paths)
        self.assertIn("二、几点建议 / 1. 建设中试基地", paths)

    def test_the_file_title_does_not_become_its_own_child(self) -> None:
        titles = [node.title for node in self.tree.nodes]
        self.assertEqual(titles.count("关于我市具身智能产业发展的思考和建议"), 1)

    def test_every_leaf_carries_the_path_that_makes_it_readable(self) -> None:
        for leaf in self.tree.leaves():
            self.assertTrue(leaf.path, leaf.text[:20])
            self.assertTrue(leaf.text.strip())

    def test_table_rows_keep_their_header(self) -> None:
        table_leaves = [leaf for leaf in self.tree.leaves() if leaf.header]
        self.assertEqual(len(table_leaves), 1)
        self.assertIn("责任单位", table_leaves[0].header)
        self.assertIn("场地选址", table_leaves[0].text)
        # 表头必须进入可检索正文，否则 "责任单位" 这种词永远命中不到行
        self.assertIn("责任单位", table_leaves[0].searchable_body())

    def test_ancestors_go_from_the_display_unit_up_to_the_file(self) -> None:
        """窗口 → 展示段 → 章节 → 文件。

        中间那层是展示单元：命中在窗口上，返回的是包住它的这一段。
        """

        leaf = next(leaf for leaf in self.tree.leaves() if "中试基地" in leaf.text)
        chain = [node.title for node in self.tree.ancestors_of(leaf.id)]
        self.assertIn("1. 建设中试基地", chain)
        self.assertEqual(chain[-1], "关于我市具身智能产业发展的思考和建议")

    def test_a_matched_window_is_never_what_gets_shown(self) -> None:
        leaf = next(leaf for leaf in self.tree.leaves() if "中试基地" in leaf.text)
        parent = self.tree.by_id()[leaf.parent_id]
        # 展示单元自带完整正文，不是把重叠的窗口拼回去
        self.assertTrue(parent.text.strip())
        self.assertNotIn(leaf.text * 2, parent.text)

    def test_a_parent_reads_as_the_original_order(self) -> None:
        section = next(
            node
            for node in self.tree.nodes
            if node.title == "1. 建设中试基地"
        )
        # 正文在表格之前，和原文一致；重排过的父节点读起来会失真
        self.assertLess(section.text.index("中试基地，解决样机"), section.text.index("场地选址"))

    def test_expansion_costs_grow_with_height(self) -> None:
        leaf = next(leaf for leaf in self.tree.leaves() if "中试基地" in leaf.text)
        costs = [node.tokens for node in self.tree.ancestors_of(leaf.id)]
        self.assertEqual(costs, sorted(costs))
        self.assertGreater(costs[-1], leaf.tokens)


class SegmentOrderTests(unittest.TestCase):
    def test_tables_and_text_stay_in_document_order(self) -> None:
        segments = split_segments("前言\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n\n后记")
        self.assertEqual([kind for kind, _, _ in segments], ["text", "table", "text"])


class ChatTreeTests(unittest.TestCase):
    def test_a_message_leaf_hangs_under_its_whole_turn(self) -> None:
        tree = build_chat_tree(
            source_id="chat:c1",
            title="报市材料",
            turns=[
                {
                    "title": "第 1 轮",
                    "occurred_at": 1_755_000_000_000,
                    "messages": [
                        {"role": "user", "content": "那个材料按业务员口吻改一版。"},
                        {"role": "assistant", "content": "已改好，路径在 meet_files/材料 下。"},
                    ],
                }
            ],
        )
        leaves = tree.leaves()
        self.assertEqual(len(leaves), 2)
        self.assertTrue(leaves[0].text.startswith("用户："))
        # 聊天块不自足（"那个材料"），所以往上两层要能拿回完整一轮
        index = tree.by_id()
        turn = index[index[leaves[0].parent_id].parent_id]
        self.assertEqual(turn.title, "第 1 轮")
        self.assertIn("业务员口吻", turn.text)
        self.assertIn("路径在", turn.text)


class StableIdTests(unittest.TestCase):
    def test_reindexing_the_same_content_yields_the_same_ids(self) -> None:
        first = build_document_tree(source_id="doc:x", title="标题", text=REPORT)
        second = build_document_tree(source_id="doc:x", title="标题", text=REPORT)
        self.assertEqual([node.id for node in first.nodes], [node.id for node in second.nodes])

    def test_different_paths_never_collide(self) -> None:
        self.assertNotEqual(node_id("doc:x", ["a"], 1), node_id("doc:x", ["b"], 1))


class FileTreeTests(unittest.TestCase):
    def test_a_markdown_file_indexes_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "报市稿.md"
            path.write_text(REPORT, encoding="utf-8")

            tree = build_file_tree(path)

            self.assertTrue(tree.leaves())
            self.assertEqual(tree.root().title, "报市稿.md")
            self.assertGreater(tree.root().occurred_at, 0)


class TokenEstimateTests(unittest.TestCase):
    def test_chinese_counts_about_one_token_per_character(self) -> None:
        self.assertEqual(estimate_tokens("一二三四五六七八九十"), 10)

    def test_ascii_counts_about_four_characters_per_token(self) -> None:
        self.assertEqual(estimate_tokens("a" * 40), 10)


if __name__ == "__main__":
    unittest.main()
