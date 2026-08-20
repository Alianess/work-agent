"""索引跟上语料增长的两条不变量：写入不挡回复，增量到节点。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from work_agent_core.recall.index import RecallIndex
from work_agent_core.recall.sync import RecallSync, turns_from_log
from work_agent_core.session_log import SessionHeader, SessionLog


class _Embedding:
    model = "bge-m3"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [[float(len(text) % 7), 1.0, 0.0, 0.0] for text in texts]


class _BrokenEmbedding:
    model = "bge-m3"

    def embed(self, texts):
        raise RuntimeError("endpoint down")


def conversation(turn_count: int) -> SessionLog:
    log = SessionLog(SessionHeader(session_id="c1"))
    for index in range(1, turn_count + 1):
        log.append("turn/start", {"turn_id": f"t{index}"})
        log.append("step/start", {"step": 1})
        log.append(
            "user/message",
            {"content": f"第{index}轮：中试基地的场地和资金到哪一步了，请给最新情况。"},
        )
        log.append("assistant/message", {"content": f"第{index}轮：场地已定，资金待批。"})
        log.append("turn/end", {"reason": "completed"})
    return log


def fresh() -> tuple[RecallIndex, RecallSync]:
    index = RecallIndex(Path(tempfile.mkdtemp()) / "recall.sqlite3")
    return index, RecallSync(index)


class TurnBoundaryTests(unittest.TestCase):
    def test_messages_group_under_their_turn(self) -> None:
        turns = turns_from_log(conversation(2))
        self.assertEqual(len(turns), 2)
        self.assertEqual([message["role"] for message in turns[0]["messages"]], ["user", "assistant"])

    def test_an_empty_turn_is_not_indexed(self) -> None:
        log = SessionLog(SessionHeader(session_id="c1"))
        log.append("turn/start", {"turn_id": "t1"})
        log.append("turn/end", {"reason": "aborted"})
        self.assertEqual(turns_from_log(log), [])


class IncrementalConversationTests(unittest.TestCase):
    def test_appending_a_turn_reuses_the_vectors_of_earlier_turns(self) -> None:
        index, sync = fresh()
        embedding = _Embedding()
        sync.index_conversation("c1", conversation(3), title="报市材料")
        sync.backfill_vectors(embedding)
        vectors_before = index.stats()["vectors"]
        self.assertEqual(sync.vector_debt("bge-m3"), 0)

        report = sync.index_conversation("c1", conversation(4), title="报市材料")

        # 只有新一轮的叶子需要重算；旧轮次的向量原地保留
        self.assertEqual(report.added, 3)
        self.assertGreaterEqual(report.unchanged, 6)
        self.assertEqual(index.stats()["vectors"], vectors_before)
        self.assertEqual(sync.vector_debt("bge-m3"), 2)

    def test_a_second_pass_over_unchanged_content_does_nothing(self) -> None:
        _, sync = fresh()
        sync.index_conversation("c1", conversation(2), title="x")
        self.assertTrue(sync.index_conversation("c1", conversation(2), title="x").skipped)

    def test_editing_a_leaf_invalidates_only_that_vector(self) -> None:
        index, sync = fresh()
        embedding = _Embedding()
        sync.index_conversation("c1", conversation(2), title="x")
        sync.backfill_vectors(embedding)

        edited = conversation(2)
        edited.append("turn/start", {"turn_id": "t3"})
        edited.append("user/message", {"content": "第3轮：资金批了吗？"})
        edited.append("turn/end", {"reason": "completed"})
        sync.index_conversation("c1", edited, title="x")

        self.assertEqual(sync.vector_debt("bge-m3"), 1)


class BackfillTests(unittest.TestCase):
    def test_a_failing_endpoint_never_breaks_lexical_recall(self) -> None:
        index, sync = fresh()
        sync.index_conversation("c1", conversation(2), title="x")

        outcome = sync.backfill_vectors(_BrokenEmbedding())

        self.assertEqual(outcome["written"], 0)
        self.assertIn("error", outcome)
        # 词法索引是本地的，端点挂了也照样能找到
        self.assertTrue(index.lexical_candidates("中试基地"))

    def test_backfill_runs_in_bounded_batches(self) -> None:
        _, sync = fresh()
        embedding = _Embedding()
        sync.index_conversation("c1", conversation(6), title="x")

        outcome = sync.backfill_vectors(embedding, budget=2)

        self.assertEqual(outcome["written"], 2)
        self.assertTrue(outcome["more"])
        self.assertEqual(len(embedding.calls[0]), 2)

    def test_embedding_text_carries_the_heading_path(self) -> None:
        _, sync = fresh()
        embedding = _Embedding()
        sync.index_conversation("c1", conversation(1), title="报市材料")

        sync.backfill_vectors(embedding)

        self.assertTrue(any("报市材料" in text for text in embedding.calls[0]))


class FileSyncTests(unittest.TestCase):
    def test_a_directory_sweep_indexes_only_indexable_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "报市稿.md").write_text("# 报市稿\n\n一、问题\n\n中试基地缺场地。\n", encoding="utf-8")
            (base / "notes.txt").write_text("零次方的联系人是谁", encoding="utf-8")
            (base / "binary.bin").write_bytes(b"\x00\x01")
            (base / ".git").mkdir()
            (base / ".git" / "hidden.md").write_text("不该被索引", encoding="utf-8")

            index, sync = fresh()
            report = sync.index_directory(base)

            self.assertEqual(report.sources, 2)
            self.assertTrue(index.lexical_candidates("中试基地"))
            self.assertEqual(index.lexical_candidates("不该被索引"), [])

    def test_reindexing_an_unchanged_file_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a.md"
            path.write_text("# a\n\n一、问题\n\n中试基地缺场地。\n", encoding="utf-8")
            _, sync = fresh()

            sync.index_file(path)

            self.assertTrue(sync.index_file(path).skipped)


if __name__ == "__main__":
    unittest.main()
