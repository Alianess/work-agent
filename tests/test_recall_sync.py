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

        # 只有新一轮的节点需要重算；旧轮次的向量原地保留
        self.assertGreater(report.added, 0)
        self.assertGreaterEqual(report.unchanged, 6)
        self.assertEqual(index.stats()["vectors"], vectors_before)
        self.assertGreater(sync.vector_debt("bge-m3"), 0)

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

        self.assertGreater(sync.vector_debt("bge-m3"), 0)


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


class AutoIndexTests(unittest.TestCase):
    """新内容要自己进索引，不靠谁记得去调。"""

    def test_a_finished_turn_indexes_itself(self) -> None:
        from work_agent_core.recall.tools import index_conversation_async, recall_index_for

        root = Path(tempfile.mkdtemp())
        log = SessionLog(SessionHeader(session_id="c-new"))
        log.append("turn/start", {"turn_id": "t1"})
        log.append("step/start", {"step": 1})
        log.append("user/message", {"content": "罍街展厅的施工延期会影响十月对接会吗？"})
        log.append("assistant/message", {"content": "会，展厅施工延期约两个月。"})
        log.append("turn/end", {"reason": "completed"})

        index_conversation_async(root, "c-new", log, title="罍街展厅").join(timeout=10)

        index = recall_index_for(root)
        hits = index.lexical_candidates("罍街 施工 延期")
        self.assertTrue(hits)
        self.assertEqual(index.node(hits[0])["source_kind"], "chat")

    def test_any_workspace_write_notifies_even_without_an_explicit_handler(self) -> None:
        """十五处地方各自构造 WorkspaceFiles，靠"记得传"必然漏。

        实测会议纪要技能就漏了——它写的 ASR 转写稿和纪要从来没进过索引。
        """

        from work_agent_core.tools import WorkspaceFiles, set_default_file_change_handler

        seen: list[Path] = []
        set_default_file_change_handler(seen.append)
        self.addCleanup(set_default_file_change_handler, None)
        root = Path(tempfile.mkdtemp())

        WorkspaceFiles(root).write_text({"path": "asr_full/x/transcript.md", "content": "转写内容。"})

        self.assertEqual([path.name for path in seen], ["transcript.md"])

    def test_an_explicit_handler_still_wins(self) -> None:
        from work_agent_core.tools import WorkspaceFiles, set_default_file_change_handler

        fallback: list[Path] = []
        explicit: list[Path] = []
        set_default_file_change_handler(fallback.append)
        self.addCleanup(set_default_file_change_handler, None)

        WorkspaceFiles(
            Path(tempfile.mkdtemp()), on_file_changed=explicit.append
        ).write_text({"path": "a.md", "content": "x"})

        self.assertEqual([path.name for path in explicit], ["a.md"])
        self.assertEqual(fallback, [])


class TranscriptClassificationTests(unittest.TestCase):
    def test_our_own_transcript_output_is_not_filed_as_a_document(self) -> None:
        from work_agent_core.recall.chunking import classify_source

        for path in (
            "meet_files/asr_full/x/transcript.txt",
            "meet_files/文字稿/20260731-新录音_会议沟通内容整理_ASR转写稿_Qwen3.md",
            "meet_files/asr_outputs/b.txt",
        ):
            self.assertEqual(classify_source(Path(path), Path(".")), "transcript", path)

    def test_real_materials_stay_documents(self) -> None:
        from work_agent_core.recall.chunking import classify_source

        for path in ("meet_files/材料/报市稿.docx", "meet_files/材料/0707会议纪要.docx"):
            self.assertEqual(classify_source(Path(path), Path(".")), "document", path)


class PoisonedLogTests(unittest.TestCase):
    def test_base64_never_reaches_the_index(self) -> None:
        """写入侧修好了，读取侧仍要扛得住历史数据。"""

        from work_agent_core.recall.sync import readable_content

        blocks = [
            {"type": "text", "text": "我记的笔记"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQSkZJRg"}},
        ]
        self.assertEqual(readable_content(blocks), "我记的笔记")
        self.assertEqual(
            readable_content("看这个 data:image/png;base64,iVBORw0KGgo 就是它"),
            "看这个 [图片] 就是它",
        )


if __name__ == "__main__":
    unittest.main()
