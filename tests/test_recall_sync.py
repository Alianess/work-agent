"""索引跟上语料增长的两条不变量：写入不挡回复，增量到节点。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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

    def test_tool_results_enter_lexical_but_never_vectors(self) -> None:
        """命令回显要能被词法找回，但不该花 embedding 钱、稀释语义索引。"""

        log = SessionLog(SessionHeader(session_id="c1"))
        log.append("turn/start", {"turn_id": "t1"})
        log.append("user/message", {"content": "查一下构建为什么失败"})
        log.append("tool/call", {"call_id": "a", "name": "run_command", "arguments": "{}"})
        log.append(
            "tool/result",
            {"call_id": "a", "name": "run_command", "content": "Error: xcodebuild exit 65 见 meet_files 目录"},
        )
        log.append("assistant/message", {"content": "构建失败的原因是签名配置缺失。"})
        index, sync = fresh()
        sync.index_conversation("c1", log, title="构建排查")
        turns = turns_from_log(log)
        self.assertEqual(
            [message["role"] for message in turns[0]["messages"]],
            ["user", "tool", "assistant"],
        )
        # 词法找得到回显原文
        hits = index.lexical_candidates("xcodebuild exit", limit=5)
        self.assertTrue(hits)
        # 向量补算名单里没有工具结果叶子
        pending = index.leaves_without_vectors("bge-m3")
        with index._connect() as connection:
            titles = {
                str(row[0])
                for row in connection.execute(
                    "SELECT DISTINCT title FROM recall_nodes WHERE id IN"
                    f" ({','.join('?' * len(pending))})",
                    [item["id"] for item in pending],
                )
            }
        self.assertNotIn("工具结果", titles)


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


class ProjectScopeTests(unittest.TestCase):
    """项目隔离：归属是过滤条件，不是语义问题。"""

    def _source_project(self, index: RecallIndex, source_id: str) -> str:
        with index._connect() as connection:  # noqa: SLF001 - 测试读同包内部
            row = connection.execute(
                "SELECT project_id FROM recall_sources WHERE source_id = ?", (source_id,)
            ).fetchone()
        return str(row["project_id"]) if row else ""

    def test_conversation_carries_its_project(self) -> None:
        index, sync = fresh()
        sync.index_conversation("c1", conversation(2), title="x", project_id="abc123def456")
        self.assertEqual(self._source_project(index, "chat:c1"), "abc123def456")

    def test_unchanged_content_still_backfills_project(self) -> None:
        """内容跳过省的是向量钱，不该把归属一起省掉。"""

        index, sync = fresh()
        sync.index_conversation("c1", conversation(2), title="x")
        self.assertEqual(self._source_project(index, "chat:c1"), "")

        report = sync.index_conversation(
            "c1", conversation(2), title="x", project_id="abc123def456"
        )

        self.assertTrue(report.skipped)
        self.assertEqual(self._source_project(index, "chat:c1"), "abc123def456")

    def test_empty_project_does_not_wipe_existing_attribution(self) -> None:
        index, sync = fresh()
        sync.index_conversation("c1", conversation(3), title="x", project_id="abc123def456")

        # 换了一轮内容重索引，但调用方没带项目信息——已有归属不能丢。
        sync.index_conversation("c1", conversation(4), title="x")

        self.assertEqual(self._source_project(index, "chat:c1"), "abc123def456")

    def test_project_files_are_recognised_by_path(self) -> None:
        from work_agent_core.recall.sync import project_id_from_path

        self.assertEqual(
            project_id_from_path("meet_files/projects/project-79da7d7ce843/sources/政策依据.txt"),
            "79da7d7ce843",
        )
        self.assertEqual(project_id_from_path("meet_files/材料/报市稿.docx"), "")

    def test_file_under_a_project_dir_inherits_its_project(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            project_dir = base / "projects" / "project-abc123def456" / "sources"
            project_dir.mkdir(parents=True)
            target = project_dir / "政策依据.txt"
            target.write_text("项目政策依据：专项债支持。\n", encoding="utf-8")
            outside = base / "材料.md"
            outside.write_text("散装材料。\n", encoding="utf-8")

            index, sync = fresh()
            sync.index_directory(base)

            self.assertEqual(
                self._source_project(index, f"doc:{target.relative_to(base).as_posix()}"),
                "abc123def456",
            )
            self.assertEqual(
                self._source_project(index, f"doc:{outside.relative_to(base).as_posix()}"),
                "",
            )

    def test_project_filter_limits_search_to_project_sources(self) -> None:
        from work_agent_core.recall.index import NodeFilter
        from work_agent_core.recall.search import RecallDeps, search

        index, sync = fresh()
        sync.index_conversation("c-in", conversation(2), title="项目内", project_id="abc123def456")
        sync.index_conversation("c-out", conversation(2), title="项目外")

        outcome = search(
            RecallDeps(index=index),
            "中试基地的场地",
            filters=NodeFilter(project_ids=("abc123def456",)),
        )

        self.assertTrue(outcome["results"])
        self.assertTrue(
            all(item["source_id"] == "chat:c-in" for item in outcome["results"])
        )

    def test_recall_tool_scope_project_degrades_without_a_project(self) -> None:
        from work_agent_core.recall.tools import register_recall_tools
        from work_agent_core.tools import ToolRegistry

        with tempfile.TemporaryDirectory() as directory:
            registry = ToolRegistry()
            register_recall_tools(registry, Path(directory))

            tool = registry.get("recall")
            self.assertEqual(tool.parameters["properties"]["scope"]["enum"], ["all"])
            # 会话不在项目里时 scope=project 不再报错：降级为账户级并说明原因。
            outcome = json.loads(
                registry.get("recall").handler({"query": "中试基地", "scope": "project"})
            )
            self.assertEqual(outcome["scope_note"], "当前会话不在任何项目里，scope=project 已自动降级为 scope=all。")

    def test_recall_excludes_the_fully_replayed_current_conversation_and_bounds_text(self) -> None:
        from work_agent_core.recall.tools import recall_index_for, register_recall_tools
        from work_agent_core.tools import ToolRegistry

        class _SessionStore:
            def load(self, _conversation_id: str):
                return SimpleNamespace(summary="", summary_message_count=0)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = recall_index_for(root)
            sync = RecallSync(index)
            current = SessionLog(SessionHeader(session_id="current"))
            current.append("turn/start", {"turn_id": "t1"})
            current.append("user/message", {"content": "寻找中试基地方案"})
            current.append(
                "assistant/message",
                {"content": "当前会话里经过核验的中试基地方案" * 10},
            )
            current.append("turn/end", {"reason": "completed"})
            other = SessionLog(SessionHeader(session_id="other"))
            other.append("turn/start", {"turn_id": "t1"})
            other.append("user/message", {"content": "寻找中试基地方案"})
            other.append(
                "assistant/message",
                {"content": "其他会话的中试基地方案：" + "经过核验的内容" * 200},
            )
            other.append("turn/end", {"reason": "completed"})
            sync.index_conversation("current", current, title="当前会话")
            sync.index_conversation("other", other, title="其他会话")
            registry = ToolRegistry()
            with (
                patch("work_agent_core.recall.tools.embedding_backend", return_value=None),
                patch("work_agent_core.recall.tools.rerank_backend", return_value=None),
                patch("work_agent_core.recall.tools.RECALL_RESULT_MAX_TEXT_CHARS", 20),
            ):
                register_recall_tools(
                    registry,
                    root,
                    session_store=_SessionStore(),
                    conversation_id="current",
                )
                outcome = json.loads(
                    registry.get("recall").handler({"query": "经过核验的内容 中试基地"})
                )

            self.assertTrue(outcome["results"])
            self.assertTrue(all(item["source_id"] != "chat:current" for item in outcome["results"]))
            self.assertTrue(any(item.get("text_truncated") for item in outcome["results"]))
            self.assertTrue(all(len(item["text"]) <= 21 for item in outcome["results"]))


class VersionFamilyTests(unittest.TestCase):
    """版本折叠：文件名是线索，内容重叠才是判决。"""

    def _write_doc(self, base: Path, name: str, paragraphs: list[str]) -> Path:
        target = base / name
        target.write_text("\n\n".join(paragraphs) + "\n", encoding="utf-8")
        return target

    def test_family_key_normalises_version_noise(self) -> None:
        from work_agent_core.recall.sync import version_family_key

        self.assertEqual(
            version_family_key("材料/17号可研报告.docx"),
            version_family_key("材料/20号可研报告.docx"),
        )
        self.assertEqual(
            version_family_key("材料/可行性研究报告_v1.md"),
            version_family_key("材料/可行性研究报告_终稿.md"),
        )
        self.assertEqual(
            version_family_key("材料/20260817-汇报.docx"),
            version_family_key("材料/20260820-汇报.docx"),
        )
        # 跨目录的同名是巧合不是版本
        self.assertNotEqual(
            version_family_key("材料/17号可研报告.docx"),
            version_family_key("归档/20号可研报告.docx"),
        )
        # "新材料"这类主干词不该被单字噪声吃掉
        self.assertEqual(
            version_family_key("材料/新材料产业报告.docx"),
            version_family_key("材料/新材料产业报告.docx"),
        )

    def test_high_overlap_folds_and_search_hides_old_versions(self) -> None:
        from work_agent_core.recall.index import NodeFilter
        from work_agent_core.recall.search import RecallDeps, search

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            # 真实规模：足够切成多个窗口，改写只影响尾部窗口，前部窗口跨文件
            # 共享——这正是"大部分没改、只改一段"的用户场景。
            filler = "低空经济发展的政策依据与产业现状，涉及空域管理、适航审定和基础设施建设多个方面。"
            shared = [f"共享段落{n}：{filler}" for n in range(8)]
            self._write_doc(base, "17号可研报告.md", shared + [f"旧版独有结论：{filler}"])
            index, sync = fresh()
            sync.index_directory(base)
            newer = self._write_doc(base, "20号可研报告.md", shared + [f"新版改写后的结论：{filler}"])
            import os
            os.utime(newer, (1_800_000_000, 1_800_000_000))
            sync.index_file(newer)

            outcome = search(
                RecallDeps(index=index), "共享段落", filters=NodeFilter()
            )
            sources = {item["source_id"] for item in outcome["results"]}
            self.assertTrue(sources)
            self.assertTrue(all("20号" in s for s in sources), sources)
            self.assertTrue(outcome["retrieval"]["superseded_excluded"])

            unfolded = search(
                RecallDeps(index=index),
                "旧版独有结论",
                filters=NodeFilter(include_superseded=True),
            )
            self.assertTrue(any("17号" in item["source_id"] for item in unfolded["results"]))

    def test_low_overlap_does_not_fold(self) -> None:
        """名字像版本、内容完全不同 → 两份独立材料，谁也不压谁。"""

        from work_agent_core.recall.index import NodeFilter
        from work_agent_core.recall.search import RecallDeps, search

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            filler_a = "甲方案聚焦整机集成，围绕飞行平台与载荷的协同设计展开论证。"
            filler_b = "乙方案聚焦空管系统，围绕通信导航监视设施的建设标准展开论证。"
            self._write_doc(
                base, "17号可研报告.md",
                [f"可研甲段落{n}：{filler_a}" for n in range(8)],
            )
            index, sync = fresh()
            sync.index_directory(base)
            newer = self._write_doc(
                base, "20号可研报告.md",
                [f"可研乙段落{n}：{filler_b}" for n in range(8)],
            )
            import os
            os.utime(newer, (1_800_000_000, 1_800_000_000))
            sync.index_file(newer)

            outcome = search(
                RecallDeps(index=index), "方案聚焦展开论证", filters=NodeFilter()
            )
            sources = {item["source_id"] for item in outcome["results"]}
            self.assertTrue(any("17号" in s for s in sources), sources)
            self.assertTrue(any("20号" in s for s in sources), sources)

    def test_reconcile_is_idempotent_and_recovers_after_forget(self) -> None:
        from work_agent_core.recall.sync import version_family_key

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            filler = "低空经济发展的政策依据与产业现状，涉及空域管理、适航审定和基础设施建设多个方面。"
            shared = [f"共享段落{n}：{filler}" for n in range(8)]
            old = self._write_doc(base, "17号可研报告.md", shared)
            index, sync = fresh()
            sync.index_directory(base)
            newer = self._write_doc(base, "20号可研报告.md", shared)
            import os
            os.utime(newer, (1_800_000_000, 1_800_000_000))
            sync.index_file(newer)

            first = index.reconcile_version_families(version_family_key)
            self.assertGreaterEqual(first["superseded"], 1)
            second = index.reconcile_version_families(version_family_key)
            self.assertEqual(second["superseded"], first["superseded"])

            # 新版文件被删后重算，旧版恢复可见
            old_key = f"doc:{old.relative_to(sync.workspace_root or base).as_posix()}"
            index.forget_source(
                f"doc:{newer.relative_to(sync.workspace_root or base).as_posix()}"
            )
            index.reconcile_version_families(version_family_key)
            info = index.superseded_info([old_key])
            self.assertEqual(info[old_key]["superseded_by"], "")


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
