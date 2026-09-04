from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from work_agent_core.cross_chat_memory import CrossChatMemoryStore, register_memory_tools
from work_agent_core.recall.tools import register_recall_tools
from work_agent_core.session_store import SessionStore
from work_agent_core.tools import ToolRegistry
from work_agent_core.web_server import explicit_memory_content


class CrossChatMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.session_store = SessionStore(root, session_dir=root / "conversation_history" / "sessions")
        self.memory_store = CrossChatMemoryStore(self.session_store)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def add(self, content: str, *, kind: str = "preference", project_id: str = "", state: str = "automatic"):
        record = {"kind": kind, "content": content}
        if state == "automatic":
            record.update(
                {
                    "importance": 0.96,
                    "confidence": 0.97,
                    "durability": "long_term",
                    "evidence": "explicit",
                }
            )
        return self.memory_store.upsert_many(
            [record], conversation_id="chat-one",
            conversation_title="来源聊天", project_id=project_id, source_excerpt="原始聊天证据", state=state,
        )[0]

    def test_records_are_source_backed_and_manageable(self) -> None:
        memory = self.add("用户偏好中文且希望先给结论。", kind="preference")
        self.assertEqual(memory["conversation_title"], "来源聊天")
        self.assertEqual(memory["source_excerpt"], "原始聊天证据")
        self.assertEqual(memory["kind"], "preference")
        corrected = self.memory_store.update(memory["id"], "用户偏好中文，先给结论再展开。")
        self.assertEqual(corrected["state"], "corrected")

    def test_corrected_record_is_not_overwritten_by_automatic_refresh(self) -> None:
        memory = self.add("用户长期偏好使用中文沟通。", kind="preference")
        self.memory_store.update(memory["id"], "用户偏好正式中文。")
        refreshed = self.add("用户长期偏好使用中文沟通。", kind="preference")
        self.assertEqual(refreshed["content"], "用户偏好正式中文。")
        self.assertEqual(refreshed["state"], "corrected")

    def test_deleted_record_is_hidden(self) -> None:
        memory = self.add("用户长期偏好保留原始材料。")
        self.memory_store.delete(memory["id"])
        self.assertEqual(self.memory_store.list(), [])
        self.assertEqual(self.memory_store.list(include_deleted=True)[0]["state"], "deleted")

    def test_project_and_account_scopes_are_isolated(self) -> None:
        self.add("用户长期偏好先核验原始材料。")
        self.add("项目 A 的长期合作边界。", kind="project", project_id="project-a")
        self.add("项目 B 的长期合作边界。", kind="project", project_id="project-b")
        account = self.memory_store.active_for_scope(scope="account")
        project_a = self.memory_store.active_for_scope(scope="project", project_id="project-a")
        self.assertEqual([item["content"] for item in account], ["用户长期偏好先核验原始材料。"])
        self.assertEqual([item["content"] for item in project_a], ["项目 A 的长期合作边界。"])

    def test_relevant_retrieval_and_profile_are_separate(self) -> None:
        self.add("用户在学习具身智能机器人。", kind="goal")
        self.add("用户偏好正式中文。", kind="preference")
        found = self.memory_store.relevant_for_scope(query="机器人学习", scope="account")
        self.assertEqual(found[0]["kind"], "goal")
        profile = self.memory_store.set_profile("## 学习\n用户正在学习具身智能。")
        self.assertIn("具身智能", profile["content"])

    def test_automatic_memory_rejects_details_and_low_confidence(self) -> None:
        saved = self.memory_store.upsert_many(
            [
                {
                    "kind": "fact",
                    "content": "公积金每月 2956 元。",
                    "importance": 0.99,
                    "confidence": 0.99,
                    "durability": "long_term",
                    "evidence": "explicit",
                },
                {
                    "kind": "preference",
                    "content": "用户偏好重要工作先给结论。",
                    "importance": 0.70,
                    "confidence": 0.99,
                    "durability": "long_term",
                    "evidence": "explicit",
                },
                {
                    "kind": "preference",
                    "content": "用户偏好重要工作先给结论，再给可执行依据。",
                    "importance": 0.96,
                    "confidence": 0.97,
                    "durability": "long_term",
                    "evidence": "repeated",
                },
            ],
            conversation_id="chat-one",
            conversation_title="来源聊天",
            state="automatic",
        )
        self.assertEqual(len(saved), 1)
        self.assertIn("可执行依据", saved[0]["content"])

    def test_automatic_memory_has_a_hard_scope_limit(self) -> None:
        preferences = [
            "用户长期偏好先给明确结论，再补充必要依据。",
            "用户长期偏好使用简洁自然的正式中文表达。",
            "用户长期偏好对未确认事实明确标注不确定性。",
            "用户长期偏好先检查真实文件和运行状态再判断。",
            "用户长期偏好采用自下而上的稳定开发顺序。",
            "用户长期偏好避免无必要的复杂项目管理框架。",
            "用户长期偏好保留原始材料并从原文核验细节。",
            "用户长期偏好重大变更先说明边界和退出条件。",
            "用户长期偏好界面保持高信息密度并减少滚动。",
            "用户长期偏好诊断时定位根因而非只给临时方案。",
            "用户长期偏好将项目事实和个人偏好分开保存。",
            "用户长期偏好完成开发后进行真实界面验证。",
        ]
        for index, content in enumerate(preferences):
            saved = self.memory_store.upsert_many(
                [
                    {
                        "kind": "preference",
                        "content": content,
                        "importance": 0.91 + index * 0.005,
                        "confidence": 0.98,
                        "durability": "long_term",
                        "evidence": "repeated",
                    }
                ],
                conversation_id=f"chat-{index}",
                conversation_title="来源聊天",
                state="automatic",
            )
            self.assertLessEqual(len(saved), 1)
        self.assertEqual(len(self.memory_store.list(project_id="")), 12)

    def test_corrected_memories_still_count_toward_the_hard_limit(self) -> None:
        contents = [f"用户明确要求长期保留的核心偏好条目 {index:02d}。" for index in range(32)]
        memories = []
        for content in contents:
            memories.extend(self.memory_store.upsert_many(
                [{"kind": "preference", "content": content}],
                conversation_id="chat-core",
                conversation_title="来源聊天",
                state="explicit",
            ))
        for memory in memories:
            self.memory_store.update(memory["id"], memory["content"])
        rejected = self.memory_store.upsert_many(
            [{
                "kind": "preference",
                "content": "用户长期偏好将执行失败直接说明并保留现场。",
                "importance": 0.99,
                "confidence": 0.99,
                "durability": "long_term",
                "evidence": "explicit",
            }],
            conversation_id="chat-extra",
            conversation_title="来源聊天",
            state="automatic",
        )
        self.assertEqual(rejected, [])
        self.assertEqual(len(self.memory_store.list(project_id="")), 32)

    def test_direct_memory_request_is_captured_without_confusing_a_question(self) -> None:
        self.assertEqual(
            explicit_memory_content("请记住：我长期偏好先核验原始材料，再给结论。"),
            "我长期偏好先核验原始材料，再给结论",
        )
        self.assertEqual(explicit_memory_content("你记住我的要求了吗？"), "")


class MemoryToolTests(unittest.TestCase):
    """remember/forget：模型自己的记忆之手，护栏照常生效。"""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.session_store = SessionStore(root, session_dir=root / "conversation_history" / "sessions")
        self.store = CrossChatMemoryStore(self.session_store)
        self.registry = ToolRegistry()
        register_memory_tools(
            self.registry,
            session_store=self.session_store,
            conversation_id="conv-1",
            conversation_title="对话一",
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _remember(self, **args: str) -> dict:
        payload = self.registry.get("remember").handler(args)
        return json.loads(payload)

    def test_recall_attaches_relevant_memories(self) -> None:
        """检索命中连带相关核心记忆——模型不必再为记忆另开一问。"""

        self._remember(
            content="用户报告永远用业务员口吻写作。",
            kind="preference",
            source_excerpt="以后报告都按业务员的口吻写",
        )
        recall_registry = ToolRegistry()
        register_recall_tools(
            recall_registry,
            Path(self.temp_dir.name) / "recall_root",
            session_store=self.session_store,
        )
        outcome = json.loads(
            recall_registry.get("recall").handler({"query": "报告用什么口吻写"})
        )
        self.assertTrue(outcome.get("memory_results"))
        self.assertIn("业务员口吻", outcome["memory_results"][0]["content"])
        # 无关查询不带记忆，避免每轮都拖着不相关条目。
        quiet = json.loads(
            recall_registry.get("recall").handler({"query": "量子计算原理"})
        )
        self.assertFalse(quiet.get("memory_results"))

    def test_remember_writes_explicit_memory(self) -> None:
        outcome = self._remember(
            content="用户报告永远用业务员口吻写作。",
            kind="preference",
            source_excerpt="以后报告都按业务员的口吻写",
        )
        self.assertTrue(outcome["saved"])
        items = self.store.list()
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["state"], "explicit")
        self.assertEqual(items[0]["source_excerpt"], "以后报告都按业务员的口吻写")

    def test_remember_rejects_paths_and_transient_state(self) -> None:
        rejected = self._remember(content="材料都放在 meet_files/材料 目录下。")
        self.assertFalse(rejected["saved"])
        self.assertIn("路径", rejected["reason"])
        transient = self._remember(content="本次会议纪要初稿已经生成。")
        self.assertFalse(transient["saved"])
        too_short = self._remember(content="太短")
        self.assertFalse(too_short["saved"])

    def test_remember_deduplicates_identical_content(self) -> None:
        self._remember(content="用户报告永远用业务员口吻写作。")
        again = self._remember(content="用户报告永远用业务员口吻写作。")
        self.assertFalse(again["saved"])
        self.assertEqual(len(self.store.list()), 1)

    def test_remember_rejects_unknown_kind(self) -> None:
        with self.assertRaises(ValueError):
            self._remember(content="用户长期偏好先给结论再给依据。", kind="mood")

    def test_forget_removes_the_closest_match(self) -> None:
        self._remember(content="用户报告永远用业务员口吻写作。")
        outcome = json.loads(
            self.registry.get("forget").handler({"content": "报告用业务员口吻"})
        )
        self.assertTrue(outcome["deleted"])
        self.assertEqual(self.store.list(), [])


if __name__ == "__main__":
    unittest.main()
