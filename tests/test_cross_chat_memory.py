from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from work_agent_core.cross_chat_memory import CrossChatMemoryStore
from work_agent_core.session_store import SessionStore


class CrossChatMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.session_store = SessionStore(root, session_dir=root / "conversation_history" / "sessions")
        self.memory_store = CrossChatMemoryStore(self.session_store)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def add(self, content: str, *, kind: str = "fact", project_id: str = "", state: str = "automatic"):
        return self.memory_store.upsert_many(
            [{"kind": kind, "content": content}], conversation_id="chat-one",
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
        memory = self.add("用户偏好中文。", kind="preference")
        self.memory_store.update(memory["id"], "用户偏好正式中文。")
        refreshed = self.add("用户偏好中文。", kind="preference")
        self.assertEqual(refreshed["content"], "用户偏好正式中文。")
        self.assertEqual(refreshed["state"], "corrected")

    def test_deleted_record_is_hidden(self) -> None:
        memory = self.add("应删除的记忆。")
        self.memory_store.delete(memory["id"])
        self.assertEqual(self.memory_store.list(), [])
        self.assertEqual(self.memory_store.list(include_deleted=True)[0]["state"], "deleted")

    def test_project_and_account_scopes_are_isolated(self) -> None:
        self.add("普通聊天记忆。")
        self.add("项目 A 记忆。", project_id="project-a")
        self.add("项目 B 记忆。", project_id="project-b")
        account = self.memory_store.active_for_scope(scope="account")
        project_a = self.memory_store.active_for_scope(scope="project", project_id="project-a")
        self.assertEqual([item["content"] for item in account], ["普通聊天记忆。"])
        self.assertEqual([item["content"] for item in project_a], ["项目 A 记忆。"])

    def test_relevant_retrieval_and_profile_are_separate(self) -> None:
        self.add("用户在学习具身智能机器人。", kind="goal")
        self.add("用户偏好正式中文。", kind="preference")
        found = self.memory_store.relevant_for_scope(query="机器人学习", scope="account")
        self.assertEqual(found[0]["kind"], "goal")
        profile = self.memory_store.set_profile("## 学习\n用户正在学习具身智能。")
        self.assertIn("具身智能", profile["content"])


if __name__ == "__main__":
    unittest.main()
