from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from work_agent_core.cross_chat_memory import CrossChatMemoryStore
from work_agent_core.session_store import ConversationSession, SessionStore


class CrossChatMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.session_store = SessionStore(root, session_dir=root / "conversation_history" / "sessions")
        self.memory_store = CrossChatMemoryStore(self.session_store)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def save_summary(
        self,
        conversation_id: str,
        summary: str,
        *,
        project_id: str = "",
        title: str = "来源聊天",
    ) -> None:
        self.session_store.save(
            ConversationSession(
                id=conversation_id,
                messages=[{"role": "user", "content": "原始聊天内容"}],
                summary=summary,
                summary_message_count=1,
                metadata={"project_id": project_id, "title": title},
            )
        )

    def test_sync_creates_traceable_memories_from_summaries(self) -> None:
        self.save_summary("chat-one", "已确认一期预算为 280 万元。", title="预算讨论")

        memories = self.memory_store.list()

        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0]["conversation_id"], "chat-one")
        self.assertEqual(memories[0]["conversation_title"], "预算讨论")
        self.assertEqual(memories[0]["content"], "已确认一期预算为 280 万元。")
        self.assertEqual(memories[0]["state"], "automatic")

    def test_correction_survives_later_summary_sync(self) -> None:
        self.save_summary("chat-one", "自动摘要第一版。")
        memory = self.memory_store.list()[0]
        corrected = self.memory_store.update(memory["id"], "用户纠正后的稳定表述。")
        self.assertEqual(corrected["state"], "corrected")

        session = self.session_store.load("chat-one")
        session.summary = "自动摘要第二版，增加了新进展。"
        self.session_store.save(session)
        refreshed = self.memory_store.list()[0]

        self.assertEqual(refreshed["content"], "用户纠正后的稳定表述。")
        self.assertEqual(refreshed["source_summary"], "自动摘要第二版，增加了新进展。")
        self.assertEqual(refreshed["state"], "corrected")

    def test_deleted_memory_is_not_recreated_by_sync(self) -> None:
        self.save_summary("chat-one", "稍后应被删除的摘要。")
        memory = self.memory_store.list()[0]
        self.memory_store.delete(memory["id"])

        session = self.session_store.load("chat-one")
        session.summary = "来源摘要后来又更新了。"
        self.session_store.save(session)

        self.assertEqual(self.memory_store.list(), [])
        deleted = self.memory_store.list(include_deleted=True)
        self.assertEqual(deleted[0]["state"], "deleted")

    def test_project_and_account_scopes_are_isolated(self) -> None:
        self.save_summary("account-chat", "普通聊天记忆。")
        self.save_summary("project-a-chat", "项目 A 的记忆。", project_id="project-a")
        self.save_summary("project-b-chat", "项目 B 的记忆。", project_id="project-b")

        account = self.memory_store.active_for_scope(scope="account")
        project_a = self.memory_store.active_for_scope(scope="project", project_id="project-a")

        self.assertEqual([item["conversation_id"] for item in account], ["account-chat"])
        self.assertEqual([item["conversation_id"] for item in project_a], ["project-a-chat"])

    def test_archive_only_summary_is_imported_with_source(self) -> None:
        archive_path = self.session_store.session_dir.parent / "conversations.json"
        archive_path.write_text(
            json.dumps(
                {
                    "items": [
                        {
                            "id": "archive-chat",
                            "title": "归档聊天",
                            "projectId": "project-a",
                            "contextSummary": "只存在于前端归档的摘要。",
                            "contextSummaryMessageCount": 12,
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        memory = self.memory_store.list()[0]
        self.assertEqual(memory["conversation_title"], "归档聊天")
        self.assertEqual(memory["project_id"], "project-a")
        self.assertEqual(memory["summary_message_count"], 12)


if __name__ == "__main__":
    unittest.main()
