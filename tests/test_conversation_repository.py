from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from work_agent_core.conversation_repository import ConversationRepository
from work_agent_core.session_log import ASSISTANT_MESSAGE, USER_MESSAGE
from work_agent_core.session_log_store import SessionLogStore
from work_agent_core.session_store import ConversationSession, SessionStore


class ConversationRepositoryTests(unittest.TestCase):
    def make_repository(self, root: Path) -> tuple[ConversationRepository, SessionStore, SessionLogStore]:
        session_store = SessionStore(root, session_dir=root / "sessions")
        log_store = SessionLogStore(root / "session_log.sqlite3")
        repository = ConversationRepository(
            root,
            session_store=session_store,
            log_store=log_store,
        )
        return repository, session_store, log_store

    def test_first_load_migrates_legacy_and_future_load_ignores_json_divergence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repository, session_store, log_store = self.make_repository(root)
            session_store.save(
                ConversationSession(
                    id="chat-1",
                    messages=[
                        {"role": "user", "content": "问题"},
                        {"role": "assistant", "content": "答案"},
                    ],
                    summary="断点",
                    summary_message_count=2,
                    metadata={"title": "原始标题", "project_id": "p1"},
                )
            )

            migrated = repository.load("chat-1")
            self.assertEqual([item["content"] for item in migrated.messages], ["问题", "答案"])
            self.assertEqual(migrated.summary, "断点")
            self.assertEqual(migrated.metadata["title"], "原始标题")
            self.assertGreater(log_store.next_seq("chat-1"), 0)

            divergent = session_store.load("chat-1")
            divergent.messages = [{"role": "user", "content": "错误缓存"}]
            divergent.summary = "错误摘要"
            session_store.save(divergent)

            recovered = repository.load("chat-1")
            self.assertEqual([item["content"] for item in recovered.messages], ["问题", "答案"])
            self.assertEqual(recovered.summary, "断点")

    def test_checkpoint_is_working_set_and_later_events_are_replayed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repository, _session_store, log_store = self.make_repository(root)
            session = repository.load(
                "chat-2",
                display_messages=[
                    {"role": "assistant", "content": "你好"},
                    {"role": "user", "content": "旧问题"},
                    {"role": "assistant", "content": "旧答案"},
                    {"role": "user", "content": "本轮问题"},
                ],
            )
            session.messages = [
                {"role": "system", "content": "压缩摘要"},
                {"role": "user", "content": "旧问题"},
                {"role": "assistant", "content": "旧答案"},
            ]
            session.summary = "压缩摘要"
            session.summary_message_count = 3
            repository.checkpoint(session)

            log_store.append_event("chat-2", USER_MESSAGE, {"content": "新问题"})
            log_store.append_event("chat-2", ASSISTANT_MESSAGE, {"content": "新答案"})
            recovered = repository.load("chat-2")

            self.assertEqual(
                [item["content"] for item in recovered.messages],
                ["压缩摘要", "旧问题", "旧答案", "新问题", "新答案"],
            )
            self.assertEqual(recovered.summary, "压缩摘要")
            self.assertEqual(recovered.summary_message_count, 3)


if __name__ == "__main__":
    unittest.main()
