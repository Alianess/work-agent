from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from work_agent_core import web_server
from work_agent_core.cross_chat_memory import CrossChatMemoryStore
from work_agent_core.history_recall import ensure_schema
from work_agent_core.session_store import ConversationSession, SessionStore


class ConversationArchiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary_directory.name).resolve()
        self.conversation_dir = self.workspace / "meet_files" / "conversation_history"
        self.history_path = self.conversation_dir / "conversations.json"
        self.workspace_patch = patch.object(web_server, "WORKSPACE_ROOT", self.workspace)
        self.dir_patch = patch.object(web_server, "user_conversation_dir", return_value=self.conversation_dir)
        self.path_patch = patch.object(
            web_server,
            "user_conversation_history_path",
            return_value=self.history_path,
        )
        self.cascade_session_store_patch = patch.object(
            web_server,
            "get_session_store",
            return_value=SessionStore(self.workspace, session_dir=self.conversation_dir / "sessions"),
        )
        self.cascade_turn_store_patch = patch.object(
            web_server,
            "get_turn_store",
            return_value=Mock(discard_pending_for_conversation=Mock(return_value=0)),
        )
        self.workspace_patch.start()
        self.dir_patch.start()
        self.path_patch.start()
        self.cascade_session_store_patch.start()
        self.cascade_turn_store_patch.start()

    def tearDown(self) -> None:
        self.path_patch.stop()
        self.dir_patch.stop()
        self.workspace_patch.stop()
        self.cascade_session_store_patch.stop()
        self.cascade_turn_store_patch.stop()
        self.temporary_directory.cleanup()

    def test_incremental_save_rejects_stale_revision_and_keeps_unrelated_item_file(self) -> None:
        first = web_server.save_conversations_payload(
            {
                "base_revision": 0,
                "upserts": [{"id": "chat-a", "title": "A", "group": "最近", "messages": []}],
            }
        )
        self.assertTrue(first["ok"])
        self.assertEqual(first["revision"], 1)

        manifest = json.loads(self.history_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["storage"], "per_item")
        self.assertEqual(manifest["order"], ["chat-a"])
        item_a_path = self.conversation_dir / "archive_items" / "chat-a.json"
        original_item_a = item_a_path.read_text(encoding="utf-8")

        stale = web_server.save_conversations_payload(
            {
                "base_revision": 0,
                "upserts": [{"id": "chat-b", "title": "B", "group": "最近", "messages": []}],
            }
        )
        self.assertFalse(stale["ok"])
        self.assertTrue(stale["conflict"])
        self.assertEqual(stale["revision"], 1)
        self.assertEqual([item["id"] for item in stale["items"]], ["chat-a"])

        second = web_server.save_conversations_payload(
            {
                "base_revision": 1,
                "upserts": [{"id": "chat-b", "title": "B", "group": "最近", "messages": []}],
            }
        )
        self.assertTrue(second["ok"])
        self.assertEqual(second["revision"], 2)
        self.assertEqual(item_a_path.read_text(encoding="utf-8"), original_item_a)
        self.assertTrue((self.conversation_dir / "archive_items" / "chat-b.json").is_file())

        deleted = web_server.save_conversations_payload(
            {"base_revision": 2, "deleted_ids": ["chat-a"]}
        )
        self.assertTrue(deleted["ok"])
        self.assertEqual(deleted["revision"], 3)
        self.assertFalse(item_a_path.exists())
        loaded = web_server.load_conversations_payload()
        self.assertEqual(loaded["revision"], 3)
        self.assertEqual([item["id"] for item in loaded["items"]], ["chat-b"])
        self.assertFalse(list(self.conversation_dir.glob(".*.tmp")))
        self.assertFalse(list((self.conversation_dir / "archive_items").glob(".*.tmp")))

    def test_first_incremental_write_migrates_legacy_single_file_archive(self) -> None:
        self.conversation_dir.mkdir(parents=True)
        self.history_path.write_text(
            json.dumps(
                {
                    "items": [
                        {"id": "legacy-chat", "title": "旧对话", "group": "最近", "messages": []}
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        result = web_server.save_conversations_payload(
            {
                "base_revision": 0,
                "upserts": [
                    {"id": "legacy-chat", "title": "已迁移", "group": "最近", "messages": []}
                ],
            }
        )

        self.assertTrue(result["ok"])
        manifest = json.loads(self.history_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["storage"], "per_item")
        migrated_item = json.loads(
            (self.conversation_dir / "archive_items" / "legacy-chat.json").read_text(encoding="utf-8")
        )
        self.assertEqual(migrated_item["title"], "已迁移")

    def test_delete_cascades_runtime_history_index_and_cross_chat_memory(self) -> None:
        session_store = SessionStore(
            self.workspace,
            session_dir=self.conversation_dir / "sessions",
        )
        session_store.save(
            ConversationSession(
                id="chat-a",
                messages=[
                    {"role": "user", "content": "删除后不应继续被召回的内容"},
                    {"role": "assistant", "content": "已记录"},
                ],
            )
        )
        memory_store = CrossChatMemoryStore(session_store)
        memory_store.upsert_many(
            [{"kind": "preference", "content": "删除对话产生的记忆也应消失。"}],
            conversation_id="chat-a",
            conversation_title="待删除对话",
            state="explicit",
        )

        history_path = self.conversation_dir / "history_search.sqlite3"
        with sqlite3.connect(history_path) as connection:
            ensure_schema(connection)
            connection.execute(
                "INSERT INTO history_conversation_meta(conversation_id, title, project_id, updated_at) VALUES (?, ?, ?, ?)",
                ("chat-a", "待删除对话", "", 1),
            )
            connection.execute(
                "INSERT INTO history_index_meta(conversation_id, signature, message_count, updated_at) VALUES (?, ?, ?, ?)",
                ("chat-a", "signature", 2, 1),
            )
            connection.execute(
                "INSERT INTO history_chunk_parent(conversation_id, message_index, chunk_index, parent_id, parent_content, source_kind) VALUES (?, ?, ?, ?, ?, ?)",
                ("chat-a", 0, 0, "parent", "删除后不应继续被召回的内容", "message"),
            )
            connection.execute(
                "INSERT INTO chat_history_fts(conversation_id, message_index, chunk_index, role, content, search_text) VALUES (?, ?, ?, ?, ?, ?)",
                ("chat-a", 0, 0, "user", "删除后不应继续被召回的内容", "删除后不应继续被召回的内容"),
            )
            connection.commit()

        turn_store = Mock()
        turn_store.discard_pending_for_conversation.return_value = 0
        with patch.object(web_server, "get_session_store", return_value=session_store), patch.object(
            web_server, "get_turn_store", return_value=turn_store
        ):
            web_server.save_conversations_payload(
                {
                    "base_revision": 0,
                    "upserts": [
                        {
                            "id": "chat-a",
                            "title": "待删除对话",
                            "group": "最近",
                            "messages": [],
                        }
                    ],
                }
            )
            result = web_server.save_conversations_payload(
                {"base_revision": 1, "deleted_ids": ["chat-a"]}
            )

        self.assertTrue(result["ok"])
        self.assertFalse((self.conversation_dir / "sessions" / "chat-a.json").exists())
        self.assertEqual(memory_store.list(), [])
        with sqlite3.connect(history_path) as connection:
            for table in (
                "chat_history_fts",
                "history_chunk_parent",
                "history_index_meta",
                "history_conversation_meta",
            ):
                self.assertEqual(
                    connection.execute(
                        f"SELECT COUNT(*) FROM {table} WHERE conversation_id = ?", ("chat-a",)
                    ).fetchone()[0],
                    0,
                )

    def test_cascade_delete_never_removes_friday_main(self) -> None:
        session_store = SessionStore(
            self.workspace,
            session_dir=self.conversation_dir / "sessions",
        )
        session_store.save(ConversationSession(id="friday-main", messages=[{"role": "user", "content": "保留"}]))
        with patch.object(web_server, "get_session_store", return_value=session_store):
            stats = web_server.cascade_delete_conversations(["friday-main"])

        self.assertEqual(stats["sessions"], 0)
        self.assertTrue((self.conversation_dir / "sessions" / "friday-main.json").exists())


if __name__ == "__main__":
    unittest.main()
