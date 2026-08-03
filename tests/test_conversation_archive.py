from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from work_agent_core import web_server


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
        self.workspace_patch.start()
        self.dir_patch.start()
        self.path_patch.start()

    def tearDown(self) -> None:
        self.path_patch.stop()
        self.dir_patch.stop()
        self.workspace_patch.stop()
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


if __name__ == "__main__":
    unittest.main()
