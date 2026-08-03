from __future__ import annotations

from pathlib import Path
import tempfile
import time
import unittest

from work_agent_core.notifications import NotificationStore


class NotificationStoreTests(unittest.TestCase):
    def test_immediate_reminder_is_visible_and_can_be_marked_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = NotificationStore(Path(directory) / "notifications.json")
            item = store.add(title="提交材料", body="今天下班前确认上会材料。")

            payload = store.payload()
            self.assertEqual(payload["unread_count"], 1)
            self.assertEqual(payload["items"][0]["id"], item["id"])

            store.mark_read(item["id"])
            self.assertEqual(store.payload()["unread_count"], 0)

    def test_scheduled_delivery_is_claimed_only_when_due(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = NotificationStore(Path(directory) / "notifications.json")
            due_at = int(time.time()) + 100
            store.add(
                kind="reminder",
                title="节点提醒",
                body="准备明早会议。",
                deliver_at=due_at,
            )
            conversation = store.add(
                kind="conversation",
                title="需要讨论",
                body="风险节点需要你确认。",
                deliver_at=due_at,
            )

            self.assertEqual(store.payload()["items"], [])
            self.assertEqual(store.claim_due(now=due_at - 1), [])
            due = store.claim_due(now=due_at)
            self.assertEqual({item["kind"] for item in due}, {"reminder", "conversation"})
            self.assertEqual(store.payload()["unread_count"], 1)
            self.assertIn(conversation["id"], {item["id"] for item in due})

    def test_delete_removes_only_the_target_and_updates_unread_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = NotificationStore(Path(directory) / "notifications.json")
            first = store.add(title="第一条", body="待删除。")
            second = store.add(title="第二条", body="应保留。")

            self.assertTrue(store.delete(first["id"]))
            payload = store.payload()
            self.assertEqual(payload["unread_count"], 1)
            self.assertEqual([item["id"] for item in payload["items"]], [second["id"]])
            self.assertFalse(store.delete(first["id"]))

    def test_conversation_is_marked_delivered_only_after_archive_append_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = NotificationStore(Path(directory) / "notifications.json")
            item = store.add(
                kind="conversation",
                title="需要讨论",
                body="请确认风险节点。",
                deliver_at=int(time.time()),
            )

            due = store.due_conversations()
            self.assertEqual([entry["id"] for entry in due], [item["id"]])
            self.assertEqual(store.due_conversations()[0]["delivered_at"], 0)
            self.assertTrue(store.mark_delivered(item["id"]))
            self.assertEqual(store.due_conversations(), [])


if __name__ == "__main__":
    unittest.main()
