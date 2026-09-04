from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from work_agent_core import web_server
from work_agent_core.cross_chat_memory import CrossChatMemoryStore
from work_agent_core.session_store import ConversationSession, SessionStore
from work_agent_core.session_log_store import SessionLogStore
from work_agent_core.session_migration import build_seed_log


class ConversationArchiveTests(unittest.TestCase):
    def test_legacy_compaction_text_does_not_claim_smaller_tokens_exceeded_larger_limit(self) -> None:
        repaired = web_server.sanitize_activity_event(
            {
                "event": "activity",
                "phase": "thinking",
                "title": "上下文已整理",
                "activity_type": "runtime_summary",
                "detail": (
                    "估算上下文 134518 tokens，已超过 230400 tokens；"
                    "分点摘要已生成，并作为后续初始上下文。"
                ),
            }
        )

        self.assertIn("134,518 tokens，低于 230,400 token 安全线", repaired["detail"])
        self.assertIn("会话体积或历史工具结果", repaired["detail"])
        self.assertNotIn("已超过", repaired["detail"])
        self.assertEqual(repaired["context_pre_compaction_tokens"], 134_518)
        self.assertEqual(repaired["context_trigger_tokens"], 230_400)

    def test_activity_projection_collapses_one_execution_lifecycle(self) -> None:
        execution_id = "exe-one"
        events = [
            {
                "event": "activity",
                "id": f"execution-{execution_id}",
                "phase": "error",
                "title": "安全执行环境",
                "activity_type": "command",
                "command_status": "error",
                "execution_id": execution_id,
                "execution_status": "failed",
                "detail": "命令退出码为 1。",
                "content": "完整错误输出",
            },
            {
                "event": "activity",
                "id": f"execution-{execution_id}",
                "phase": "action",
                "title": "安全执行环境",
                "activity_type": "runtime_summary",
                "command_status": "running",
                "execution_id": execution_id,
                "execution_status": "running",
                "detail": "隔离执行环境已就绪。",
                "content": "",
            },
            {
                "event": "activity",
                "id": f"execution-{execution_id}",
                "phase": "error",
                "title": "安全执行环境",
                "activity_type": "runtime_summary",
                "command_status": "error",
                "execution_id": execution_id,
                "execution_status": "failed",
                "detail": "命令退出码为 1。",
                "content": "",
            },
        ]

        projected = web_server.project_activity_events(events)

        self.assertEqual(len(projected), 1)
        self.assertEqual(projected[0]["activity_type"], "command")
        self.assertEqual(projected[0]["phase"], "error")
        self.assertEqual(projected[0]["command_status"], "error")
        self.assertEqual(projected[0]["execution_status"], "failed")
        self.assertEqual(projected[0]["content"], "完整错误输出")

    def test_activity_projection_does_not_erase_reasoning_with_empty_recovery_snapshot(self) -> None:
        event_id = "model-plan-1"
        events = [
            {
                "event": "activity_delta",
                "id": event_id,
                "phase": "thinking",
                "title": "第 1 轮 · 模型思考",
                "content": "模型正在流式返回。",
                "reasoning_content": "已经收到的思考内容",
                "append_mode": "replace",
            },
            {
                "event": "activity_delta",
                "id": event_id,
                "phase": "thinking",
                "title": "第 1 轮 · 模型思考",
                "content": "正在启动兼容恢复。",
                "reasoning_content": "",
                "append_mode": "replace",
                "stream_status": "recovery_started",
            },
        ]

        projected = web_server.project_activity_events(events)

        self.assertEqual(len(projected), 1)
        self.assertEqual(projected[0]["content"], "正在启动兼容恢复。")
        self.assertEqual(projected[0]["reasoning_content"], "已经收到的思考内容")

    def test_load_recovers_completed_log_chat_but_hides_internal_log_only_record(self) -> None:
        log_store = SessionLogStore(self.conversation_dir / "session_log.sqlite3")
        recovered = build_seed_log(
            "local-recovered",
            [
                {"role": "user", "content": "帮我看产业方案"},
                {"role": "assistant", "content": "已经完成研判"},
            ],
        )
        internal = build_seed_log(
            "conversation-1",
            [{"role": "assistant", "content": "done"}],
        )
        for log in (recovered, internal):
            log_store.put_header(log.header)
            log_store.append(log.header.session_id, log.events)

        with patch.object(web_server, "get_session_log_store", return_value=log_store):
            loaded = web_server.load_conversations_payload()

        items = {item["id"]: item for item in loaded["items"]}
        self.assertIn("local-recovered", items)
        self.assertEqual(items["local-recovered"]["title"], "帮我看产业方案")
        self.assertNotIn("conversation-1", items)

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

    def test_sidebar_index_excludes_conversation_bodies(self) -> None:
        web_server.save_conversations_payload(
            {
                "base_revision": 0,
                "upserts": [
                    {
                        "id": "chat-index",
                        "title": "轻量列表",
                        "group": "最近",
                        "messages": [
                            {"role": "user", "content": "一段很长的正文" * 100},
                            {"role": "assistant", "content": "已经处理"},
                        ],
                        "activities": {
                            "1": {
                                "events": [{"phase": "thinking", "content": "内部过程" * 100}],
                                "elapsedMs": 12,
                                "completed": True,
                            }
                        },
                    }
                ],
            }
        )

        payload = web_server.load_conversation_index_payload()

        self.assertEqual(payload["revision"], 1)
        self.assertEqual(len(payload["items"]), 1)
        item = payload["items"][0]
        self.assertEqual(item["id"], "chat-index")
        self.assertEqual(item["messageCount"], 2)
        self.assertNotIn("messages", item)
        self.assertNotIn("activities", item)
        self.assertNotIn("contextSummary", item)

    def test_conversation_detail_loads_one_complete_body(self) -> None:
        web_server.save_conversations_payload(
            {
                "base_revision": 0,
                "upserts": [
                    {
                        "id": "chat-detail",
                        "title": "按需正文",
                        "group": "最近",
                        "messages": [
                            {"role": "user", "content": "只在打开时加载"},
                            {"role": "assistant", "content": "正文已返回"},
                        ],
                    }
                ],
            }
        )

        payload = web_server.load_conversation_detail_payload("chat-detail")

        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertEqual(payload["revision"], 1)
        self.assertEqual(payload["item"]["title"], "按需正文")
        self.assertEqual(
            [message["content"] for message in payload["item"]["messages"]],
            ["只在打开时加载", "正文已返回"],
        )

    def test_conversation_detail_honors_tombstones(self) -> None:
        web_server.save_conversations_payload(
            {
                "base_revision": 0,
                "upserts": [
                    {"id": "chat-deleted", "title": "已删除", "group": "最近", "messages": []}
                ],
            }
        )
        web_server.delete_conversations_payload({"conversation_ids": ["chat-deleted"]})

        self.assertIsNone(web_server.load_conversation_detail_payload("chat-deleted"))

    def test_load_with_current_revision_returns_unchanged_without_items(self) -> None:
        """Friday 视图 30 秒轮询一次全量存档（2MB+）；revision 未变时只回空体。"""
        web_server.save_conversations_payload(
            {
                "base_revision": 0,
                "upserts": [{"id": "chat-a", "title": "A", "group": "最近", "messages": []}],
            }
        )
        revision = web_server.load_conversations_payload()["revision"]

        unchanged = web_server.load_conversations_payload(since_revision=revision)

        self.assertTrue(unchanged["unchanged"])
        self.assertEqual(unchanged["items"], [])
        self.assertEqual(unchanged["revision"], revision)

        fresh = web_server.load_conversations_payload(since_revision=revision - 1)
        self.assertNotIn("unchanged", fresh)
        self.assertEqual([item["id"] for item in fresh["items"]], ["chat-a"])

    def test_load_hides_legacy_stringified_read_file_image_messages(self) -> None:
        legacy = (
            "[{'type': 'text', 'text': '以下是刚才用 read_file 载入的图片。'}, "
            "{'type': 'image_url', 'image_url': {'url': "
            "'data:image/jpeg;base64,abc'}}]"
        )
        web_server.save_conversations_payload(
            {
                "base_revision": 0,
                "upserts": [
                    {
                        "id": "chat-image",
                        "title": "图片对话",
                        "group": "最近",
                        "messages": [
                            {"role": "user", "content": "请看图"},
                            {"role": "user", "content": legacy},
                            {"role": "assistant", "content": "看到了。"},
                        ],
                    }
                ],
            }
        )

        loaded = web_server.load_conversations_payload()
        item = next(entry for entry in loaded["items"] if entry["id"] == "chat-image")
        self.assertEqual(
            item["messages"],
            [
                {"role": "user", "content": "请看图"},
                {"role": "assistant", "content": "看到了。"},
            ],
        )

    def test_archive_sanitizer_collapses_adjacent_legacy_duplicate_user_messages(self) -> None:
        sanitized = web_server.sanitize_conversation_archive_item(
            {
                "id": "legacy-duplicate",
                "messages": [
                    {"role": "user", "content": "同一条很长的请求"},
                    {"role": "user", "content": "同一条很长的请求"},
                    {"role": "assistant", "content": "处理中"},
                ],
            }
        )

        self.assertEqual(
            [(message["role"], message["content"]) for message in sanitized["messages"]],
            [("user", "同一条很长的请求"), ("assistant", "处理中")],
        )

    def test_runtime_timeline_replaces_same_length_stale_archive_messages(self) -> None:
        session_store = web_server.get_session_store()
        session_store.save(
            ConversationSession(
                id="same-length-stale",
                messages=[
                    {"role": "user", "content": "最初的短请求"},
                    {"role": "user", "content": "补充后的完整请求"},
                    {"role": "assistant", "content": "已经开始处理"},
                ],
            )
        )

        web_server.save_conversations_payload(
            {
                "base_revision": 0,
                "upserts": [
                    {
                        "id": "same-length-stale",
                        "title": "测试",
                        "group": "最近",
                        "messages": [
                            {"role": "user", "content": "补充后的完整请求"},
                            {"role": "user", "content": "补充后的完整请求"},
                            {"role": "assistant", "content": "已经开始处理"},
                        ],
                    }
                ],
            }
        )

        loaded = web_server.load_conversations_payload()
        item = next(entry for entry in loaded["items"] if entry["id"] == "same-length-stale")
        self.assertEqual(
            [(message["role"], message["content"]) for message in item["messages"]],
            [
                ("user", "最初的短请求"),
                ("user", "补充后的完整请求"),
                ("assistant", "已经开始处理"),
            ],
        )

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

    def test_delete_cascades_recall_index_and_cross_chat_memory(self) -> None:
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
        # 预置统一检索索引里的这份会话，删除后它必须一并消失。
        from work_agent_core.recall.sync import RecallSync
        from work_agent_core.session_log import (
            ASSISTANT_MESSAGE,
            SessionHeader,
            SessionLog,
            TURN_START,
            USER_MESSAGE,
        )

        log = SessionLog(SessionHeader(session_id="chat-a"))
        log.append(TURN_START, {"turn_id": "turn-1"})
        log.append(USER_MESSAGE, {"content": "删除后不应继续被召回的内容"})
        log.append(ASSISTANT_MESSAGE, {"content": "已记录"})
        recall_sync = RecallSync(
            web_server.recall_index_for(self.conversation_dir),
            workspace_root=self.workspace,
        )
        recall_sync.index_conversation("chat-a", log, title="待删除对话")

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
        with sqlite3.connect(self.conversation_dir / "recall" / "recall.sqlite3") as connection:
            remaining = connection.execute(
                "SELECT COUNT(*) FROM recall_sources WHERE source_id = 'chat:chat-a'"
            ).fetchone()[0]
        self.assertEqual(remaining, 0)

    def test_direct_delete_is_synchronous_and_removes_every_store(self) -> None:
        session_store = web_server.get_session_store()
        session_store.save(
            ConversationSession(
                id="chat-direct-delete",
                messages=[{"role": "user", "content": "需要彻底删除的向量内容"}],
            )
        )
        memory_store = CrossChatMemoryStore(session_store)
        memory_store.upsert_many(
            [{"kind": "fact", "content": "这条派生记忆必须物理删除。"}],
            conversation_id="chat-direct-delete",
            conversation_title="待删除",
            state="explicit",
        )
        from work_agent_core.recall.sync import RecallSync
        from work_agent_core.session_log import SessionHeader, SessionLog, TURN_START, USER_MESSAGE

        log = SessionLog(SessionHeader(session_id="chat-direct-delete"))
        log.append(TURN_START, {"turn_id": "turn-delete"})
        log.append(USER_MESSAGE, {"content": "需要彻底删除的向量内容"})
        RecallSync(
            web_server.recall_index_for(self.conversation_dir),
            workspace_root=self.workspace,
        ).index_conversation("chat-direct-delete", log, title="待删除")
        web_server.save_conversations_payload(
            {
                "base_revision": 0,
                "upserts": [
                    {
                        "id": "chat-direct-delete",
                        "title": "待删除",
                        "group": "最近",
                        "messages": [],
                    }
                ],
            }
        )

        result = web_server.delete_conversations_payload(
            {"conversation_ids": ["chat-direct-delete"]}
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["deleted_ids"], ["chat-direct-delete"])
        self.assertFalse(
            (self.conversation_dir / "archive_items" / "chat-direct-delete.json").exists()
        )
        self.assertFalse(
            (self.conversation_dir / "sessions" / "chat-direct-delete.json").exists()
        )
        with sqlite3.connect(memory_store.database_path) as connection:
            memory_rows = connection.execute(
                "SELECT COUNT(*) FROM memory_items WHERE conversation_id=?",
                ("chat-direct-delete",),
            ).fetchone()[0]
        self.assertEqual(memory_rows, 0)
        with sqlite3.connect(
            self.conversation_dir / "recall" / "recall.sqlite3"
        ) as connection:
            recall_rows = connection.execute(
                "SELECT COUNT(*) FROM recall_sources WHERE source_id=?",
                ("chat:chat-direct-delete",),
            ).fetchone()[0]
        self.assertEqual(recall_rows, 0)
        loaded = web_server.load_conversations_payload()
        self.assertEqual(loaded["items"], [])
        self.assertEqual(loaded["deleted_ids"], ["chat-direct-delete"])

        # A tab that was open before deletion may keep submitting its stale
        # local snapshot. The server tombstone must make that upsert a no-op.
        stale_save = web_server.save_conversations_payload(
            {
                "base_revision": result["revision"],
                "upserts": [
                    {
                        "id": "chat-direct-delete",
                        "title": "待删除",
                        "group": "最近",
                        "messages": [{"role": "user", "content": "旧页面重新保存"}],
                    }
                ],
            }
        )
        self.assertTrue(stale_save["ok"])
        self.assertEqual(web_server.load_conversations_payload()["items"], [])
        self.assertFalse(
            (self.conversation_dir / "archive_items" / "chat-direct-delete.json").exists()
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

    def test_save_rejects_bootstrap_greeting_wipe_of_friday_archive(self) -> None:
        """重启后残留的问候语空壳不得覆盖 Friday 存档（8/20、8/21 两次事故）。"""
        web_server.get_session_store().save(
            ConversationSession(
                id="friday-main",
                messages=[
                    {"role": "assistant", "content": "你好，我是Friday，你的项目经理助理。"},
                    {"role": "user", "content": "桌面有几张截图？"},
                    {"role": "assistant", "content": "", "tool_calls": [
                        {"id": "call-1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
                    ]},
                    {"role": "tool", "tool_call_id": "call-1", "content": "共 19 张"},
                    {"role": "assistant", "content": "19 张。"},
                ],
            )
        )
        saved = web_server.save_conversations_payload(
            {
                "base_revision": 0,
                "upserts": [
                    {
                        "id": "friday-main",
                        "title": "Friday",
                        "group": "助理",
                        "messages": [
                            {"role": "assistant", "content": "你好，我是Friday，你的项目经理助理。"}
                        ],
                    }
                ],
            }
        )
        self.assertTrue(saved["ok"])

        loaded = web_server.load_conversations_payload()
        friday = next(item for item in loaded["items"] if item["id"] == "friday-main")
        self.assertEqual(
            [(message["role"], message["content"]) for message in friday["messages"]],
            [
                ("assistant", "你好，我是Friday，你的项目经理助理。"),
                ("user", "桌面有几张截图？"),
                ("assistant", "19 张。"),
            ],
        )

    def test_load_rebuilds_wiped_friday_archive_from_runtime_session(self) -> None:
        session_store = web_server.get_session_store()
        session_store.save(
            ConversationSession(
                id="friday-main",
                messages=[
                    {"role": "assistant", "content": "你好，我是Friday。"},
                    {"role": "user", "content": "在吗"},
                    {"role": "assistant", "content": "在。"},
                ],
            )
        )
        web_server.save_conversations_payload(
            {
                "base_revision": 0,
                "upserts": [
                    {
                        "id": "friday-main",
                        "title": "Friday",
                        "group": "助理",
                        "messages": [
                            {"role": "assistant", "content": "你好，我是Friday。"},
                            {"role": "user", "content": "在吗"},
                            {"role": "assistant", "content": "在。"},
                        ],
                    }
                ],
            }
        )
        wiped_path = self.conversation_dir / "archive_items" / "friday-main.json"
        wiped_path.write_text(
            json.dumps(
                {
                    "id": "friday-main",
                    "title": "Friday",
                    "group": "助理",
                    "messages": [{"role": "assistant", "content": "你好，我是Friday。"}],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        loaded = web_server.load_conversations_payload()
        friday = next(item for item in loaded["items"] if item["id"] == "friday-main")
        self.assertEqual(len(friday["messages"]), 3)
        self.assertEqual(friday["messages"][-1]["content"], "在。")

    def test_load_rebuilds_short_normal_archive_from_runtime_session(self) -> None:
        session_store = web_server.get_session_store()
        session_store.save(
            ConversationSession(
                id="normal-chat",
                messages=[
                    {"role": "user", "content": "完成两份工作"},
                    {"role": "assistant", "content": "正在编制。"},
                    {"role": "user", "content": "继续"},
                    {"role": "assistant", "content": "两份工作已经完成。"},
                ],
            )
        )
        web_server.save_conversations_payload(
            {
                "base_revision": 0,
                "upserts": [
                    {
                        "id": "normal-chat",
                        "title": "两份工作",
                        "group": "最近",
                        "messages": [{"role": "user", "content": "完成两份工作"}],
                    }
                ],
            }
        )

        loaded = web_server.load_conversations_payload()
        normal = next(item for item in loaded["items"] if item["id"] == "normal-chat")
        self.assertEqual(len(normal["messages"]), 4)
        self.assertEqual(normal["messages"][-1]["content"], "两份工作已经完成。")


if __name__ == "__main__":
    unittest.main()
