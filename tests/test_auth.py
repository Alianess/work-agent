from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from datetime import datetime
from unittest.mock import patch

from work_agent_core.auth import AuthStore
from work_agent_core import web_server
from work_agent_core.session_store import ConversationSession


class AuthStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.store = AuthStore(Path(self.temporary_directory.name) / "auth.sqlite3")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_register_authenticate_and_session(self) -> None:
        user = self.store.register("worker_one", "password123")

        self.assertEqual(user.role, "member")
        self.assertEqual(self.store.authenticate("WORKER_ONE", "password123"), user)
        self.assertIsNone(self.store.authenticate("worker_one", "wrong-password"))

        token = self.store.create_session(user.id)
        self.assertEqual(self.store.user_for_session(token), user)
        self.store.revoke_session(token)
        self.assertIsNone(self.store.user_for_session(token))

    def test_admin_bootstrap_is_idempotent(self) -> None:
        first = self.store.ensure_admin("admin", "admin123")
        second = self.store.ensure_admin("admin", "different-password")

        self.assertEqual(first, second)
        self.assertEqual(first.role, "admin")
        self.assertEqual(self.store.authenticate("admin", "admin123"), first)
        self.assertIsNone(self.store.authenticate("admin", "different-password"))

    def test_first_admin_is_none_until_an_admin_exists(self) -> None:
        self.assertIsNone(self.store.first_admin())
        admin = self.store.ensure_admin("admin", "admin123")
        self.assertEqual(self.store.first_admin(), admin)

    def test_registration_validation_and_password_change(self) -> None:
        user = self.store.register("member.two", "old-pass-123")

        with self.assertRaisesRegex(ValueError, "用户名已存在"):
            self.store.register("MEMBER.TWO", "another-pass")
        with self.assertRaisesRegex(ValueError, "至少需要 8 位"):
            self.store.register("shortpass", "123")

        self.store.change_password(user.id, "old-pass-123", "new-pass-456")
        self.assertIsNone(self.store.authenticate("member.two", "old-pass-123"))
        self.assertEqual(self.store.authenticate("member.two", "new-pass-456"), user)

    def test_registration_requires_configured_invite_code(self) -> None:
        with patch.dict(web_server.os.environ, {"WORK_AGENT_INVITE_CODE": "team-invite-2026"}):
            web_server.validate_registration_invite({"invite_code": "team-invite-2026"})
            with self.assertRaisesRegex(ValueError, "邀请码无效"):
                web_server.validate_registration_invite({"invite_code": "wrong-code"})

        with patch.dict(web_server.os.environ, {"WORK_AGENT_INVITE_CODE": ""}):
            with self.assertRaisesRegex(ValueError, "注册已关闭"):
                web_server.validate_registration_invite({"invite_code": "anything"})

    def test_account_conversations_and_settings_are_separate(self) -> None:
        original_root = web_server.WORKSPACE_ROOT
        original_store = web_server.AUTH_STORE
        original_user = getattr(web_server.REQUEST_AUTH, "user", None)
        original_session_stores = web_server.USER_SESSION_STORES
        try:
            web_server.WORKSPACE_ROOT = Path(self.temporary_directory.name)
            web_server.AUTH_STORE = self.store
            web_server.USER_SESSION_STORES = {}
            first = self.store.register("first_user", "password-111")
            second = self.store.register("second_user", "password-222")

            web_server.REQUEST_AUTH.user = first
            web_server.save_conversations_payload({"items": [{"id": "first-chat", "messages": []}]})
            web_server.save_agent_settings_payload(
                {
                    "assistant_name": "周五",
                    "work_background": "first background",
                    "company_document_format": "标题：二号小标宋",
                }
            )
            from work_agent_core.cross_chat_memory import CrossChatMemoryStore
            CrossChatMemoryStore(web_server.get_session_store()).upsert_many(
                [{"kind": "identity", "content": "仅属于第一个账户的核心记忆。"}],
                conversation_id="first-memory-chat", conversation_title="first",
                source_excerpt="证据", state="explicit",
            )
            self.assertEqual(web_server.cross_chat_memories_payload()["count"], 1)

            web_server.REQUEST_AUTH.user = second
            self.assertEqual(web_server.load_conversations_payload(), {"items": [], "revision": 0})
            self.assertEqual(web_server.cross_chat_memories_payload()["count"], 0)
            self.assertNotEqual(
                web_server.load_agent_settings()["work_background"],
                "first background",
            )
            self.assertNotEqual(
                web_server.load_agent_settings()["company_document_format"],
                "标题：二号小标宋",
            )
            web_server.save_conversations_payload({"items": [{"id": "second-chat", "messages": []}]})

            web_server.REQUEST_AUTH.user = first
            first_items = web_server.load_conversations_payload()["items"]
            self.assertEqual([item["id"] for item in first_items], ["first-chat"])
            self.assertEqual(web_server.load_agent_settings()["work_background"], "first background")
            self.assertEqual(web_server.load_agent_settings()["assistant_name"], "周五")
            self.assertEqual(
                web_server.load_agent_settings()["company_document_format"],
                "标题：二号小标宋",
            )
            context = web_server.agent_system_context(mode="friday")
            self.assertIn("项目经理助理“周五”", context)
            self.assertNotIn("系统当前本地日期时间（本轮提示词生成时读取，含时区偏移）", context)
            turn_time_context = web_server.agent_turn_time_context()
            self.assertIn("系统当前本地日期时间（本轮提示词生成时读取，含时区偏移）", turn_time_context)
            self.assertIn(datetime.now().astimezone().date().isoformat(), turn_time_context)
            self.assertIn("公司标准文件格式（纯文字设置）", context)
            self.assertIn("official-document", context)
            task_context = web_server.agent_system_context(mode="task")
            self.assertNotIn("周五", task_context)
            self.assertNotIn("微信", task_context)
            self.assertIn("普通任务聊天", task_context)
        finally:
            web_server.WORKSPACE_ROOT = original_root
            web_server.AUTH_STORE = original_store
            web_server.USER_SESSION_STORES = original_session_stores
            if original_user is None:
                try:
                    del web_server.REQUEST_AUTH.user
                except AttributeError:
                    pass
            else:
                web_server.REQUEST_AUTH.user = original_user

    def test_core_memory_refresh_is_deliberately_infrequent(self) -> None:
        ordinary = [
            {"role": "user", "content": f"继续处理第 {index} 个普通问题"}
            for index in range(12)
        ]
        self.assertFalse(
            web_server.should_refresh_core_memory(
                user_messages=ordinary[:11],
                last_count=0,
            )
        )
        self.assertTrue(
            web_server.should_refresh_core_memory(
                user_messages=ordinary,
                last_count=0,
            )
        )
        self.assertFalse(
            web_server.should_refresh_core_memory(
                user_messages=ordinary + [{"role": "user", "content": "补充一个普通问题"}],
                last_count=12,
            )
        )
        self.assertTrue(
            web_server.should_refresh_core_memory(
                user_messages=[{"role": "user", "content": "请记住我一直偏好先给结论。"}],
                last_count=0,
            )
        )


if __name__ == "__main__":
    unittest.main()
