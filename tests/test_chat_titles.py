from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from work_agent_core import web_server


class ChatTitlePayloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = SimpleNamespace(default_profile="default", get=lambda _: SimpleNamespace(name="default"))
        self.messages = [
            {"role": "assistant", "content": "已进入项目。"},
            {"role": "user", "content": "你好，项目推进的怎么样了"},
            {"role": "assistant", "content": "项目目前处于前期洽谈与方案起草阶段。"},
        ]

    def test_invalid_model_title_uses_readable_fallback(self) -> None:
        with (
            patch.object(web_server, "load_registry", return_value=self.registry),
            patch.object(web_server, "request_conversation_title", return_value="待命名对话"),
        ):
            payload = web_server.generate_chat_title_payload({"messages": self.messages})

        self.assertEqual(payload["title"], "你好，项目推进的怎么样了")

    def test_title_request_failure_does_not_leave_chat_untitled(self) -> None:
        with (
            patch.object(web_server, "load_registry", return_value=self.registry),
            patch.object(web_server, "request_conversation_title", side_effect=RuntimeError("network")),
        ):
            payload = web_server.generate_chat_title_payload({"messages": self.messages})

        self.assertNotEqual(payload["title"], web_server.PENDING_CONVERSATION_TITLE)
