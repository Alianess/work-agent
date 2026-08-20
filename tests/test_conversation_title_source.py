"""会话标题住在 metadata 里，不是 ConversationSession 的字段。

写成 session.title 不会在导入时报错，只会在真正跑到那一行时抛 AttributeError。
它出现在 persist_runtime_history 里——每条退出路径都会走——于是模型算完了，
整轮却在收尾时炸掉，工具结果一起丢。
"""

from __future__ import annotations

import unittest

from work_agent_core.session_store import ConversationSession


class ConversationTitleSourceTests(unittest.TestCase):
    def test_the_title_is_not_an_attribute(self) -> None:
        session = ConversationSession(id="conversation-1")

        self.assertFalse(hasattr(session, "title"))

    def test_the_title_is_read_from_metadata(self) -> None:
        session = ConversationSession(id="conversation-1", metadata={"title": "这是什么页面"})

        self.assertEqual(session.metadata.get("title"), "这是什么页面")

    def test_the_title_survives_a_save_and_load_round_trip(self) -> None:
        original = ConversationSession(id="conversation-1", metadata={"title": "中试基地会议"})

        restored = ConversationSession.from_payload(
            original.to_payload(), fallback_id="conversation-1"
        )

        self.assertEqual(restored.metadata.get("title"), "中试基地会议")
        self.assertFalse(hasattr(restored, "title"))


if __name__ == "__main__":
    unittest.main()
