from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from work_agent_core.config import ModelProfile
from work_agent_core.memory import ContextCompactionError, prepare_session_memory
from work_agent_core.session_store import ConversationSession


class RecordingClient:
    def __init__(self, content: str = "## 当前目标与用户意图\n- 继续当前任务") -> None:
        self.content = content
        self.calls: list[dict[str, object]] = []

    def chat(self, messages, *, profile, max_tokens):
        self.calls.append(
            {"messages": messages, "profile": profile, "max_tokens": max_tokens}
        )
        return SimpleNamespace(content=self.content)


class ChatMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = ModelProfile(
            name="test",
            provider="openai-compatible",
            base_url="http://example.invalid",
            model="test-model",
            api_key_env="TEST_API_KEY",
        )

    def test_compaction_merges_all_completed_messages_and_keeps_two_visible_turns(self) -> None:
        messages = [
            {"role": "user", "content": "第一轮问题"},
            {"role": "assistant", "content": "第一轮最终回答"},
            {"role": "user", "content": "第二轮问题"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path":"/tmp/evidence.md"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_2",
                "name": "read_file",
                "content": "关键工具证据：预算 280 万元",
            },
            {"role": "assistant", "content": "第二轮最终回答"},
            {"role": "user", "content": "第三轮问题"},
            {"role": "assistant", "content": "第三轮最终回答"},
            {"role": "user", "content": "当前正在处理的问题"},
        ]
        session = ConversationSession(
            id="chat-memory-test",
            messages=deepcopy(messages),
            summary="旧工作摘要：已经完成准备工作。",
            summary_message_count=0,
        )
        client = RecordingClient()

        with patch(
            "work_agent_core.memory.CHAT_SUMMARY_TRIGGER_TOKENS", 1
        ):
            prepared = prepare_session_memory(client, self.profile, session)

        self.assertTrue(prepared.compacted)
        self.assertEqual(session.messages[0:3], messages[0:3])
        self.assertEqual(session.messages[5:], messages[5:])
        self.assertNotIn(
            "关键工具证据：预算 280 万元",
            session.messages[4]["content"],
        )
        self.assertIn("read_file 已完成", session.messages[4]["content"])
        self.assertTrue(session.recall_episodes)
        archived = session.recall_episodes[0]
        self.assertIn("第二轮问题", archived["user_texts"])
        self.assertTrue(session.compaction_events)
        self.assertEqual(prepared.summary_message_count, len(messages) - 1)
        self.assertEqual(session.summary_message_count, len(messages) - 1)
        self.assertEqual(
            prepared.messages,
            [
                {"role": "user", "content": "第二轮问题"},
                {"role": "assistant", "content": "第二轮最终回答"},
                {"role": "user", "content": "第三轮问题"},
                {"role": "assistant", "content": "第三轮最终回答"},
                {"role": "user", "content": "当前正在处理的问题"},
            ],
        )
        self.assertEqual(client.calls[0]["max_tokens"], 8192)
        summary_input = client.calls[0]["messages"][1]["content"]
        self.assertIn("旧工作摘要：已经完成准备工作。", summary_input)
        self.assertIn("read_file", summary_input)
        self.assertIn("/tmp/evidence.md", summary_input)
        self.assertIn("关键工具证据：预算 280 万元", summary_input)
        self.assertIn("第一轮问题", summary_input)
        self.assertIn("第三轮最终回答", summary_input)
        self.assertIn("不是长期记忆", prepared.system_context)
        self.assertIn("当前任务", prepared.system_context)

        followup_client = RecordingClient()
        with patch(
            "work_agent_core.memory.CHAT_SUMMARY_TRIGGER_TOKENS", 1_000_000
        ):
            followup = prepare_session_memory(followup_client, self.profile, session)
        self.assertFalse(followup.compacted)
        self.assertEqual(followup.messages, prepared.messages)
        self.assertEqual(followup_client.calls, [])

    def test_old_incomplete_turn_is_compacted_but_current_turn_stays_raw(self) -> None:
        messages = [
            {"role": "user", "content": "已完成问题"},
            {"role": "assistant", "content": "已完成回答"},
            {"role": "user", "content": "等待审批的工作"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_pending",
                        "type": "function",
                        "function": {"name": "write_file", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_pending",
                "name": "write_file",
                "content": "TOOL_MISSING: 等待审批",
            },
            {"role": "user", "content": "补充要求"},
        ]
        session = ConversationSession(id="pending-test", messages=deepcopy(messages))
        client = RecordingClient()

        with patch(
            "work_agent_core.memory.CHAT_SUMMARY_TRIGGER_TOKENS", 1
        ):
            prepared = prepare_session_memory(client, self.profile, session)

        self.assertEqual(prepared.summary_message_count, len(messages) - 1)
        self.assertEqual(prepared.messages[-1], messages[-1])
        summary_input = client.calls[0]["messages"][1]["content"]
        self.assertIn("已完成问题", summary_input)
        self.assertIn("等待审批的工作", summary_input)
        self.assertIn("TOOL_MISSING: 等待审批", summary_input)
        self.assertNotIn("补充要求", summary_input)

    def test_below_threshold_keeps_raw_runtime_messages(self) -> None:
        messages = [
            {"role": "user", "content": "问题"},
            {"role": "assistant", "content": "回答"},
            {"role": "user", "content": "当前问题"},
        ]
        session = ConversationSession(id="raw-test", messages=deepcopy(messages))
        client = RecordingClient()

        with patch(
            "work_agent_core.memory.CHAT_SUMMARY_TRIGGER_TOKENS", 1_000_000
        ):
            prepared = prepare_session_memory(client, self.profile, session)

        self.assertFalse(prepared.compacted)
        self.assertEqual(prepared.messages, messages)
        self.assertEqual(client.calls, [])

    def test_runtime_reserve_is_counted_before_compaction(self) -> None:
        messages = [
            {"role": "user", "content": "读取三场会议材料"},
            {"role": "assistant", "content": "三场材料已经整理完成"},
            {"role": "user", "content": "继续完成剩余验收"},
        ]
        session = ConversationSession(id="reserve-test", messages=deepcopy(messages))
        client = RecordingClient()

        raw_tokens = sum(
            len(str(message.get("content") or ""))
            for message in messages
        )
        with patch(
            "work_agent_core.memory.CHAT_SUMMARY_TRIGGER_TOKENS",
            raw_tokens + 10,
        ):
            prepared = prepare_session_memory(
                client,
                self.profile,
                session,
                reserved_tokens=1_000,
            )

        self.assertTrue(prepared.compacted)
        self.assertEqual(prepared.summary_message_count, 2)
        self.assertGreaterEqual(prepared.estimated_tokens, 1_000)
        self.assertEqual(len(client.calls), 1)

    def test_force_compaction_runs_below_the_automatic_threshold(self) -> None:
        messages = [
            {"role": "user", "content": "先读取项目材料"},
            {"role": "assistant", "content": "已经读取并确认当前版本"},
        ]
        session = ConversationSession(id="forced-compact", messages=deepcopy(messages))
        client = RecordingClient()

        with patch(
            "work_agent_core.memory.CHAT_SUMMARY_TRIGGER_TOKENS", 1_000_000
        ):
            prepared = prepare_session_memory(
                client,
                self.profile,
                session,
                force=True,
            )

        self.assertTrue(prepared.compacted)
        self.assertEqual(prepared.summary_message_count, 2)
        self.assertEqual(len(client.calls), 1)

    def test_compaction_failure_preserves_original_messages_without_fallback(self) -> None:
        class FailingClient:
            def chat(self, *_args, **_kwargs):
                raise RuntimeError("LLM stream idle timed out")

        messages = [
            {"role": "user", "content": "先完成材料核验"},
            {"role": "assistant", "content": "已核验两份材料，待生成汇总。"},
        ]
        session = ConversationSession(id="failed-compact", messages=deepcopy(messages))

        with patch("work_agent_core.memory.CHAT_SUMMARY_TRIGGER_TOKENS", 1):
            with self.assertRaises(ContextCompactionError) as captured:
                prepare_session_memory(FailingClient(), self.profile, session, force=True)

        self.assertIn("不会自动切换或重试模型", str(captured.exception))
        self.assertEqual(session.messages, messages)
        self.assertEqual(session.summary, "")
        self.assertEqual(session.summary_message_count, 0)
        self.assertEqual(session.compaction_events, [])


if __name__ == "__main__":
    unittest.main()
