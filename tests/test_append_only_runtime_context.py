from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from work_agent_core.config import ModelProfile
from work_agent_core.memory import completed_message_prefix_end, extract_recent_visible_turns
from work_agent_core.react import ReActAgent
from work_agent_core.session_runtime import ConversationRuntime
from work_agent_core.session_store import ConversationSession, SessionStore
from work_agent_core.tool_bus import ToolBus


PROFILE = ModelProfile(
    name="append-only-context-test",
    provider="openai-compatible",
    base_url="https://example.invalid/v1",
    model="test-model",
    api_key_env="UNUSED",
)


class AppendOnlyRuntimeContextTests(unittest.TestCase):
    def test_second_turn_keeps_the_first_request_as_an_exact_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(directory)
            session = ConversationSession(id="cache-prefix")
            store.append_user_message(
                session,
                "User1",
                runtime_context="当前轮 runtime context：\n系统当前时间：16:14:38",
            )
            store.save(session)
            first_history = store.load(session.id).messages

            agent = ReActAgent(
                client=object(),  # type: ignore[arg-type]
                profile=PROFILE,
                tools=ToolBus(),
                system_prompt="Fixed System",
                late_task_plan_context=False,
            )
            first_request = agent._request_messages(
                ConversationRuntime.from_messages(first_history)
            )

            session = store.load(session.id)
            session.messages.append({"role": "assistant", "content": "Assistant1"})
            store.append_user_message(
                session,
                "User2",
                runtime_context="当前轮 runtime context：\n系统当前时间：16:20:15",
            )
            store.save(session)
            second_request = agent._request_messages(
                ConversationRuntime.from_messages(store.load(session.id).messages)
            )

        self.assertEqual(second_request[: len(first_request)], first_request)
        self.assertEqual(
            [item["role"] for item in second_request],
            ["system", "system", "user", "assistant", "system", "user"],
        )

    def test_compaction_keeps_runtime_event_with_its_user_turn(self) -> None:
        first_event = {
            "role": "system",
            "content": "当前轮 runtime context：\n系统当前时间：16:14:38",
        }
        second_event = {
            "role": "system",
            "content": "当前轮 runtime context：\n系统当前时间：16:20:15",
        }
        messages = [
            first_event,
            {"role": "user", "content": "User1"},
            {"role": "assistant", "content": "Assistant1"},
            second_event,
            {"role": "user", "content": "User2"},
        ]

        self.assertEqual(completed_message_prefix_end(messages), 3)
        self.assertEqual(
            extract_recent_visible_turns(messages[:3], turn_limit=1),
            [
                first_event,
                {"role": "user", "content": "User1"},
                {"role": "assistant", "content": "Assistant1"},
            ],
        )

    def test_rewind_removes_the_runtime_event_belonging_to_discarded_user(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(directory)
            session = ConversationSession(
                id="rewind-runtime-context",
                messages=[
                    {"role": "system", "content": "当前轮 runtime context：\ntime #1"},
                    {"role": "user", "content": "User1"},
                    {"role": "assistant", "content": "Assistant1"},
                    {"role": "system", "content": "当前轮 runtime context：\ntime #2"},
                    {"role": "user", "content": "User2"},
                ],
            )

            store.rewind_before_user_message(session, 1)

        self.assertEqual(
            session.messages,
            [
                {"role": "system", "content": "当前轮 runtime context：\ntime #1"},
                {"role": "user", "content": "User1"},
                {"role": "assistant", "content": "Assistant1"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
