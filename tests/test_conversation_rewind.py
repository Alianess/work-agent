from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from work_agent_core.session_store import ConversationSession, SessionStore
from work_agent_core.turn_store import TurnStore
from work_agent_core.web_server import rewind_session_or_rebuild_from_display, sanitize_rewind_user_message_ordinal


class ConversationRewindTests(unittest.TestCase):
    def test_rewind_keeps_exact_runtime_prefix_before_selected_user_turn(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(directory)
            session = ConversationSession(
                id="conversation-1",
                summary="summary that includes discarded turns",
                summary_message_count=5,
                messages=[
                    {"role": "user", "content": "first"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "read_text_file", "arguments": "{}"},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call-1", "content": "kept"},
                    {"role": "assistant", "content": "first answer"},
                    {"role": "user", "content": "second"},
                    {"role": "assistant", "content": "second answer"},
                    {"role": "user", "content": "third"},
                ],
            )

            store.rewind_before_user_message(session, 1)

            self.assertEqual([message["role"] for message in session.messages], ["user", "assistant", "tool", "assistant"])
            self.assertEqual(session.messages[2]["content"], "kept")
            self.assertEqual(session.summary, "")
            self.assertEqual(session.summary_message_count, 0)

    def test_rewind_rejects_unknown_user_turn(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(directory)
            session = ConversationSession(id="conversation-1", messages=[{"role": "user", "content": "only"}])
            with self.assertRaises(ValueError):
                store.rewind_before_user_message(session, 2)

    def test_rewind_rebuilds_from_display_when_backend_session_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(directory)
            session = ConversationSession(
                id="conversation-1",
                summary="stale summary",
                summary_message_count=2,
                messages=[{"role": "user", "content": "first"}],
            )
            display_messages = [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "first answer"},
                {"role": "user", "content": "replacement second"},
            ]

            rebuilt = rewind_session_or_rebuild_from_display(store, session, display_messages, 1)

            self.assertTrue(rebuilt)
            self.assertEqual(
                session.messages,
                [
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "first answer"},
                ],
            )
            self.assertEqual(session.summary, "")
            self.assertEqual(session.summary_message_count, 0)

    def test_discard_pending_approval_prevents_old_branch_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = TurnStore(directory)
            turn = store.create(conversation_id="conversation-1")
            store.set_pending_approval(turn.id, {"conversation_id": "conversation-1", "commands": ["pwd"]})

            discarded = store.discard_pending_for_conversation("conversation-1")
            retired = store.load(turn.id)

            self.assertEqual(discarded, 1)
            self.assertNotIn("pending_approval", retired.metadata)
            self.assertEqual(retired.status, "cancelled")
            self.assertTrue(retired.cancel_requested)

    def test_new_run_retires_interrupted_running_turn(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = TurnStore(directory)
            interrupted = store.create(conversation_id="conversation-1")
            waiting = store.create(conversation_id="conversation-1")
            store.set_pending_approval(waiting.id, {"tool": "shell_exec"})
            other = store.create(conversation_id="conversation-2")

            failed = store.fail_interrupted_running_for_conversation("conversation-1")

            self.assertEqual(failed, 1)
            interrupted_after = store.load(interrupted.id)
            self.assertEqual(interrupted_after.status, "failed")
            self.assertIn("中断", interrupted_after.error)
            self.assertEqual(store.load(waiting.id).status, "waiting_approval")
            self.assertEqual(store.load(other.id).status, "running")

    def test_old_terminal_turn_logs_fold_to_public_path_and_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = TurnStore(directory)
            turn_ids: list[str] = []
            for index in range(4):
                turn = store.create(conversation_id="conversation-1")
                turn_ids.append(turn.id)
                store.append_event(
                    turn.id,
                    {
                        "event": "activity",
                        "phase": "thinking",
                        "title": "实施路径",
                        "detail": f"第 {index + 1} 轮先核对材料，再修改代码。",
                        "activity_type": "work_note",
                    },
                )
                store.append_event(
                    turn.id,
                    {
                        "event": "activity",
                        "phase": "action",
                        "title": "已运行命令",
                        "content": "数千行终端回显",
                        "command": "rg -n test .",
                        "activity_type": "command",
                    },
                )
                store.append_event(
                    turn.id,
                    {"event": "final", "content": f"第 {index + 1} 轮完成。"},
                )

            changed = store.compact_terminal_history("conversation-1", keep_recent=2)

            self.assertEqual(changed, 0)
            loaded = [store.load(turn_id) for turn_id in turn_ids]
            compacted_turns = [
                turn for turn in loaded if turn.metadata.get("runtime_log_compacted")
            ]
            self.assertEqual(len(compacted_turns), 0)
            serialized = json.dumps(
                [event for turn in loaded for event in turn.events],
                ensure_ascii=False,
            )
            self.assertIn("先核对材料，再修改代码", serialized)
            self.assertNotIn("执行细节已折叠", serialized)
            self.assertIn("数千行终端回显", serialized)

    def test_rewind_ordinal_payload_validation(self) -> None:
        self.assertIsNone(sanitize_rewind_user_message_ordinal({}))
        self.assertEqual(sanitize_rewind_user_message_ordinal({"rewind_user_message_ordinal": "2"}), 2)
        with self.assertRaises(ValueError):
            sanitize_rewind_user_message_ordinal({"rewind_user_message_ordinal": -1})


if __name__ == "__main__":
    unittest.main()
