from __future__ import annotations

import unittest

from work_agent_core.facts import (
    ADD,
    COMMITMENT,
    Fact,
    NOOP,
    UPDATE,
    existing_facts,
    gate_turn,
    reconcile,
    record_facts,
    turn_transcript,
)
from work_agent_core.session_log import SessionHeader, SessionLog
from work_agent_core.session_runtime import ConversationRuntime


def turn(text: str, tool: str = "") -> SessionLog:
    log = SessionLog(SessionHeader(session_id="c"))
    log.append("turn/start", {"turn_id": "t"})
    log.append("step/start", {"step": 1})
    log.append("user/message", {"content": text})
    if tool:
        log.append("tool/call", {"call_id": "c1", "name": tool, "arguments": "{}"})
    return log


class GateTests(unittest.TestCase):
    """Most turns leave nothing durable, and that is cheap to establish."""

    def test_turns_worth_a_model_call(self) -> None:
        for text in (
            "报市材料明天晚上出一稿，周三前提交",
            "批示要求定期报告工作进展，定成双周报",
            "不是柔性科天，是水性科天写错了",
            "下周一之前把双周报发我",
        ):
            self.assertTrue(gate_turn(turn(text)).worth_reading, text)

    def test_ordinary_turns_are_skipped(self) -> None:
        for text in (
            "帮我看看这个 PDF",
            "你好",
            "这个怎么弄",
            "解释一下什么是具身智能",
            "刚才那段再改改",
        ):
            self.assertFalse(gate_turn(turn(text)).worth_reading, text)

    def test_producing_work_is_always_worth_reading(self) -> None:
        self.assertTrue(gate_turn(turn("整理一下", tool="write_text_file")).worth_reading)


class ReconcileTests(unittest.TestCase):
    """Appending a fresh copy every turn is how a memory store fills with noise."""

    def setUp(self) -> None:
        self.known = [
            Fact(kind=COMMITMENT, subject="报市材料", statement="周三前提交", due_at="2026-08-19")
        ]

    def test_an_unseen_fact_is_added(self) -> None:
        candidate = Fact(kind=COMMITMENT, subject="双周报", statement="每两周提交一次")
        self.assertEqual(reconcile(candidate, self.known)[0], ADD)

    def test_repeating_a_known_fact_changes_nothing(self) -> None:
        candidate = Fact(
            kind=COMMITMENT, subject="报市材料", statement="周三前提交", due_at="2026-08-19"
        )
        self.assertEqual(reconcile(candidate, self.known)[0], NOOP)

    def test_a_changed_deadline_updates_rather_than_duplicates(self) -> None:
        candidate = Fact(
            kind=COMMITMENT, subject="报市材料", statement="周四前提交", due_at="2026-08-20"
        )
        operation, resolved = reconcile(candidate, self.known)
        self.assertEqual(operation, UPDATE)
        self.assertEqual(resolved.due_at, "2026-08-20")


class RecordingTests(unittest.TestCase):
    def test_facts_land_in_the_log_and_never_reach_the_model_history(self) -> None:
        runtime = ConversationRuntime(SessionLog(SessionHeader(session_id="c")))
        runtime.begin_turn("t1")
        runtime.begin_step(1)
        runtime.record_user("报市材料周三前提交")

        applied = record_facts(
            runtime,
            [Fact(kind=COMMITMENT, subject="报市材料", statement="周三前提交", due_at="2026-08-19")],
        )
        self.assertEqual([operation for operation, _ in applied], [ADD])
        self.assertEqual(len(existing_facts(runtime.log)), 1)
        # A fact is a record about the conversation, not a message inside it.
        self.assertEqual(
            [item["role"] for item in runtime.build_request_messages("系统提示")],
            ["system", "user"],
        )

    def test_recording_the_same_fact_twice_does_not_duplicate_it(self) -> None:
        runtime = ConversationRuntime(SessionLog(SessionHeader(session_id="c")))
        runtime.begin_turn("t1")
        runtime.begin_step(1)
        fact = Fact(kind=COMMITMENT, subject="报市材料", statement="周三前提交")
        record_facts(runtime, [fact])
        record_facts(runtime, [fact])
        self.assertEqual(len(existing_facts(runtime.log)), 1)

    def test_incomplete_candidates_are_dropped(self) -> None:
        runtime = ConversationRuntime(SessionLog(SessionHeader(session_id="c")))
        runtime.begin_turn("t1")
        runtime.begin_step(1)
        record_facts(
            runtime,
            [
                Fact(kind=COMMITMENT, subject="", statement="没有主语"),
                Fact(kind=COMMITMENT, subject="有主语", statement=""),
                Fact(kind="不认识的种类", subject="x", statement="y"),
            ],
        )
        self.assertEqual(existing_facts(runtime.log), [])


class TranscriptTests(unittest.TestCase):
    def test_transcript_carries_what_was_asked_and_answered(self) -> None:
        log = SessionLog(SessionHeader(session_id="c"))
        log.append("turn/start", {"turn_id": "t"})
        log.append("step/start", {"step": 1})
        log.append("user/message", {"content": "周三前交"})
        log.append("assistant/message", {"content": "记下了"})
        log.append("tool/call", {"call_id": "x", "name": "read_text_file", "arguments": "{}"})
        text = turn_transcript(log)
        self.assertIn("用户：周三前交", text)
        self.assertIn("助手：记下了", text)
        self.assertNotIn("read_text_file", text)


if __name__ == "__main__":
    unittest.main()
