from __future__ import annotations

import tempfile
import unittest

from pathlib import Path

from work_agent_core.session_log import (
    ABORTED_BEFORE_DISPATCH,
    ASSISTANT_MESSAGE,
    COMPACTION_REPLACEMENT,
    CONTEXT_INJECTED,
    SessionHeader,
    SessionLog,
    SessionLogError,
    STEP_END,
    STEP_START,
    TOOL_CALL,
    TOOL_RESULT,
    TURN_END,
    TURN_END_ABORTED,
    TURN_END_INTERRUPTED,
    TURN_START,
    USER_MESSAGE,
    check_session_invariants,
    synthesize_interrupted_turn_ends,
)
from work_agent_core.session_log_store import SessionLogStore, SessionLogWriter


def new_log() -> SessionLog:
    return SessionLog(SessionHeader(session_id="conv-1"))


def open_turn(log: SessionLog, *, turn_id: str = "turn-1", step: int = 1) -> None:
    log.append(TURN_START, {"turn_id": turn_id})
    log.append(STEP_START, {"step": step})


class SessionLogProjectionTests(unittest.TestCase):
    def test_messages_derive_from_the_log_and_skip_log_only_events(self) -> None:
        log = new_log()
        open_turn(log)
        log.append(USER_MESSAGE, {"content": "整理会议纪要"})
        log.append(ASSISTANT_MESSAGE, {"content": "", "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "read_text_file", "arguments": "{}"}}
        ]})
        log.append(TOOL_CALL, {"call_id": "call_1", "name": "read_text_file", "arguments": {}})
        log.append(TOOL_RESULT, {"call_id": "call_1", "name": "read_text_file", "content": "纪要正文"})
        log.append(ASSISTANT_MESSAGE, {"content": "已整理完成"})

        messages = log.derive_messages()
        self.assertEqual(
            [item["role"] for item in messages],
            ["user", "assistant", "tool", "assistant"],
        )
        # tool/call is durable but never sent on its own: the assistant message
        # already carries tool_calls, so replaying it would duplicate the call.
        self.assertEqual(messages[1]["tool_calls"][0]["id"], "call_1")
        self.assertEqual(messages[2]["tool_call_id"], "call_1")
        self.assertEqual(check_session_invariants(self._closed(log)), [])

    def test_injected_context_is_model_visible_and_recorded(self) -> None:
        log = new_log()
        open_turn(log)
        log.append(CONTEXT_INJECTED, {"kind": "task_plan", "role": "system", "content": "断点计划"})
        log.append(USER_MESSAGE, {"content": "继续"})

        messages = log.derive_messages()
        self.assertEqual(messages[0], {"role": "system", "content": "断点计划"})
        # The whole point: what the model saw is reconstructible from the log.
        self.assertEqual(log.event(2).type, CONTEXT_INJECTED)

    def test_durable_data_is_frozen_against_later_mutation(self) -> None:
        log = new_log()
        open_turn(log)
        payload = {"content": "原始答复"}
        log.append(ASSISTANT_MESSAGE, payload)

        payload["content"] = "被改写的答复"
        self.assertEqual(log.derive_messages()[0]["content"], "原始答复")

        with self.assertRaises(TypeError):
            log.event(2).data["content"] = "直接改事件"

    def test_derived_messages_are_detached_copies(self) -> None:
        log = new_log()
        open_turn(log)
        log.append(USER_MESSAGE, {"content": "你好"})
        first = log.derive_messages()
        first[0]["content"] = "改掉"
        self.assertEqual(log.derive_messages()[0]["content"], "你好")

    def _closed(self, log: SessionLog) -> SessionLog:
        log.append(STEP_END, {"step": 1})
        log.append(TURN_END, {"turn_id": "turn-1", "reason": {"kind": "completed"}})
        return log


class CompactionTests(unittest.TestCase):
    def test_replacement_shadows_the_model_view_but_keeps_the_transcript(self) -> None:
        log = new_log()
        open_turn(log)
        log.append(USER_MESSAGE, {"content": "写材料"})
        first = log.append(ASSISTANT_MESSAGE, {"content": "第一段冗长推理"})
        second = log.append(ASSISTANT_MESSAGE, {"content": "第二段冗长推理"})
        log.append(
            COMPACTION_REPLACEMENT,
            {"content": "检查点：已读取底稿，正在撰写"},
            source_event_seqs=[first.seq, second.seq],
        )

        model_view = log.derive_messages()
        self.assertEqual([item["content"] for item in model_view],
                         ["写材料", "检查点：已读取底稿，正在撰写"])

        transcript = log.derive_transcript()
        self.assertEqual(
            [item["content"] for item in transcript],
            ["写材料", "第一段冗长推理", "第二段冗长推理"],
        )

    def test_replacement_must_cite_earlier_events(self) -> None:
        log = new_log()
        open_turn(log)
        with self.assertRaises(SessionLogError):
            log.append(COMPACTION_REPLACEMENT, {"content": "空引用"})
        with self.assertRaises(SessionLogError):
            log.append(COMPACTION_REPLACEMENT, {"content": "越界"}, source_event_seqs=[99])


class InterruptionRecoveryTests(unittest.TestCase):
    def test_open_turn_is_closed_as_interrupted_with_synthetic_results(self) -> None:
        log = new_log()
        open_turn(log)
        log.append(USER_MESSAGE, {"content": "转写录音"})
        log.append(ASSISTANT_MESSAGE, {"content": "", "tool_calls": [
            {"id": "call_a", "type": "function", "function": {"name": "sys_skill", "arguments": "{}"}},
            {"id": "call_b", "type": "function", "function": {"name": "sys_skill", "arguments": "{}"}},
        ]})
        log.append(TOOL_CALL, {"call_id": "call_a", "name": "sys_skill"})
        log.append(TOOL_RESULT, {"call_id": "call_a", "name": "sys_skill", "content": "完成"})
        log.append(TOOL_CALL, {"call_id": "call_b", "name": "sys_skill"})
        # process dies here, before call_b is dispatched

        recovered = synthesize_interrupted_turn_ends(log.events)
        self.assertEqual([event.type for event in recovered], [TOOL_RESULT, STEP_END, TURN_END])
        self.assertEqual(recovered[0].data["error"], ABORTED_BEFORE_DISPATCH)
        self.assertEqual(recovered[0].data["call_id"], "call_b")
        self.assertEqual(recovered[-1].data["reason"]["kind"], TURN_END_INTERRUPTED)
        self.assertTrue(recovered[-1].data["synthesized"])

        replayed = SessionLog(log.header, list(log.events) + recovered)
        self.assertEqual(check_session_invariants(replayed), [])

    def test_closed_turn_needs_no_recovery(self) -> None:
        log = new_log()
        open_turn(log)
        log.append(STEP_END, {"step": 1})
        log.append(TURN_END, {"turn_id": "turn-1", "reason": {"kind": TURN_END_ABORTED}})
        self.assertEqual(synthesize_interrupted_turn_ends(log.events), [])


class InvariantTests(unittest.TestCase):
    def test_unpaired_tool_call_within_a_step_is_reported(self) -> None:
        log = new_log()
        open_turn(log)
        log.append(TOOL_CALL, {"call_id": "call_x", "name": "shell_exec"})
        log.append(STEP_END, {"step": 1})
        log.append(TURN_END, {"turn_id": "turn-1", "reason": {"kind": "completed"}})
        problems = check_session_invariants(log)
        self.assertTrue(any("call_x" in item for item in problems), problems)

    def test_interrupted_reason_written_by_a_loop_is_rejected(self) -> None:
        log = new_log()
        log.append(TURN_START, {"turn_id": "turn-1"})
        log.append(TURN_END, {"turn_id": "turn-1", "reason": {"kind": TURN_END_INTERRUPTED}})
        problems = check_session_invariants(log)
        self.assertTrue(any("interrupted" in item for item in problems), problems)

    def test_nested_turn_and_dangling_step_are_reported(self) -> None:
        log = new_log()
        log.append(TURN_START, {"turn_id": "turn-1"})
        log.append(TURN_START, {"turn_id": "turn-2"})
        log.append(STEP_START, {"step": 1})
        log.append(TURN_END, {"turn_id": "turn-2", "reason": {"kind": "completed"}})
        problems = check_session_invariants(log)
        self.assertTrue(any("未关闭的 turn" in item for item in problems), problems)
        self.assertTrue(any("仍未关闭" in item for item in problems), problems)


class SessionLogStoreTests(unittest.TestCase):
    def test_round_trip_preserves_events_and_projections(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            store = SessionLogStore(Path(root) / "log.sqlite3")
            log = new_log()
            store.put_header(log.header)
            open_turn(log)
            log.append(USER_MESSAGE, {"content": "你好"})
            log.append(ASSISTANT_MESSAGE, {"content": "在的"})
            log.append(STEP_END, {"step": 1})
            log.append(TURN_END, {"turn_id": "turn-1", "reason": {"kind": "completed"}})
            store.append("conv-1", log.events)

            loaded = store.load("conv-1")
            self.assertEqual(loaded.derive_messages(), log.derive_messages())
            self.assertEqual(loaded.seq, log.seq)
            self.assertEqual(check_session_invariants(loaded), [])

    def test_append_rejects_a_gap(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            store = SessionLogStore(Path(root) / "log.sqlite3")
            log = new_log()
            log.append(TURN_START, {"turn_id": "turn-1"})
            log.append(STEP_START, {"step": 1})
            log.append(STEP_END, {"step": 1})
            store.append("conv-1", log.events[:1])

            # seq 1 is missing, so a batch starting at seq 2 leaves a gap.
            with self.assertRaises(SessionLogError):
                store.append("conv-1", log.events[2:])
            # replaying an already-landed batch is the same violation
            with self.assertRaises(SessionLogError):
                store.append("conv-1", log.events[:1])
            self.assertEqual(store.next_seq("conv-1"), 1)

    def test_load_persists_the_synthesized_interruption(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            store = SessionLogStore(Path(root) / "log.sqlite3")
            log = new_log()
            open_turn(log)
            log.append(USER_MESSAGE, {"content": "跑一下"})
            store.append("conv-1", log.events)

            first = store.load("conv-1")
            self.assertEqual(first.events[-1].type, TURN_END)
            self.assertEqual(first.events[-1].data["reason"]["kind"], TURN_END_INTERRUPTED)

            # A second reader sees the same history the first one returned.
            second = store.load("conv-1")
            self.assertEqual(second.seq, first.seq)
            self.assertEqual(check_session_invariants(second), [])

    def test_revision_changes_only_when_events_land(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            store = SessionLogStore(Path(root) / "log.sqlite3")
            before = store.revision("conv-1")
            log = new_log()
            log.append(TURN_START, {"turn_id": "turn-1"})
            store.append("conv-1", log.events)
            after = store.revision("conv-1")
            self.assertEqual(before.next_seq, 0)
            self.assertEqual(after.next_seq, 1)
            self.assertNotEqual(before, after)


class SessionLogWriterTests(unittest.TestCase):
    def test_write_behind_batches_and_flush_is_the_barrier(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            store = SessionLogStore(Path(root) / "log.sqlite3")
            writer = SessionLogWriter(store, "conv-1", window_seconds=5.0)
            try:
                log = new_log()
                open_turn(log)
                for event in log.events:
                    writer.on_event(event)
                # Still inside the batching window: nothing is durable yet.
                self.assertEqual(store.next_seq("conv-1"), 0)
                writer.flush()
                self.assertEqual(store.next_seq("conv-1"), 2)
                self.assertEqual(writer.pending_count, 0)
            finally:
                writer.close()

    def test_a_rejected_write_is_retained_and_surfaced_by_flush(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            store = SessionLogStore(Path(root) / "log.sqlite3")
            log = new_log()
            log.append(TURN_START, {"turn_id": "turn-1"})
            store.append("conv-1", log.events)

            writer = SessionLogWriter(store, "conv-1", window_seconds=0.01)
            try:
                # seq 0 already landed, so this batch collides on contiguity.
                writer.on_event(log.events[0])
                with self.assertRaises(SessionLogError):
                    writer.flush()
                self.assertEqual(writer.pending_count, 1)
            finally:
                try:
                    writer.close()
                except SessionLogError:
                    pass


if __name__ == "__main__":
    unittest.main()
