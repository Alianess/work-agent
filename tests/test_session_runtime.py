from __future__ import annotations

import tempfile
import unittest

from pathlib import Path

from work_agent_core.session_log import (
    REQUEST_HEADER,
    SessionHeader,
    SessionLog,
    TURN_END_ABORTED,
    check_session_invariants,
)
from work_agent_core.session_log_store import DurableTurnMirror, SessionLogStore
from work_agent_core.session_migration import build_seed_log, compare_derived_messages, migrate_conversation
from work_agent_core.session_runtime import ConversationRuntime
from work_agent_core.session_store import SessionStore


def new_runtime() -> ConversationRuntime:
    return ConversationRuntime(SessionLog(SessionHeader(session_id="conv-1")))


class RequestAssemblyTests(unittest.TestCase):
    def test_late_blocks_sit_immediately_before_the_newest_user_message(self) -> None:
        runtime = new_runtime()
        runtime.begin_turn("t1")
        runtime.begin_step(1)
        runtime.record_user("第一个问题")
        runtime.record_assistant("第一个回答")
        runtime.record_user("第二个问题")

        messages = runtime.build_request_messages(
            "系统提示", [{"role": "system", "content": "断点计划"}]
        )
        self.assertEqual(
            [(item["role"], item["content"]) for item in messages],
            [
                ("system", "系统提示"),
                ("user", "第一个问题"),
                ("assistant", "第一个回答"),
                ("system", "断点计划"),
                ("user", "第二个问题"),
            ],
        )

    def test_assembly_without_any_user_message_appends_late_blocks_last(self) -> None:
        runtime = new_runtime()
        runtime.begin_turn("t1")
        runtime.begin_step(1)
        messages = runtime.build_request_messages("系统提示", [{"role": "system", "content": "上下文"}])
        self.assertEqual([item["role"] for item in messages], ["system", "system"])


class TurnLifecycleTests(unittest.TestCase):
    def test_end_turn_closes_a_dangling_step_and_is_idempotent(self) -> None:
        runtime = new_runtime()
        runtime.begin_turn("t1")
        runtime.begin_step(1)
        runtime.record_user("做点事")
        runtime.end_turn(TURN_END_ABORTED)

        self.assertIsNone(runtime.end_turn(TURN_END_ABORTED))
        self.assertFalse(runtime.turn_is_open)
        self.assertEqual(check_session_invariants(runtime.log), [])

    def test_cancelled_turn_keeps_the_work_it_already_did(self) -> None:
        runtime = new_runtime()
        runtime.begin_turn("t1")
        runtime.begin_step(1)
        runtime.record_user("转写录音")
        runtime.record_assistant("", tool_calls=[
            {"id": "call_a", "type": "function", "function": {"name": "transcribe", "arguments": "{}"}}
        ])
        runtime.record_tool_call("call_a", "transcribe", {})
        runtime.record_tool_result("call_a", "transcribe", "转写完成，7264 字")
        runtime.end_turn(TURN_END_ABORTED)

        # The point of the rebuild: a cancel keeps everything already done.
        messages = runtime.build_request_messages("系统提示")
        self.assertEqual([item["role"] for item in messages], ["system", "user", "assistant", "tool"])
        self.assertIn("7264", messages[-1]["content"])
        self.assertEqual(check_session_invariants(runtime.log), [])


class CompactionTests(unittest.TestCase):
    def test_compaction_folds_the_active_turn_but_keeps_the_prompt(self) -> None:
        runtime = new_runtime()
        runtime.begin_turn("t1")
        runtime.begin_step(1)
        runtime.record_user("写材料")
        runtime.record_assistant("读取了底稿")
        runtime.record_tool_result("call_a", "read_text_file", "很长的正文" * 100)

        folded = runtime.active_turn_surface_seqs()
        self.assertEqual(len(folded), 2)
        runtime.record_compaction(folded, "检查点：底稿已读取")

        messages = runtime.build_request_messages("系统提示")
        self.assertEqual(
            [(item["role"], item["content"]) for item in messages],
            [("system", "系统提示"), ("user", "写材料"), ("assistant", "检查点：底稿已读取")],
        )
        # The prompt that started the turn survives compaction.
        self.assertEqual(messages[1]["content"], "写材料")
        # And the raw path is still readable for a human.
        transcript = runtime.log.derive_transcript()
        self.assertEqual(len(transcript), 3)


class RequestHeaderTests(unittest.TestCase):
    def test_request_header_records_what_history_cannot(self) -> None:
        runtime = new_runtime()
        runtime.begin_turn("t1")
        runtime.begin_step(1)
        runtime.record_request_header(
            system_prompt="系统提示",
            late_blocks=[{"role": "system", "content": "断点计划"}],
            tool_names=["shell_exec", "sys_skill"],
            profile="gpt-5.6-terra",
            model="gpt-5.6-terra",
            endpoint="https://example.invalid/v1/chat/completions",
            step=1,
            temperature=0.7,
            max_tokens=None,
        )
        header = runtime.log.latest(REQUEST_HEADER)
        self.assertIsNotNone(header)
        self.assertEqual(header.data["tool_names"], ("shell_exec", "sys_skill"))
        self.assertEqual(header.data["late_blocks"][0]["content"], "断点计划")
        self.assertEqual(header.data["params"]["temperature"], 0.7)
        # An unset parameter is omitted rather than recorded as null.
        self.assertNotIn("max_tokens", header.data["params"])
        # Log-only: it never reaches the derived message list.
        self.assertEqual(runtime.log.derive_messages(), [])


class MigrationTests(unittest.TestCase):
    def test_seed_log_is_structurally_valid_and_derives_the_same_history(self) -> None:
        messages = [
            {"role": "user", "content": "整理纪要"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "read_text_file", "arguments": "{\"path\": \"a.md\"}"}}
            ]},
            {"role": "tool", "tool_call_id": "call_1", "name": "read_text_file", "content": "正文"},
            {"role": "assistant", "content": "整理好了"},
            {"role": "user", "content": "再改一版"},
            {"role": "assistant", "content": "改好了"},
        ]
        log = build_seed_log("conv-1", messages)
        self.assertEqual(check_session_invariants(log), [])
        self.assertEqual(log.derive_messages(), messages)
        # Two user messages means two reconstructed turns.
        self.assertEqual(sum(1 for event in log.iter_type("turn/start")), 2)

    def test_history_starting_mid_conversation_still_gets_an_enclosure(self) -> None:
        log = build_seed_log("conv-2", [{"role": "assistant", "content": "先说一句"}])
        self.assertEqual(check_session_invariants(log), [])

    def test_migrating_twice_does_not_duplicate_history(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            session_dir = Path(root) / "sessions"
            session_store = SessionStore(root, session_dir=session_dir)
            session = session_store.load("conv-3")
            session.messages = [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "在"}]
            session_store.save(session)

            log_store = SessionLogStore(Path(root) / "log.sqlite3")
            first = migrate_conversation(session_store, log_store, "conv-3")
            second = migrate_conversation(session_store, log_store, "conv-3")

            self.assertEqual(first["status"], "migrated")
            self.assertEqual(second["status"], "skipped")
            self.assertEqual(compare_derived_messages(session_store, log_store, "conv-3"), [])


if __name__ == "__main__":
    unittest.main()


class PersistenceFidelityTests(unittest.TestCase):
    """Compaction must not be able to delete history from the durable store."""

    def test_compaction_folds_the_model_view_but_the_transcript_keeps_everything(self) -> None:
        runtime = ConversationRuntime.from_messages(
            [{"role": "user", "content": "老问题"}, {"role": "assistant", "content": "老答案"}]
        )
        before = len(runtime.log.derive_transcript())
        runtime.begin_turn("t1")
        runtime.begin_step(1)
        runtime.append_message({"role": "user", "content": "新问题"})
        for index in range(4):
            runtime.append_message({"role": "assistant", "content": f"中间推理{index}"})
        runtime.record_compaction(runtime.active_turn_surface_seqs(), "检查点")
        runtime.append_message({"role": "assistant", "content": "最终答复"})

        persisted = runtime.log.derive_transcript()[before:]
        self.assertEqual(
            [item["content"] for item in persisted],
            ["新问题", "中间推理0", "中间推理1", "中间推理2", "中间推理3", "最终答复"],
        )
        # The model, meanwhile, is spared the folded path.
        self.assertNotIn("中间推理0", [item["content"] for item in runtime.log.derive_messages()])

    def test_transcript_slicing_stays_aligned_when_the_model_view_shrinks(self) -> None:
        runtime = ConversationRuntime.from_messages([{"role": "user", "content": "开始"}])
        before_transcript = len(runtime.log.derive_transcript())
        before_messages = len(runtime.log.derive_messages())
        runtime.begin_turn("t1")
        runtime.begin_step(1)
        for index in range(3):
            runtime.append_message({"role": "assistant", "content": f"步骤{index}"})
        runtime.record_compaction(runtime.active_turn_surface_seqs(), "检查点")

        # The compacted projection is now shorter than it was before the turn,
        # so a positional slice into it would silently drop the whole turn.
        self.assertLess(len(runtime.log.derive_messages()), before_messages + 3)
        self.assertEqual(len(runtime.log.derive_transcript()[before_transcript:]), 3)


class SliceBasisTests(unittest.TestCase):
    def test_history_that_sanitizes_away_does_not_shift_the_turn_boundary(self) -> None:
        """Seeding drops unprojectable history, so list length is not a basis.

        Taking the basis from the caller's list instead of the seeded log made a
        whole turn slice to empty and vanish from the store without an error.
        """

        raw = [
            {"role": "user", "content": "问题"},
            {"role": "assistant", "content": ""},   # sanitizes away
            {"role": "assistant", "content": "答案"},
        ]
        runtime = ConversationRuntime.from_messages(raw)
        basis = len(runtime.log.derive_transcript())
        self.assertLess(basis, len(raw))

        runtime.begin_turn("t1")
        runtime.begin_step(1)
        runtime.append_message({"role": "user", "content": "新一轮"})

        self.assertEqual(
            [item["content"] for item in runtime.log.derive_transcript()[basis:]], ["新一轮"]
        )
        self.assertEqual([item["content"] for item in runtime.log.derive_transcript()[len(raw):]], [])


class DurableMirrorTests(unittest.TestCase):
    def _turn(self, store, cid, seed, prompt):
        runtime = ConversationRuntime.from_messages(seed, session_id=cid)
        mirror = DurableTurnMirror(store, cid, skip_before_seq=runtime.log.seq)
        runtime.writer = mirror
        mirror.record_prompt(prompt)
        runtime.begin_turn("t")
        runtime.begin_step(1)
        return runtime, mirror

    def test_durable_log_survives_a_crash_and_reconstructs_the_conversation(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            store = SessionLogStore(Path(root) / "log.sqlite3")
            cid = "conv-1"

            runtime, mirror = self._turn(store, cid, [{"role": "user", "content": "第一个问题"}], "第一个问题")
            runtime.append_message({"role": "assistant", "content": "第一个回答"})
            runtime.end_turn()
            mirror.flush()

            # Second turn dies mid-flight: events are durable, the turn is not closed.
            runtime2, mirror2 = self._turn(
                store,
                cid,
                [
                    {"role": "user", "content": "第一个问题"},
                    {"role": "assistant", "content": "第一个回答"},
                    {"role": "user", "content": "转写录音"},
                ],
                "转写录音",
            )
            runtime2.append_message({"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_a", "type": "function",
                 "function": {"name": "transcribe", "arguments": "{}"}}]})
            mirror2.flush()

            recovered = store.load(cid)
            self.assertEqual(
                [item["role"] for item in recovered.derive_messages()],
                ["user", "assistant", "user", "assistant", "tool"],
            )
            self.assertIn("aborted before dispatch", recovered.derive_messages()[-1]["content"])
            ends = list(recovered.iter_type("turn/end"))
            self.assertEqual(dict(ends[-1].data)["reason"]["kind"], "interrupted")
            self.assertEqual(check_session_invariants(recovered), [])

    def test_mirror_failure_never_propagates_into_the_turn(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            store = SessionLogStore(Path(root) / "log.sqlite3")
            runtime, mirror = self._turn(store, "conv-2", [{"role": "user", "content": "问"}], "问")
            runtime.append_message({"role": "assistant", "content": "答"})
            # Force a contiguity rejection by advancing the durable log underneath.
            mirror.skip_before_seq = 10_000
            mirror._offset = -10_000
            mirror.flush()
            self.assertIsNotNone(mirror.last_error)
