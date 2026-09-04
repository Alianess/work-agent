from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from work_agent_core.artifact_ledger import ToolArtifactCollector
from work_agent_core.conversation_repository import ConversationRepository
from work_agent_core.external_event_inbox import ExternalEventInboxAdapter
from work_agent_core.react import parse_tool_arguments_result
from work_agent_core.regression_eval import evaluate_regression
from work_agent_core.session_log import AGENT_ERROR, SessionHeader, SessionLog
from work_agent_core.session_log_store import SessionLogStore
from work_agent_core.session_runtime import ConversationRuntime
from work_agent_core.session_store import ConversationSession, SessionStore
from work_agent_core.turn_runtime import TurnRuntime
from work_agent_core.turn_store import TurnStore


class ArchitectureRegressionEvalTests(unittest.TestCase):
    def assert_case(self, case_id: str, outcome_ok: bool, runtime: ConversationRuntime) -> None:
        failures = evaluate_regression(
            case_id,
            outcome_ok=outcome_ok,
            trajectory=[event.type for event in runtime.log.events],
        )
        self.assertEqual(failures, [], "\n".join(failures))

    def test_missing_chat_is_rebuilt_from_log_not_divergent_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sessions = SessionStore(root, session_dir=root / "sessions")
            logs = SessionLogStore(root / "events.sqlite3")
            repository = ConversationRepository(root, session_store=sessions, log_store=logs)
            sessions.save(ConversationSession(id="chat", messages=[
                {"role": "user", "content": "产业方案"},
                {"role": "assistant", "content": "已定位"},
            ]))
            repository.load("chat")
            bad_cache = sessions.load("chat")
            bad_cache.messages = [{"role": "assistant", "content": "空壳"}]
            sessions.save(bad_cache)

            recovered = repository.load("chat")
            runtime = ConversationRuntime(logs.load_live("chat"))
            self.assert_case(
                "missing_chat_after_restart",
                [item["content"] for item in recovered.messages] == ["产业方案", "已定位"],
                runtime,
            )

    def test_error_is_trace_evidence_not_a_persistent_chat_message(self) -> None:
        runtime = ConversationRuntime(SessionLog(SessionHeader(session_id="chat")))
        runtime.begin_turn("t")
        runtime.record_user("继续")
        runtime.log.append(AGENT_ERROR, {"phase": "model", "message": "connection reset"})
        runtime.end_turn("failed", detail="connection reset")

        timeline = runtime.log.derive_timeline()
        self.assert_case(
            "stale_error_delivery_state",
            len(timeline) == 1 and timeline[0]["content"] == "继续",
            runtime,
        )

    def test_disconnect_recovery_preserves_one_tool_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SessionLogStore(Path(directory) / "events.sqlite3")
            runtime = ConversationRuntime(SessionLog(SessionHeader(session_id="chat")))
            runtime.begin_turn("t")
            runtime.begin_step(1)
            runtime.record_assistant("", tool_calls=[{
                "id": "call-1",
                "type": "function",
                "function": {"name": "save", "arguments": "{}"},
            }])
            runtime.record_tool_call("call-1", "save", "{}")
            runtime.record_tool_result("call-1", "save", '{"ok":true}')
            store.put_header(runtime.log.header)
            store.append("chat", runtime.log.events)

            recovered = ConversationRuntime(store.load("chat"))
            results = [event for event in recovered.log.events if event.type == "tool/result"]
            self.assert_case("disconnect_after_tool_result", len(results) == 1, recovered)

    def test_delivery_comes_from_artifact_event_not_final_prose(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "work_reports" / "report.md"
            output.parent.mkdir(parents=True)
            output.write_text("done", encoding="utf-8")
            runtime = ConversationRuntime(SessionLog(SessionHeader(session_id="chat")))
            runtime.begin_turn("t")
            runtime.begin_step(1)
            runtime.record_tool_result("call-1", "save", json.dumps({
                "ok": True,
                "content_path": "work_reports/report.md",
            }))
            collector = ToolArtifactCollector(root)
            collector.record_tool_result(
                runtime,
                tool_name="save",
                tool_input={},
                observation=json.dumps({"content_path": "work_reports/report.md"}),
                step=1,
            )
            runtime.end_turn()

            self.assert_case(
                "unstructured_delivery",
                runtime.log.derive_artifacts()[0]["path"] == "work_reports/report.md",
                runtime,
            )

    def test_mid_turn_followup_is_logged_then_acknowledged(self) -> None:
        store = TurnStore(tempfile.mkdtemp())
        turn = TurnRuntime.start(store, conversation_id="chat")
        turn.enqueue_message("补充约束")
        runtime = ConversationRuntime(SessionLog(SessionHeader(session_id="chat")))
        runtime.begin_turn(turn.turn_id)
        adapter = ExternalEventInboxAdapter(turn, runtime)
        texts = adapter.take_messages()
        for text in texts:
            runtime.record_user(text, source="queued_followup")
        runtime.flush()
        adapter.acknowledge(texts)

        self.assert_case(
            "lost_mid_turn_followup",
            texts == ["补充约束"] and not store.load(turn.turn_id).queued_messages,
            runtime,
        )

    def test_invalid_tool_json_requires_repair_before_call_and_result(self) -> None:
        parsed, error = parse_tool_arguments_result('{"content":')
        repaired, repaired_error = parse_tool_arguments_result('{"content":"fixed"}')
        trajectory = ["tool/arguments_invalid", "tool/call", "tool/result"]
        failures = evaluate_regression(
            "invalid_tool_json",
            outcome_ok=bool(error) and not parsed and not repaired_error and repaired["content"] == "fixed",
            trajectory=trajectory,
        )
        self.assertEqual(failures, [], "\n".join(failures))

    def test_large_archive_uses_bounded_checkpoint_workset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sessions = SessionStore(root, session_dir=root / "sessions")
            logs = SessionLogStore(root / "events.sqlite3")
            repository = ConversationRepository(root, session_store=sessions, log_store=logs)
            messages = [
                {"role": "user" if index % 2 == 0 else "assistant", "content": f"历史-{index}"}
                for index in range(200)
            ]
            sessions.save(ConversationSession(id="long-chat", messages=messages))
            bounded = repository.load("long-chat")
            bounded.messages = messages[-4:]
            bounded.summary = "前 196 条已形成断点摘要。"
            bounded.summary_message_count = 196
            repository.checkpoint(bounded)
            log = logs.load_live("long-chat")
            runtime = ConversationRuntime(log)
            runtime.record_request_header(
                system_prompt="system",
                late_blocks=(),
                tool_names=(),
                profile="test",
                model="test",
            )
            logs.append("long-chat", runtime.log.events[logs.next_seq("long-chat") :])

            recovered = repository.load("long-chat")
            self.assert_case(
                "unbounded_retrieval_context",
                len(recovered.messages) == 4 and len(runtime.log.derive_transcript()) == 200,
                runtime,
            )


if __name__ == "__main__":
    unittest.main()
