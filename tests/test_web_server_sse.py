from __future__ import annotations

import io
import json
import unittest
from types import MethodType, SimpleNamespace

from work_agent_core.web_server import (
    WorkAgentHandler,
    context_compaction_detail,
    context_compaction_metrics,
    mark_task_plan_run_status,
)
from work_agent_core.session_runtime import ConversationRuntime
from work_agent_core.session_store import ConversationSession


class _DisconnectingWriter:
    def write(self, _data: bytes) -> None:
        raise BrokenPipeError("client closed connection")

    def flush(self) -> None:
        return None


class _CapturingWriter(io.BytesIO):
    def flush(self) -> None:
        return None


class ServerSentEventTests(unittest.TestCase):
    def test_failed_turn_marks_resumable_plan_terminal_without_dropping_steps(self) -> None:
        state = {
            "turn_id": "turn-1",
            "steps": [{"step": "完成材料", "status": "in_progress"}],
        }
        session = ConversationSession(
            id="chat-1",
            metadata={"active_task_plan": state},
        )
        runtime = ConversationRuntime.from_messages([], session_id="chat-1")

        mark_task_plan_run_status(
            session,
            runtime,
            "turn-1",
            "failed",
            detail="模型额度不足",
        )

        marked = session.metadata["active_task_plan"]
        self.assertEqual(marked["run_status"], "failed")
        self.assertEqual(marked["steps"], state["steps"])
        self.assertEqual(marked["terminal_detail"], "模型额度不足")

    def test_compaction_detail_distinguishes_token_and_non_token_pressure(self) -> None:
        prepared = SimpleNamespace(
            estimated_tokens=134_518,
            post_compaction_estimated_tokens=61_204,
            serialized_bytes=412_000,
            tool_result_chars=173_000,
            post_compaction_serialized_bytes=31_000,
            post_compaction_tool_result_chars=0,
            pressure_reasons=("serialized_bytes", "tool_results"),
            summary_message_count=38,
        )

        detail = context_compaction_detail(prepared, 230_400)
        metrics = context_compaction_metrics(prepared, 230_400)

        self.assertIn("134,518 tokens，低于 230,400 token 安全线", detail)
        self.assertIn("待整理会话体积 412,000 字节", detail)
        self.assertIn("历史工具结果累计 173,000 字符", detail)
        self.assertIn("压缩后工作上下文约 61,204 tokens", detail)
        self.assertNotIn("134,518 tokens，已超过 230,400", detail)
        self.assertEqual(metrics["context_pre_compaction_tokens"], 134_518)
        self.assertEqual(metrics["context_post_compaction_tokens"], 61_204)
        self.assertEqual(
            metrics["context_pressure_reasons"],
            ["serialized_bytes", "tool_results"],
        )

    def test_unicode_paragraph_separator_does_not_split_json_data_line(self) -> None:
        writer = _CapturingWriter()
        handler = SimpleNamespace(wfile=writer)

        WorkAgentHandler._write_sse_event(
            handler,
            {"event": "activity", "content": "第一段\u2029第二段"},
        )

        encoded = writer.getvalue().decode("utf-8")
        self.assertEqual(encoded.count("data:"), 1)
        self.assertIn("\\u2029", encoded)
        data = encoded.split("data: ", 1)[1].split("\n\n", 1)[0]
        self.assertEqual(json.loads(data)["content"], "第一段\u2029第二段")

    def test_client_disconnect_ends_stream_without_secondary_exception(self) -> None:
        handler = SimpleNamespace(
            wfile=_DisconnectingWriter(),
            send_response=lambda _status: None,
            _send_common_headers=lambda: None,
            send_header=lambda _name, _value: None,
            end_headers=lambda: None,
        )
        handler._write_sse_event = MethodType(WorkAgentHandler._write_sse_event, handler)

        WorkAgentHandler._send_sse(handler, iter([{"event": "delta", "content": "hello"}]))


if __name__ == "__main__":
    unittest.main()
