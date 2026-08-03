from __future__ import annotations

import time
import threading
import unittest
import json

from work_agent_core.config import ModelProfile
from work_agent_core.approval_review import ApprovalReview
from work_agent_core.llm import LLMResponse, LLMStreamChunk
from work_agent_core.react import (
    NativeToolCall,
    ReActAgent,
    pending_tool_batch_state,
    tool_observation_failed,
)
from work_agent_core.tool_bus import LocalToolProvider, ToolBus
from work_agent_core.tools import Tool


class _StalledClient:
    def chat_tools_stream(self, *_args, on_delta=None, **_kwargs) -> LLMResponse:
        if on_delta:
            on_delta(LLMStreamChunk(reasoning="partial"))
        time.sleep(2.0)
        return LLMResponse(
            content="late",
            raw={"choices": [{"message": {"role": "assistant", "content": "late"}}]},
        )


class _NoStreamClient:
    def chat_tools_stream(self, *_args, cancel_event=None, **_kwargs) -> LLMResponse:
        if cancel_event is not None:
            cancel_event.wait(5)
        raise RuntimeError("cancelled before first stream chunk")


class _ActiveReasoningClient:
    def chat_tools_stream(self, *_args, on_delta=None, **_kwargs) -> LLMResponse:
        for _ in range(9):
            if on_delta:
                on_delta(LLMStreamChunk(reasoning="still working"))
            time.sleep(0.15)
        return LLMResponse(
            content="done",
            raw={"choices": [{"message": {"role": "assistant", "content": "done"}}]},
        )


class _RecoveryAfterActiveReasoningClient:
    def chat_tools_stream(self, *_args, on_delta=None, **_kwargs) -> LLMResponse:
        for _ in range(4):
            if on_delta:
                on_delta(LLMStreamChunk(reasoning="still working"))
            time.sleep(0.15)
        if on_delta:
            on_delta(LLMStreamChunk(status="recovery_started"))
        # The recovery phase may legitimately be quieter than the primary
        # stream's idle budget. It has its own bounded deadline.
        time.sleep(1.1)
        return LLMResponse(
            content="recovered",
            raw={"choices": [{"message": {"role": "assistant", "content": "recovered"}}]},
        )


class _PartialContentThenRecoveryClient:
    def chat_tools_stream(self, *_args, on_delta=None, **_kwargs) -> LLMResponse:
        if on_delta:
            on_delta(LLMStreamChunk(content="流"))
            time.sleep(0.3)
            on_delta(LLMStreamChunk(status="recovery_started"))
            on_delta(LLMStreamChunk(content="流式状态机验证通过", status="recovery_streaming"))
        return LLMResponse(
            content="流式状态机验证通过",
            raw={
                "choices": [
                    {"message": {"role": "assistant", "content": "流式状态机验证通过"}}
                ]
            },
        )


class _CancellableStalledClient:
    def __init__(self) -> None:
        self.cancelled = threading.Event()

    def chat_tools_stream(
        self,
        *_args,
        on_delta=None,
        cancel_event=None,
        **_kwargs,
    ) -> LLMResponse:
        if on_delta:
            on_delta(LLMStreamChunk(reasoning="partial"))
        if cancel_event is not None and cancel_event.wait(5):
            self.cancelled.set()
            raise RuntimeError("cancelled")
        raise RuntimeError("stream was not cancelled")


class _ToolThenFinalClient:
    def __init__(self) -> None:
        self.calls = 0

    def chat_tools_stream(self, *_args, **_kwargs) -> LLMResponse:
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                content="",
                raw={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "call_test",
                                        "type": "function",
                                        "function": {
                                            "name": "test_tool",
                                            "arguments": "{\"value\": \"ok\"}",
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                },
            )
        return LLMResponse(
            content="done",
            raw={"choices": [{"message": {"role": "assistant", "content": "done"}}]},
        )


class _ReportSaveThenNetworkFailureClient:
    def __init__(self) -> None:
        self.calls = 0

    def chat_tools_stream(self, *_args, **_kwargs) -> LLMResponse:
        self.calls += 1
        if self.calls > 1:
            raise RuntimeError("TLS handshake timed out")
        return LLMResponse(
            content="日报内容已核验，正在保存。",
            raw={
                "choices": [{"message": {
                    "role": "assistant",
                    "content": "日报内容已核验，正在保存。",
                    "tool_calls": [{
                        "id": "call_report",
                        "type": "function",
                        "function": {
                            "name": "sys_skill",
                            "arguments": json.dumps({
                                "op": "call",
                                "skill_id": "work-reports",
                                "tool_name": "save_work_report",
                                "arguments": {
                                    "report_type": "daily",
                                    "content": "test",
                                },
                            }, ensure_ascii=False),
                        },
                    }],
                }}],
            },
        )


class _StreamingFinalClient:
    def chat_tools_stream(self, *_args, on_delta=None, **_kwargs) -> LLMResponse:
        if on_delta:
            on_delta(LLMStreamChunk(reasoning="internal"))
            on_delta(LLMStreamChunk(content="你好"))
            on_delta(LLMStreamChunk(content="，世界"))
        return LLMResponse(
            content="你好，世界",
            raw={"choices": [{"message": {"role": "assistant", "content": "你好，世界"}}]},
        )


class _ShellThenFinalClient:
    def __init__(self) -> None:
        self.calls = 0

    def chat_tools_stream(self, *_args, **_kwargs) -> LLMResponse:
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                content="",
                raw={
                    "choices": [{"message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "id": "call_shell",
                            "type": "function",
                            "function": {
                                "name": "shell_exec",
                                "arguments": '{"command":"mkdir output"}',
                            },
                        }],
                    }}],
                },
            )
        return LLMResponse(
            content="done",
            raw={"choices": [{"message": {"role": "assistant", "content": "done"}}]},
        )


class _ApprovalReviewerStub:
    def __init__(self, *, approve: bool) -> None:
        self.approve = approve
        self.calls = 0

    def review(self, _messages, payload) -> ApprovalReview:
        self.calls += 1
        return ApprovalReview(
            decision="approve" if self.approve else "deny",
            reason="测试审查通过" if self.approve else "测试审查拒绝",
            action_id=str(payload.get("action_id") or "approval-test"),
            reviewer_profile="reviewer-test",
        )


class AgentResilienceTests(unittest.TestCase):
    def test_successful_report_save_finishes_without_another_model_request(self) -> None:
        client = _ReportSaveThenNetworkFailureClient()
        provider = LocalToolProvider("core")
        provider.register(Tool(
            name="sys_skill",
            description="test skill gateway",
            parameters={"type": "object", "properties": {}},
            handler=lambda _arguments: json.dumps({
                "ok": True,
                "report_type": "daily",
                "start_date": "2026-07-09",
                "end_date": "2026-07-09",
                "source_coverage": "partial",
                "needs_user_input": False,
                "content_path": "work_reports/daily/2026-07-09.md",
                "verified": True,
            }, ensure_ascii=False),
        ))
        tools = ToolBus()
        tools.add_provider(provider)
        profile = ModelProfile(
            name="report-terminal-test",
            provider="openai-compatible",
            base_url="https://example.invalid/v1",
            model="test-model",
            api_key_env="UNUSED",
        )
        agent = ReActAgent(client=client, profile=profile, tools=tools)  # type: ignore[arg-type]

        session_messages = [{"role": "user", "content": "补写日报"}]
        events = list(agent.iter_message_events(session_messages))

        self.assertEqual(client.calls, 1)
        self.assertFalse(any(event.get("event") == "error" for event in events))
        final = next(event for event in events if event.get("event") == "final")
        self.assertTrue(final["deterministic_tool_final"])
        self.assertIn("已完成并核验保存2026-07-09 日报", final["content"])
        self.assertIn("work_reports/daily/2026-07-09.md", final["content"])
        self.assertEqual(session_messages[-1]["role"], "assistant")

    def _approval_agent(self, *, auto_approve: bool, auto_approvable: bool):
        calls: list[dict] = []

        def shell_handler(arguments: dict) -> str:
            calls.append(dict(arguments))
            if arguments.get("_approval_source") in {"user", "reviewer"}:
                return json.dumps({"ok": True, "status": "executed", "stdout": ""})
            return json.dumps({
                "ok": False,
                "status": "approval_required",
                "risk_category": "MODIFY" if auto_approvable else "NETWORK",
                "reason": "test approval",
                "command": "mkdir output",
                "preview": "mkdir output",
                "auto_approvable": auto_approvable,
                "reviewable_by_model": auto_approvable,
            })

        provider = LocalToolProvider("core")
        provider.register(Tool(
            name="shell_exec",
            description="test shell",
            parameters={"type": "object", "properties": {}},
            handler=shell_handler,
        ))
        tools = ToolBus()
        tools.add_provider(provider)
        profile = ModelProfile(
            name="approval-test",
            provider="openai-compatible",
            base_url="https://example.invalid/v1",
            model="test-model",
            api_key_env="UNUSED",
        )
        reviewer = _ApprovalReviewerStub(approve=auto_approvable)
        return ReActAgent(
            client=_ShellThenFinalClient(),  # type: ignore[arg-type]
            profile=profile,
            tools=tools,
            auto_approve=auto_approve,
            approval_reviewer=reviewer,  # type: ignore[arg-type]
        ), calls, reviewer

    def test_auto_approval_replays_low_risk_command_and_continues_react(self) -> None:
        agent, calls, reviewer = self._approval_agent(auto_approve=True, auto_approvable=True)

        events = list(agent.iter_message_events([{"role": "user", "content": "test"}]))

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1]["_approval_source"], "reviewer")
        self.assertTrue(calls[1]["_approval_action_id"])
        self.assertEqual(reviewer.calls, 1)
        self.assertTrue(any(event.get("title") == "独立审查已批准" for event in events))
        self.assertTrue(any(event.get("event") == "final" and event.get("content") == "done" for event in events))
        self.assertFalse(any(event.get("waiting_approval") for event in events))

    def test_auto_approval_keeps_high_risk_command_waiting(self) -> None:
        agent, calls, reviewer = self._approval_agent(auto_approve=True, auto_approvable=False)

        events = list(agent.iter_message_events([{"role": "user", "content": "test"}]))

        self.assertEqual(len(calls), 1)
        self.assertEqual(reviewer.calls, 0)
        final = next(event for event in events if event.get("event") == "final")
        self.assertTrue(final["waiting_approval"])
        self.assertTrue(final["pending_approval"]["auto_approve"])
        self.assertNotIn("测试审查拒绝", final["content"])

    def test_resuming_one_shell_approval_does_not_grant_later_shell_calls(self) -> None:
        calls: list[dict] = []

        def shell_handler(arguments: dict) -> str:
            calls.append(dict(arguments))
            if arguments.get("command") == "mkdir first" and arguments.get("_approval_source") == "user":
                return json.dumps({"ok": True, "status": "executed"})
            return json.dumps(
                {
                    "ok": False,
                    "status": "approval_required",
                    "risk_category": "MODIFY",
                    "reason": "second action needs its own confirmation",
                    "command": str(arguments.get("command") or ""),
                    "preview": str(arguments.get("command") or ""),
                    "reviewable_by_model": False,
                }
            )

        provider = LocalToolProvider("core")
        provider.register(
            Tool(
                name="shell_exec",
                description="test shell",
                parameters={"type": "object", "properties": {}},
                handler=shell_handler,
            )
        )
        tools = ToolBus()
        tools.add_provider(provider)
        profile = ModelProfile(
            name="approval-batch-test",
            provider="openai-compatible",
            base_url="https://example.invalid/v1",
            model="test-model",
            api_key_env="UNUSED",
        )
        first = NativeToolCall(id="first", name="shell_exec", arguments={"command": "mkdir first"})
        second = NativeToolCall(id="second", name="shell_exec", arguments={"command": "mkdir second"})
        pending = pending_tool_batch_state(
            runtime_messages_before_batch=[{"role": "user", "content": "create both"}],
            assistant_message={"role": "assistant", "content": ""},
            tool_calls=[first, second],
            approval_index=0,
            completed_tool_messages=[],
            step=1,
            profile_name=profile.name,
            model=profile.model,
            max_steps=10,
            system_context="",
            extra_system_context="",
            approval_payload={"action_id": "approval-first"},
        )
        agent = ReActAgent(client=object(), profile=profile, tools=tools)  # type: ignore[arg-type]
        session_messages = [{"role": "user", "content": "create both"}]

        events = list(agent.iter_approved_tool_batch_events(session_messages, pending))

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["command"], "mkdir first")
        self.assertEqual(calls[0]["_approval_source"], "user")
        self.assertEqual(calls[1]["command"], "mkdir second")
        self.assertNotIn("_approval_source", calls[1])
        final = next(event for event in events if event.get("event") == "final")
        self.assertTrue(final["waiting_approval"])
        next_pending = final["pending_approval"]
        self.assertEqual(next_pending["approval_index"], 1)
        self.assertEqual(len(next_pending["completed_tool_messages"]), 1)

    def test_tool_failures_are_detected_for_prominent_activity_display(self) -> None:
        self.assertTrue(tool_observation_failed("TOOL_ERROR: connection failed"))
        self.assertTrue(tool_observation_failed('{"ok": false, "error": "bad input"}'))
        self.assertTrue(tool_observation_failed('{"status": "failed"}'))
        self.assertFalse(tool_observation_failed('{"ok": true, "result": "done"}'))
        self.assertFalse(tool_observation_failed("ordinary tool output"))

    def test_model_that_never_starts_streaming_hits_start_timeout(self) -> None:
        profile = ModelProfile(
            name="start-timeout-test",
            provider="openai-compatible",
            base_url="https://example.invalid/v1",
            model="test-model",
            api_key_env="UNUSED",
            timeout_seconds=1,
        )
        agent = ReActAgent(
            client=_NoStreamClient(),  # type: ignore[arg-type]
            profile=profile,
            tools=ToolBus(),
        )

        events = list(agent.iter_message_events([{"role": "user", "content": "test"}]))
        progress = "\n".join(
            str(event.get("content") or "")
            for event in events
            if event.get("event") == "activity_delta"
        )

        self.assertIn("没有开始返回有效流", progress)
        self.assertTrue(any(event.get("event") == "error" for event in events))

    def test_stalled_model_stream_times_out_and_keeps_compact_progress(self) -> None:
        profile = ModelProfile(
            name="deadline-test",
            provider="openai-compatible",
            base_url="https://example.invalid/v1",
            model="test-model",
            api_key_env="UNUSED",
            timeout_seconds=1,
        )
        agent = ReActAgent(
            client=_StalledClient(),  # type: ignore[arg-type]
            profile=profile,
            tools=ToolBus(),
        )

        started_at = time.monotonic()
        events = list(agent.iter_message_events([{"role": "user", "content": "test"}]))
        elapsed = time.monotonic() - started_at

        self.assertLess(elapsed, 1.7)
        self.assertTrue(any(event.get("event") == "error" for event in events))
        progress_updates = [event for event in events if event.get("event") == "activity_delta"]
        self.assertLessEqual(len(progress_updates), 2)
        self.assertTrue(any("没有新内容" in str(event.get("content") or "") for event in progress_updates))

    def test_active_reasoning_stream_can_finish_past_profile_timeout(self) -> None:
        profile = ModelProfile(
            name="active-stream-test",
            provider="openai-compatible",
            base_url="https://example.invalid/v1",
            model="test-model",
            api_key_env="UNUSED",
            timeout_seconds=1,
        )
        agent = ReActAgent(
            client=_ActiveReasoningClient(),  # type: ignore[arg-type]
            profile=profile,
            tools=ToolBus(),
        )

        started_at = time.monotonic()
        events = list(agent.iter_message_events([{"role": "user", "content": "test"}]))
        elapsed = time.monotonic() - started_at

        self.assertGreater(elapsed, profile.timeout_seconds)
        self.assertTrue(any(event.get("event") == "final" and event.get("content") == "done" for event in events))
        self.assertFalse(any(event.get("event") == "error" for event in events))

    def test_reasoning_is_streamed_in_activity_without_debug_request_noise(self) -> None:
        profile = ModelProfile(
            name="visible-reasoning-test",
            provider="openai-compatible",
            base_url="https://example.invalid/v1",
            model="test-model",
            api_key_env="UNUSED",
            timeout_seconds=2,
        )
        agent = ReActAgent(
            client=_StreamingFinalClient(),  # type: ignore[arg-type]
            profile=profile,
            tools=ToolBus(),
        )

        events = list(agent.iter_message_events([{"role": "user", "content": "test"}]))
        model_events = [
            event
            for event in events
            if str(event.get("id") or "").startswith("model-plan-")
        ]

        self.assertTrue(model_events)
        self.assertTrue(any("模型思考" in str(event.get("title") or "") for event in model_events))
        self.assertTrue(any("internal" in str(event.get("content") or "") for event in model_events))
        self.assertFalse(any(event.get("activity_type") == "command" for event in model_events))
        self.assertFalse(any("LLM tool planning" in str(event) for event in model_events))
        self.assertFalse(any("--stream-idle-timeout" in str(event) for event in model_events))

    def test_recovery_phase_has_separate_deadline_after_active_reasoning(self) -> None:
        profile = ModelProfile(
            name="recovery-phase-test",
            provider="openai-compatible",
            base_url="https://example.invalid/v1",
            model="test-model",
            api_key_env="UNUSED",
            timeout_seconds=1,
        )
        agent = ReActAgent(
            client=_RecoveryAfterActiveReasoningClient(),  # type: ignore[arg-type]
            profile=profile,
            tools=ToolBus(),
        )

        events = list(agent.iter_message_events([{"role": "user", "content": "test"}]))
        progress = "\n".join(
            str(event.get("content") or "")
            for event in events
            if event.get("event") == "activity_delta"
        )

        self.assertIn("正在启动流式恢复", progress)
        self.assertTrue(
            any(event.get("event") == "final" and event.get("content") == "recovered" for event in events)
        )
        self.assertFalse(any(event.get("event") == "error" for event in events))

    def test_recovery_resets_partial_primary_draft_before_streaming_full_answer(self) -> None:
        profile = ModelProfile(
            name="recovery-draft-test",
            provider="openai-compatible",
            base_url="https://example.invalid/v1",
            model="test-model",
            api_key_env="UNUSED",
        )
        agent = ReActAgent(
            client=_PartialContentThenRecoveryClient(),  # type: ignore[arg-type]
            profile=profile,
            tools=ToolBus(),
        )

        visible = ""
        events = list(agent.iter_message_events([{"role": "user", "content": "test"}]))
        for event in events:
            if event.get("event") == "draft_reset":
                visible = ""
            elif event.get("event") == "draft_delta":
                visible += str(event.get("content") or "")

        self.assertEqual(visible, "流式状态机验证通过")
        self.assertTrue(any(event.get("event") == "draft_reset" for event in events))
        self.assertTrue(
            any(
                event.get("event") == "final"
                and event.get("content") == "流式状态机验证通过"
                for event in events
            )
        )

    def test_stream_timeout_cancels_worker_request(self) -> None:
        profile = ModelProfile(
            name="cancel-stream-test",
            provider="openai-compatible",
            base_url="https://example.invalid/v1",
            model="test-model",
            api_key_env="UNUSED",
            timeout_seconds=1,
        )
        client = _CancellableStalledClient()
        agent = ReActAgent(
            client=client,  # type: ignore[arg-type]
            profile=profile,
            tools=ToolBus(),
        )

        events = list(agent.iter_message_events([{"role": "user", "content": "test"}]))

        self.assertTrue(client.cancelled.wait(0.5))
        self.assertTrue(any(event.get("event") == "error" for event in events))

    def test_stream_tool_activity_reuses_one_defined_id(self) -> None:
        profile = ModelProfile(
            name="tool-activity-test",
            provider="openai-compatible",
            base_url="https://example.invalid/v1",
            model="test-model",
            api_key_env="UNUSED",
        )
        provider = LocalToolProvider("core")
        provider.register(
            Tool(
                name="test_tool",
                description="test",
                parameters={"type": "object", "properties": {}},
                handler=lambda arguments: str(arguments["value"]),
            )
        )
        tools = ToolBus()
        tools.add_provider(provider)
        agent = ReActAgent(
            client=_ToolThenFinalClient(),  # type: ignore[arg-type]
            profile=profile,
            tools=tools,
        )

        events = list(agent.iter_message_events([{"role": "user", "content": "test"}]))

        action = next(event for event in events if event.get("title") == "执行工具：test_tool")
        observation = next(event for event in events if event.get("title") == "test_tool 返回结果")
        self.assertEqual(action["id"], "tool-1-0-call_test")
        self.assertEqual(observation["id"], action["id"])
        self.assertEqual(observation["event"], "activity_delta")
        self.assertEqual(observation["command_status"], "success")
        self.assertEqual(observation["input_summary"], '{"value": "ok"}')
        self.assertEqual(observation["result_summary"], "ok")
        self.assertTrue(any(event.get("event") == "final" and event.get("content") == "done" for event in events))

    def test_visible_model_content_streams_into_answer_draft_without_duplication(self) -> None:
        profile = ModelProfile(
            name="streaming-final-test",
            provider="openai-compatible",
            base_url="https://example.invalid/v1",
            model="test-model",
            api_key_env="UNUSED",
        )
        agent = ReActAgent(
            client=_StreamingFinalClient(),  # type: ignore[arg-type]
            profile=profile,
            tools=ToolBus(),
        )

        events = list(agent.iter_message_events([{"role": "user", "content": "test"}]))
        draft = "".join(
            str(event.get("content") or "")
            for event in events
            if event.get("event") == "draft_delta"
        )

        self.assertEqual(draft, "你好，世界")
        self.assertTrue(any(event.get("event") == "final" and event.get("content") == draft for event in events))


if __name__ == "__main__":
    unittest.main()
