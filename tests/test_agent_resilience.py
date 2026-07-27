from __future__ import annotations

import time
import threading
import unittest

from work_agent_core.config import ModelProfile
from work_agent_core.llm import LLMResponse, LLMStreamChunk
from work_agent_core.react import ReActAgent
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


class AgentResilienceTests(unittest.TestCase):
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
