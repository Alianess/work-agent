from __future__ import annotations

import json
import os
import socket
import threading
import time
import unittest
from unittest.mock import patch

from work_agent_core.config import ModelProfile
from work_agent_core.llm import OpenAICompatibleClient, prepare_messages_for_profile
from work_agent_core.react import assistant_message_for_history
from work_agent_core.session_store import sanitize_runtime_message


class _StreamingResponse:
    def __init__(self, lines: list[bytes]) -> None:
        self.lines = lines

    def __enter__(self) -> "_StreamingResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def __iter__(self):
        return iter(self.lines)


class _JsonResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self) -> "_JsonResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class _TimeoutStreamingResponse(_StreamingResponse):
    def __iter__(self):
        yield from self.lines
        raise socket.timeout("upstream stopped sending chunks")


class _CancellableStreamingResponse(_StreamingResponse):
    def __init__(self) -> None:
        super().__init__([])
        self.started = threading.Event()
        self.closed = threading.Event()

    def __iter__(self):
        self.started.set()
        yield b'data: {"choices":[{"delta":{"reasoning_content":"thinking"}}]}\n'
        self.closed.wait(2)
        yield b"\n"

    def close(self) -> None:
        self.closed.set()


class ReasoningContentHistoryTests(unittest.TestCase):
    def test_cancelling_stream_closes_active_response(self) -> None:
        profile = ModelProfile(
            name="luna-test",
            provider="openai-compatible",
            base_url="https://example.invalid/v1",
            model="gpt-5.6-luna",
            api_key_env="WORK_AGENT_REASONING_TEST_KEY",
        )
        stream = _CancellableStreamingResponse()
        cancel_event = threading.Event()
        errors: list[Exception] = []
        previous_key = os.environ.get(profile.api_key_env)
        os.environ[profile.api_key_env] = "test-key"

        def run_request() -> None:
            try:
                OpenAICompatibleClient().chat_tools_stream(
                    [{"role": "user", "content": "reply"}],
                    profile=profile,
                    cancel_event=cancel_event,
                )
            except Exception as error:
                errors.append(error)

        try:
            with patch(
                "work_agent_core.llm.urllib.request.urlopen",
                return_value=stream,
            ):
                thread = threading.Thread(target=run_request)
                thread.start()
                self.assertTrue(stream.started.wait(0.5))
                cancel_event.set()
                thread.join(1)
        finally:
            if previous_key is None:
                os.environ.pop(profile.api_key_env, None)
            else:
                os.environ[profile.api_key_env] = previous_key

        self.assertFalse(thread.is_alive())
        self.assertTrue(stream.closed.is_set())
        self.assertTrue(any("已取消" in str(error) for error in errors))

    def test_stream_recovers_final_content_from_full_message_frame(self) -> None:
        profile = ModelProfile(
            name="luna-test",
            provider="openai-compatible",
            base_url="https://example.invalid/v1",
            model="gpt-5.6-luna",
            api_key_env="WORK_AGENT_REASONING_TEST_KEY",
        )
        lines = [
            b'data: {"choices":[{"delta":{"reasoning_content":"thinking"}}]}\n',
            b'data: {"choices":[{"message":{"role":"assistant","content":"final answer"},"finish_reason":"stop"}]}\n',
            b"data: [DONE]\n",
        ]
        previous_key = os.environ.get(profile.api_key_env)
        os.environ[profile.api_key_env] = "test-key"
        try:
            with patch(
                "work_agent_core.llm.urllib.request.urlopen",
                return_value=_StreamingResponse(lines),
            ):
                response = OpenAICompatibleClient().chat_tools_stream(
                    [{"role": "user", "content": "reply"}],
                    profile=profile,
                )
        finally:
            if previous_key is None:
                os.environ.pop(profile.api_key_env, None)
            else:
                os.environ[profile.api_key_env] = previous_key

        self.assertEqual(response.content, "final answer")

    def test_reasoning_only_stream_falls_back_to_streaming_response(self) -> None:
        profile = ModelProfile(
            name="luna-test",
            provider="openai-compatible",
            base_url="https://example.invalid/v1",
            model="gpt-5.6-luna",
            api_key_env="WORK_AGENT_REASONING_TEST_KEY",
        )
        stream = _StreamingResponse(
            [
                b'data: {"choices":[{"delta":{"reasoning_content":"thinking"}}]}\n',
                b"data: [DONE]\n",
            ]
        )
        fallback = _StreamingResponse(
            [
                b'data: {"choices":[{"delta":{"content":"recovered"},"finish_reason":"stop"}]}\n',
                b"data: [DONE]\n",
            ]
        )
        previous_key = os.environ.get(profile.api_key_env)
        os.environ[profile.api_key_env] = "test-key"
        try:
            with patch(
                "work_agent_core.llm.urllib.request.urlopen",
                side_effect=[stream, fallback],
            ):
                response = OpenAICompatibleClient().chat_tools_stream(
                    [{"role": "user", "content": "reply"}],
                    profile=profile,
                )
        finally:
            if previous_key is None:
                os.environ.pop(profile.api_key_env, None)
            else:
                os.environ[profile.api_key_env] = previous_key

        self.assertEqual(response.content, "recovered")

    def test_stalled_stream_recovers_once_with_bounded_reasoning(self) -> None:
        profile = ModelProfile(
            name="luna-test",
            provider="openai-compatible",
            base_url="https://example.invalid/v1",
            model="gpt-5.6-luna",
            api_key_env="WORK_AGENT_REASONING_TEST_KEY",
            timeout_seconds=120,
        )
        stream = _TimeoutStreamingResponse(
            [b'data: {"choices":[{"delta":{"reasoning_content":"thinking"}}]}\n']
        )
        fallback = _StreamingResponse(
            [
                b'data: {"choices":[{"delta":{"content":"recovered"},"finish_reason":"stop"}]}\n',
                b"data: [DONE]\n",
            ]
        )
        previous_key = os.environ.get(profile.api_key_env)
        os.environ[profile.api_key_env] = "test-key"
        try:
            with patch(
                "work_agent_core.llm.urllib.request.urlopen",
                side_effect=[stream, fallback],
            ) as urlopen:
                response = OpenAICompatibleClient().chat_tools_stream(
                    [{"role": "user", "content": "reply"}],
                    profile=profile,
                    reasoning_effort="high",
                )
        finally:
            if previous_key is None:
                os.environ.pop(profile.api_key_env, None)
            else:
                os.environ[profile.api_key_env] = previous_key

        self.assertEqual(response.content, "recovered")
        self.assertEqual(response.raw["_work_agent"]["recovery"]["reasoning_effort"], "light")
        self.assertEqual(urlopen.call_args_list[0].kwargs["timeout"], 45)
        recovery_payload = json.loads(urlopen.call_args_list[1].args[0].data.decode("utf-8"))
        self.assertEqual(recovery_payload["reasoning_effort"], "low")
        self.assertTrue(recovery_payload["stream"])

    def test_recovery_budget_is_independent_after_long_active_stream(self) -> None:
        profile = ModelProfile(
            name="luna-test",
            provider="openai-compatible",
            base_url="https://example.invalid/v1",
            model="gpt-5.6-luna",
            api_key_env="WORK_AGENT_REASONING_TEST_KEY",
            timeout_seconds=120,
        )
        fallback = _StreamingResponse(
            [
                b'data: {"choices":[{"delta":{"content":"recovered"},"finish_reason":"stop"}]}\n',
                b"data: [DONE]\n",
            ]
        )
        statuses: list[str] = []
        previous_key = os.environ.get(profile.api_key_env)
        os.environ[profile.api_key_env] = "test-key"
        try:
            with patch("work_agent_core.llm.urllib.request.urlopen", return_value=fallback) as urlopen:
                response = OpenAICompatibleClient()._recover_tools_response(
                    [{"role": "user", "content": "reply"}],
                    profile=profile,
                    temperature=None,
                    max_tokens=None,
                    tools=None,
                    tool_choice=None,
                    reasoning_effort="high",
                    started_at=time.monotonic() - 198,
                    reason="empty_stream",
                    cause=None,
                    on_delta=lambda chunk: statuses.append(chunk.status),
                    cancel_event=None,
                )
        finally:
            if previous_key is None:
                os.environ.pop(profile.api_key_env, None)
            else:
                os.environ[profile.api_key_env] = previous_key

        self.assertEqual(response.content, "recovered")
        self.assertEqual(statuses, ["recovery_started", "recovery_streaming"])
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 45)
        self.assertGreaterEqual(
            response.raw["_work_agent"]["recovery"]["primary_stream_elapsed_seconds"],
            198,
        )

    def test_empty_stream_and_empty_fallback_raise_clear_error(self) -> None:
        profile = ModelProfile(
            name="luna-test",
            provider="openai-compatible",
            base_url="https://example.invalid/v1",
            model="gpt-5.6-luna",
            api_key_env="WORK_AGENT_REASONING_TEST_KEY",
        )
        stream = _StreamingResponse([b"data: [DONE]\n"])
        fallback = _StreamingResponse(
            [
                b'data: {"choices":[{"delta":{"reasoning_content":"still thinking"}}],"usage":{"completion_tokens":32000,"completion_tokens_details":{"reasoning_tokens":32000}}}\n',
                b'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n',
                b"data: [DONE]\n",
            ]
        )
        previous_key = os.environ.get(profile.api_key_env)
        os.environ[profile.api_key_env] = "test-key"
        try:
            with patch(
                "work_agent_core.llm.urllib.request.urlopen",
                side_effect=[stream, fallback],
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "finish_reason=length.*completion_tokens=32000.*reasoning_tokens=32000",
                ):
                    OpenAICompatibleClient().chat_tools_stream(
                        [{"role": "user", "content": "reply"}],
                        profile=profile,
                    )
        finally:
            if previous_key is None:
                os.environ.pop(profile.api_key_env, None)
            else:
                os.environ[profile.api_key_env] = previous_key

    def test_reasoning_only_primary_and_recovery_do_not_switch_models(self) -> None:
        profile = ModelProfile(
            name="luna-test",
            provider="openai-compatible",
            base_url="https://example.invalid/v1",
            model="gpt-5.6-luna",
            api_key_env="WORK_AGENT_REASONING_TEST_KEY",
        )
        reasoning_only = lambda: _StreamingResponse(
            [
                b'data: {"choices":[{"delta":{"reasoning_content":"thinking"}}]}\n',
                b"data: [DONE]\n",
            ]
        )
        statuses: list[tuple[str, str]] = []
        previous_primary = os.environ.get(profile.api_key_env)
        os.environ[profile.api_key_env] = "test-key"
        try:
            with patch(
                "work_agent_core.llm.urllib.request.urlopen",
                side_effect=[reasoning_only(), reasoning_only()],
            ) as urlopen:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "当前模型流式恢复也失败",
                ):
                    OpenAICompatibleClient().chat_tools_stream(
                        [{"role": "user", "content": "reply"}],
                        profile=profile,
                        on_delta=lambda chunk: statuses.append((chunk.status, chunk.status_detail)),
                    )
        finally:
            if previous_primary is None:
                os.environ.pop(profile.api_key_env, None)
            else:
                os.environ[profile.api_key_env] = previous_primary

        self.assertEqual(urlopen.call_count, 2)
        self.assertIn(("recovery_started", "finish_reason=unknown"), statuses)
        self.assertFalse(any(status.startswith("fallback_") for status, _ in statuses))

    def test_tool_call_reasoning_survives_stream_history_and_sanitizing(self) -> None:
        profile = ModelProfile(
            name="deepseek-test",
            provider="openai-compatible",
            base_url="https://example.invalid/v1",
            model="deepseek-test",
            api_key_env="WORK_AGENT_REASONING_TEST_KEY",
        )
        lines = [
            b'data: {"choices":[{"delta":{"reasoning_content":"check "}}]}\n',
            b'data: {"choices":[{"delta":{"reasoning_content":"files"}}]}\n',
            (
                b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
                b'"id":"call_1","type":"function","function":{"name":"read_text_file",'
                b'"arguments":"{\\"path\\":\\"notes.md\\"}"}}]},"finish_reason":"tool_calls"}]}\n'
            ),
            b"data: [DONE]\n",
        ]
        previous_key = os.environ.get(profile.api_key_env)
        os.environ[profile.api_key_env] = "test-key"
        try:
            with patch(
                "work_agent_core.llm.urllib.request.urlopen",
                return_value=_StreamingResponse(lines),
            ):
                response = OpenAICompatibleClient().chat_tools_stream(
                    [{"role": "user", "content": "read it"}],
                    profile=profile,
                    tools=[{"type": "function", "function": {"name": "read_text_file"}}],
                )
        finally:
            if previous_key is None:
                os.environ.pop(profile.api_key_env, None)
            else:
                os.environ[profile.api_key_env] = previous_key

        message = response.raw["choices"][0]["message"]
        self.assertEqual(message["reasoning_content"], "check files")
        history = assistant_message_for_history(message)
        self.assertEqual(history["reasoning_content"], "check files")
        self.assertEqual(
            sanitize_runtime_message(history)["reasoning_content"],
            "check files",
        )
        self.assertEqual(
            json.loads(history["tool_calls"][0]["function"]["arguments"]),
            {"path": "notes.md"},
        )

    def test_reasoning_without_tool_calls_is_not_persisted(self) -> None:
        clean = sanitize_runtime_message(
            {
                "role": "assistant",
                "content": "done",
                "reasoning_content": "private reasoning",
            }
        )
        self.assertNotIn("reasoning_content", clean)

    def test_legacy_deepseek_tool_call_gets_reasoning_placeholder(self) -> None:
        profile = ModelProfile(
            name="deepseek-v4-flash",
            provider="openai-compatible",
            base_url="https://api.deepseek.com",
            model="deepseek-v4-flash",
            api_key_env="UNUSED",
        )
        original = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read_text_file", "arguments": "{}"},
                    }
                ],
            }
        ]

        prepared = prepare_messages_for_profile(original, profile)

        self.assertIn("reasoning_content", prepared[0])
        self.assertNotIn("reasoning_content", original[0])

    def test_non_deepseek_history_is_left_unchanged(self) -> None:
        profile = ModelProfile(
            name="other",
            provider="openai-compatible",
            base_url="https://example.invalid/v1",
            model="other-model",
            api_key_env="UNUSED",
        )
        messages = [{"role": "assistant", "content": "", "tool_calls": [{}]}]
        self.assertIs(prepare_messages_for_profile(messages, profile), messages)


if __name__ == "__main__":
    unittest.main()
