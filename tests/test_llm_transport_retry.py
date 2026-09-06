"""连不上端点时，重试同一个端点；重试耗尽后走 profile 显式声明的备用链。

换端点等于换模型：同一份材料换个 endpoint 写出来不是同一份东西。所以传输
故障先做同端点退避重试；链上的备用 profile 必须由用户在配置里逐个声明，
且每次切换都通过 fallback_started 状态显式宣布，绝不静默换端点。
"""

from __future__ import annotations

import http.client
import json
import os
import socket
import time
import unittest
import urllib.error
from unittest import mock

import work_agent_core.llm as llm_module
from work_agent_core.llm import (
    ModelProfile,
    OpenAICompatibleClient,
    RATE_LIMIT_BACKOFF_SECONDS,
    TRANSPORT_RETRIES,
    TRANSPORT_RETRY_BACKOFF_SECONDS,
    is_transport_failure,
    retryable_status,
)


class _StreamingResponse:
    def __init__(self, lines: list[bytes]) -> None:
        self.lines = lines

    def __enter__(self) -> "_StreamingResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def __iter__(self):
        return iter(self.lines)


class TransportFailureClassificationTests(unittest.TestCase):
    def test_dns_and_connection_failures_are_transport_failures(self) -> None:
        self.assertTrue(is_transport_failure(socket.gaierror(8, "nodename nor servname")))
        self.assertTrue(is_transport_failure(urllib.error.URLError(socket.gaierror(8, "x"))))
        self.assertTrue(is_transport_failure(ConnectionResetError()))
        self.assertTrue(is_transport_failure(TimeoutError()))

    def test_rate_limits_and_server_errors_are_retried(self) -> None:
        """429 的报文里写的就是 Please try again later。

        把它当永久失败，等于把一次可恢复的限流变成一轮彻底失败——实测一次会话
        6 轮全废，其中 2 轮就是 429。
        """

        for status in (429, 500, 502, 503, 504):
            self.assertTrue(
                is_transport_failure(urllib.error.HTTPError("u", status, "x", None, None)),
                status,
            )

    def test_a_refusal_is_not_retried(self) -> None:
        # 401/400 再试一百次也一样：服务端答了，而且答的是"不行"。
        for status in (400, 401, 403, 404):
            self.assertFalse(
                is_transport_failure(urllib.error.HTTPError("u", status, "x", None, None)),
                status,
            )
        self.assertFalse(is_transport_failure(ValueError("bad json")))
        self.assertFalse(is_transport_failure(None))

    def test_a_status_wrapped_in_a_message_is_still_found(self) -> None:
        # 流式路径把状态码包进了 RuntimeError 的文本里
        self.assertEqual(
            retryable_status(RuntimeError("LLM stream failed with HTTP 429: {...}")), 429
        )

    def test_rate_limits_wait_for_a_quota_window_not_a_network_blip(self) -> None:
        self.assertGreaterEqual(RATE_LIMIT_BACKOFF_SECONDS[0], 5.0)
        self.assertLess(TRANSPORT_RETRY_BACKOFF_SECONDS[0], RATE_LIMIT_BACKOFF_SECONDS[0])

    def test_a_wrapped_cause_is_still_found(self) -> None:
        try:
            try:
                raise socket.gaierror(8, "nodename nor servname")
            except OSError as inner:
                raise RuntimeError("stream failed") from inner
        except RuntimeError as outer:
            self.assertTrue(is_transport_failure(outer))

    def test_retry_budget_is_bounded(self) -> None:
        # An unreachable host must not be retried forever.
        self.assertEqual(TRANSPORT_RETRIES, 3)

    def test_remote_disconnect_before_headers_enters_same_endpoint_recovery(self) -> None:
        profile = ModelProfile(
            name="remote-disconnect-test",
            provider="openai-compatible",
            base_url="https://example.invalid/v1",
            model="test-model",
            api_key_env="UNUSED",
            timeout_seconds=10,
        )
        success = _StreamingResponse([
            b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n',
            b"data: [DONE]\n",
        ])
        statuses: list[str] = []

        with (
            mock.patch.dict(os.environ, {"UNUSED": "test-key"}),
            mock.patch(
                "work_agent_core.llm.urllib.request.urlopen",
                side_effect=[http.client.RemoteDisconnected("closed"), success],
            ) as urlopen,
        ):
            response = OpenAICompatibleClient().chat_tools_stream(
                [{"role": "user", "content": "reply"}],
                profile=profile,
                on_delta=lambda chunk: statuses.append(str(chunk.status or "")),
            )

        self.assertEqual(response.content, "ok")
        self.assertEqual(urlopen.call_count, 2)
        self.assertIn("recovery_started", statuses)
        recovery_payload = json.loads(urlopen.call_args_list[1].args[0].data.decode("utf-8"))
        self.assertIn("恢复阶段", recovery_payload["messages"][0]["content"])


class RecoveryRetryBudgetTests(unittest.TestCase):
    """恢复请求的重试预算，属于恢复请求自己。

    实测踩过的洞：主流"返回了但没正文或工具调用"时 cause 为空，于是恢复请求
    只被允许发一次；正好撞上 429，一轮就彻底判死，而 429 是最该退避重试的那类。
    """

    def _profile(self) -> ModelProfile:
        return ModelProfile(
            name="recovery-budget-test",
            provider="openai-compatible",
            base_url="https://example.invalid/v1",
            model="test-model",
            api_key_env="UNUSED",
            timeout_seconds=10,
        )

    def _run_recovery(self, errors: list[Exception]) -> tuple[int, Exception | None]:
        client = OpenAICompatibleClient()
        attempts = {"n": 0}

        def fake_stream(*_args: object, **_kwargs: object) -> None:
            index = attempts["n"]
            attempts["n"] += 1
            raise errors[min(index, len(errors) - 1)]

        client.chat_tools_stream = fake_stream  # type: ignore[method-assign]
        failure: Exception | None = None
        with mock.patch.object(llm_module.time, "sleep"):
            try:
                client._recover_tools_response(
                    [{"role": "user", "content": "hi"}],
                    profile=self._profile(),
                    temperature=None,
                    max_tokens=None,
                    tools=None,
                    tool_choice=None,
                    reasoning_effort=None,
                    started_at=time.monotonic(),
                    reason="empty",
                    cause=None,
                    on_delta=None,
                    cancel_event=None,
                )
            except Exception as error:  # noqa: BLE001 - 这里就是要看它抛什么
                failure = error
        return attempts["n"], failure

    def test_an_empty_primary_still_buys_the_recovery_a_full_retry_budget(self) -> None:
        rate_limited = urllib.error.HTTPError("u", 429, "Too Many Requests", None, None)
        attempts, failure = self._run_recovery([rate_limited])

        self.assertEqual(attempts, TRANSPORT_RETRIES + 1)
        self.assertIsNotNone(failure)

    def test_a_refusal_during_recovery_is_not_retried(self) -> None:
        # 401 再试一百次也一样，退避只会把失败拖慢。
        attempts, failure = self._run_recovery(
            [urllib.error.HTTPError("u", 401, "Unauthorized", None, None)]
        )

        self.assertEqual(attempts, 1)
        self.assertIsNotNone(failure)

    def test_quota_exhaustion_is_reported_as_quota_not_as_raw_json(self) -> None:
        """限流和额度用尽都走 429，但对用户是两件事。

        一个等一会儿就好，一个要去后台提额——把整段 JSON 糊上去，用户两件事
        都分不出来。
        """

        exhausted = RuntimeError(
            'LLM stream failed with HTTP 429: {"error":{"code":"insufficient_quota",'
            '"message":"Workspace allocated quota exceeded"}}'
        )
        _attempts, failure = self._run_recovery([exhausted])

        self.assertIsNotNone(failure)
        message = str(failure)
        self.assertIn("额度已用尽", message)
        self.assertNotIn("insufficient_quota", message)

    def test_plain_rate_limiting_tells_the_user_to_wait_not_to_top_up(self) -> None:
        _attempts, failure = self._run_recovery(
            [RuntimeError("LLM stream failed with HTTP 429: rate limit reached")]
        )

        self.assertIsNotNone(failure)
        self.assertIn("正在限流", str(failure))
        self.assertNotIn("提额", str(failure))


class _FakeFallbackClient:
    """备用链包装层的假客户端；真客户端内部的退避重试由既有测试覆盖。"""

    def __init__(self) -> None:
        self.chat_plan: list[str] = []
        self.stream_plan: list[str] = []
        self.chat_calls: list[str] = []
        self.stream_calls: list[str] = []

    def chat(self, _messages, *, profile, **_kwargs):
        self.chat_calls.append(profile.name)
        action = self.chat_plan.pop(0)
        if action == "refuse":
            raise ConnectionError(61, "Connection refused")
        if action == "reject":
            raise RuntimeError("LLM request failed with HTTP 400: bad request")
        return llm_module.LLMResponse(content="ok", raw={})

    def chat_tools_stream(self, _messages, *, profile, on_delta=None, **_kwargs):
        self.stream_calls.append(profile.name)
        action = self.stream_plan.pop(0)
        if action == "refuse":
            raise ConnectionError(61, "Connection refused")
        if action == "reject":
            raise RuntimeError("LLM request failed with HTTP 400: bad request")
        if action == "emit_then_refuse":
            if on_delta is not None:
                on_delta(llm_module.LLMStreamChunk(content="先说了半句"))
            raise ConnectionError(61, "Connection refused")
        return llm_module.LLMResponse(content="ok", raw={})


class FallbackChainTests(unittest.TestCase):
    def _profiles(self) -> tuple[ModelProfile, ModelProfile]:
        primary = ModelProfile(
            name="primary",
            provider="openai-compatible",
            base_url="https://primary.invalid/v1",
            model="m",
            api_key_env="UNUSED",
            fallback_profiles=("backup",),
        )
        backup = ModelProfile(
            name="backup",
            provider="openai-compatible",
            base_url="https://backup.invalid/v1",
            model="m2",
            api_key_env="UNUSED",
        )
        return primary, backup

    def test_stream_fallback_engages_when_no_model_output_yet(self) -> None:
        client = _FakeFallbackClient()
        client.stream_plan = ["refuse", "ok"]
        seen: list[llm_module.LLMStreamChunk] = []
        primary, backup = self._profiles()

        _response, used = llm_module.chat_tools_stream_with_fallback(
            client,
            [],
            profile=primary,
            fallback_profiles=(backup,),
            on_delta=seen.append,
        )

        self.assertEqual(used.name, "backup")
        self.assertEqual(client.stream_calls, ["primary", "backup"])
        self.assertTrue(any(chunk.status == "fallback_started" for chunk in seen))

    def test_stream_fallback_never_switches_after_model_output_started(self) -> None:
        client = _FakeFallbackClient()
        client.stream_plan = ["emit_then_refuse"]
        primary, backup = self._profiles()

        with self.assertRaises(ConnectionError):
            llm_module.chat_tools_stream_with_fallback(
                client, [], profile=primary, fallback_profiles=(backup,)
            )

        self.assertEqual(client.stream_calls, ["primary"])

    def test_non_transport_failure_does_not_switch_profile(self) -> None:
        client = _FakeFallbackClient()
        client.stream_plan = ["reject"]
        primary, backup = self._profiles()

        with self.assertRaises(RuntimeError):
            llm_module.chat_tools_stream_with_fallback(
                client, [], profile=primary, fallback_profiles=(backup,)
            )

        self.assertEqual(client.stream_calls, ["primary"])

    def test_exhausted_chain_raises_the_last_error(self) -> None:
        client = _FakeFallbackClient()
        client.stream_plan = ["refuse", "refuse"]
        primary, backup = self._profiles()

        with self.assertRaises(ConnectionError):
            llm_module.chat_tools_stream_with_fallback(
                client, [], profile=primary, fallback_profiles=(backup,)
            )

        self.assertEqual(client.stream_calls, ["primary", "backup"])

    def test_nonstream_chat_fallback_walks_the_declared_chain(self) -> None:
        client = _FakeFallbackClient()
        client.chat_plan = ["refuse", "ok"]
        primary, backup = self._profiles()

        response, used = llm_module.chat_with_fallback(
            client, [], profile=primary, fallback_profiles=(backup,)
        )

        self.assertEqual(response.content, "ok")
        self.assertEqual(used.name, "backup")
        self.assertEqual(client.chat_calls, ["primary", "backup"])

    def test_registry_resolves_chain_skipping_self_missing_and_cycles(self) -> None:
        from work_agent_core.config import ModelRegistry

        primary, backup = self._profiles()
        loop = ModelProfile(
            name="loop",
            provider="openai-compatible",
            base_url="https://loop.invalid/v1",
            model="m3",
            api_key_env="UNUSED",
            fallback_profiles=("primary", "missing"),
        )
        # 真环：a -> b -> a。b 的备用 a 已在链上，跳过，不能死循环。
        profile_a = ModelProfile(
            name="cycle-a",
            provider="openai-compatible",
            base_url="https://a.invalid/v1",
            model="ma",
            api_key_env="UNUSED",
            fallback_profiles=("cycle-b",),
        )
        profile_b = ModelProfile(
            name="cycle-b",
            provider="openai-compatible",
            base_url="https://b.invalid/v1",
            model="mb",
            api_key_env="UNUSED",
            fallback_profiles=("cycle-a",),
        )

        registry = ModelRegistry(
            {
                "primary": primary,
                "backup": backup,
                "loop": loop,
                "cycle-a": profile_a,
                "cycle-b": profile_b,
            },
            "primary",
        )
        # loop -> primary -> backup 是合法的传递链；missing 缺失，跳过。
        self.assertEqual(
            [p.name for p in registry.fallback_chain(loop)],
            ["primary", "backup"],
        )
        self.assertEqual([p.name for p in registry.fallback_chain(profile_a)], ["cycle-b"])


if __name__ == "__main__":
    unittest.main()
