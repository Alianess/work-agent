"""连不上端点时，重试同一个端点——绝不换端点。

换端点等于悄悄换模型：同一份材料换个 endpoint 写出来不是同一份东西，
而用户不会知道换过。所以传输故障只做同端点退避重试。
"""

from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
