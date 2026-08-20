"""连不上端点时，重试同一个端点——绝不换端点。

换端点等于悄悄换模型：同一份材料换个 endpoint 写出来不是同一份东西，
而用户不会知道换过。所以传输故障只做同端点退避重试。
"""

from __future__ import annotations

import socket
import unittest
import urllib.error

from work_agent_core.llm import (
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


if __name__ == "__main__":
    unittest.main()
