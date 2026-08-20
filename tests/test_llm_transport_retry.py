"""连不上端点时，重试同一个端点——绝不换端点。

换端点等于悄悄换模型：同一份材料换个 endpoint 写出来不是同一份东西，
而用户不会知道换过。所以传输故障只做同端点退避重试。
"""

from __future__ import annotations

import socket
import unittest
import urllib.error

from work_agent_core.llm import (
    TRANSPORT_RETRIES,
    is_transport_failure,
)


class TransportFailureClassificationTests(unittest.TestCase):
    def test_dns_and_connection_failures_are_transport_failures(self) -> None:
        self.assertTrue(is_transport_failure(socket.gaierror(8, "nodename nor servname")))
        self.assertTrue(is_transport_failure(urllib.error.URLError(socket.gaierror(8, "x"))))
        self.assertTrue(is_transport_failure(ConnectionResetError()))
        self.assertTrue(is_transport_failure(TimeoutError()))

    def test_endpoint_answered_is_not_a_transport_failure(self) -> None:
        # A 500 means the host was reached; retrying the socket is not the fix.
        self.assertFalse(
            is_transport_failure(urllib.error.HTTPError("u", 500, "boom", None, None))
        )
        self.assertFalse(is_transport_failure(ValueError("bad json")))
        self.assertFalse(is_transport_failure(None))

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
