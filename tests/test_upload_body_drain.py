"""拒绝一个上传之前，必须先把请求体读掉。

不读就回包，等于在浏览器还在上传时掐断连接。浏览器把它报成 "Failed to fetch"，
前端再翻译成"无法连接本地服务，请确认后端服务正在运行"——于是用户被指去查一个
根本没问题的服务，而真正的原因（文件太大、没登录）永远到不了他眼前。
"""

from __future__ import annotations

import io
import unittest
from types import MethodType, SimpleNamespace

from work_agent_core.web_server import (
    DRAIN_REQUEST_BODY_MAX_BYTES,
    UPLOAD_MAX_FILE_BYTES,
    WorkAgentHandler,
)


def make_handler(*, body: bytes, content_length: str | None = None) -> SimpleNamespace:
    length = str(len(body)) if content_length is None else content_length
    handler = SimpleNamespace(
        rfile=io.BytesIO(body),
        headers={"Content-Length": length},
        close_connection=False,
    )
    handler._drain_request_body = MethodType(WorkAgentHandler._drain_request_body, handler)
    handler._read_binary_body = MethodType(WorkAgentHandler._read_binary_body, handler)
    return handler


class UploadBodyDrainTests(unittest.TestCase):
    def test_the_body_is_consumed_so_the_response_can_reach_the_browser(self) -> None:
        handler = make_handler(body=b"x" * 4096)

        handler._drain_request_body()

        self.assertEqual(handler.rfile.read(), b"")
        self.assertFalse(handler.close_connection)

    def test_a_body_too_big_to_be_worth_reading_closes_the_connection_instead(self) -> None:
        handler = make_handler(
            body=b"", content_length=str(DRAIN_REQUEST_BODY_MAX_BYTES + 1)
        )

        handler._drain_request_body()

        self.assertTrue(handler.close_connection)

    def test_an_oversized_upload_drains_before_raising(self) -> None:
        # 超限时既要报出"单个文件不能超过 N MB"，也不能把连接掐死在半路。
        oversized = str(UPLOAD_MAX_FILE_BYTES + 1)
        handler = make_handler(body=b"y" * 2048, content_length=oversized)
        handler.headers["Content-Length"] = oversized

        with self.assertRaises(ValueError) as caught:
            handler._read_binary_body()

        self.assertIn("MB", str(caught.exception))
        self.assertTrue(handler.close_connection)

    def test_a_client_that_stops_sending_does_not_hang_the_drain(self) -> None:
        # 声称 8192 字节却只发了 100：读到空就停，不能一直等。
        handler = make_handler(body=b"z" * 100, content_length="8192")

        handler._drain_request_body()

        self.assertEqual(handler.rfile.read(), b"")

    def test_a_missing_or_invalid_length_is_not_an_error(self) -> None:
        handler = make_handler(body=b"", content_length="not-a-number")

        handler._drain_request_body()

        self.assertFalse(handler.close_connection)


if __name__ == "__main__":
    unittest.main()
