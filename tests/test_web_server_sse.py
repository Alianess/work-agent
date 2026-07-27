from __future__ import annotations

import unittest
from types import MethodType, SimpleNamespace

from work_agent_core.web_server import WorkAgentHandler


class _DisconnectingWriter:
    def write(self, _data: bytes) -> None:
        raise BrokenPipeError("client closed connection")

    def flush(self) -> None:
        return None


class ServerSentEventTests(unittest.TestCase):
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
