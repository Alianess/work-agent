from __future__ import annotations

import socket
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from work_agent_core.execution.network_broker import NetworkBroker, _ProxyHandler, _allowed_host, _connect_public


class NetworkBrokerTests(unittest.TestCase):
    def test_allowlist_is_exact_not_suffix_permissive(self) -> None:
        allowed = frozenset({"pypi.org"})
        self.assertTrue(_allowed_host("pypi.org", allowed))
        self.assertFalse(_allowed_host("files.pypi.org", allowed))
        self.assertFalse(_allowed_host("pypi.org.evil.test", allowed))

    def test_handler_rejects_destination_outside_allowlist(self) -> None:
        broker = NetworkBroker(allowed_domains=frozenset({"pypi.org"}))
        client, server_side = socket.socketpair()
        server = SimpleNamespace(broker=broker)
        thread = threading.Thread(
            target=_ProxyHandler,
            args=(server_side, ("local", 0), server),
            daemon=True,
        )
        thread.start()
        try:
            client.sendall(b"CONNECT example.com:443 HTTP/1.1\r\n\r\n")
            self.assertIn(b"403", client.recv(256))
        finally:
            client.close()
            server_side.close()
            thread.join(timeout=1)

    def test_private_dns_resolution_is_rejected(self) -> None:
        answer = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))]
        with patch("socket.getaddrinfo", return_value=answer):
            with self.assertRaises(OSError):
                _connect_public("pypi.org", 443)


if __name__ == "__main__":
    unittest.main()
