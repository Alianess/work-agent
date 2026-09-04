from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import select
import socket
import socketserver
import threading


DEFAULT_ALLOWED_DOMAINS = frozenset(
    {"pypi.org", "files.pythonhosted.org", "registry.npmjs.org"}
)


def _allowed_host(host: str, allowed_domains: frozenset[str]) -> bool:
    normalized = host.rstrip(".").lower()
    return normalized in allowed_domains


def _connect_public(host: str, port: int, timeout: float = 10) -> socket.socket:
    last_error: OSError | None = None
    for family, socktype, proto, _, address in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM):
        ip = ipaddress.ip_address(address[0])
        if any((ip.is_private, ip.is_loopback, ip.is_link_local, ip.is_multicast, ip.is_reserved, ip.is_unspecified)):
            continue
        candidate = socket.socket(family, socktype, proto)
        candidate.settimeout(timeout)
        try:
            candidate.connect(address)
            return candidate
        except OSError as error:
            last_error = error
            candidate.close()
    raise last_error or OSError("dependency registry did not resolve to a public address")


class _ProxyHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        broker: "NetworkBroker" = self.server.broker  # type: ignore[attr-defined]
        self.request.settimeout(10)
        buffer = b""
        while b"\r\n\r\n" not in buffer and len(buffer) <= 64 * 1024:
            chunk = self.request.recv(4096)
            if not chunk:
                return
            buffer += chunk
        if b"\r\n\r\n" not in buffer:
            self._error(400, "request headers too large")
            return
        head, _ = buffer.split(b"\r\n\r\n", 1)
        lines = head.decode("latin-1").split("\r\n")
        try:
            method, target, _ = lines[0].split(" ", 2)
        except ValueError:
            self._error(400, "invalid proxy request")
            return
        if method.upper() != "CONNECT":
            self._error(403, "only HTTPS dependency registry tunnels are allowed")
            return
        host, _, port_text = target.partition(":")
        try:
            port = int(port_text or "443")
        except ValueError:
            self._error(400, "invalid destination port")
            return
        if port != 443 or not _allowed_host(host, broker.allowed_domains):
            self._error(403, "destination is outside the dependency allowlist")
            return
        self._tunnel(host, port)

    def _tunnel(self, host: str, port: int) -> None:
        broker: "NetworkBroker" = self.server.broker  # type: ignore[attr-defined]
        try:
            upstream = _connect_public(host, port, timeout=10)
        except OSError:
            self._error(502, "upstream unavailable")
            return
        with upstream:
            self.request.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            sockets = [self.request, upstream]
            bytes_in = 0
            bytes_out = 0
            while True:
                readable, _, _ = select.select(sockets, [], [], 30)
                if not readable:
                    return
                for source in readable:
                    data = source.recv(64 * 1024)
                    if not data:
                        return
                    if source is self.request:
                        bytes_out += len(data)
                        if bytes_out > broker.max_bytes_out:
                            return
                    else:
                        bytes_in += len(data)
                        if bytes_in > broker.max_bytes_in:
                            return
                    (upstream if source is self.request else self.request).sendall(data)

    def _error(self, status: int, message: str) -> None:
        body = message.encode("utf-8")
        try:
            self.request.sendall(
                f"HTTP/1.1 {status} Error\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode("latin-1")
                + body
            )
        except OSError:
            return


class _ThreadingProxyServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = False
    daemon_threads = True


@dataclass
class NetworkBroker:
    allowed_domains: frozenset[str] = DEFAULT_ALLOWED_DOMAINS
    max_bytes_in: int = 512 * 1024 * 1024
    max_bytes_out: int = 64 * 1024 * 1024

    def __post_init__(self) -> None:
        self._server: _ThreadingProxyServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> int:
        if self._server is not None:
            return int(self._server.server_address[1])
        self._server = _ThreadingProxyServer(("127.0.0.1", 0), _ProxyHandler)
        self._server.broker = self  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._server.serve_forever, name="work-agent-network-broker", daemon=True)
        self._thread.start()
        return int(self._server.server_address[1])

    def close(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._server = None
        self._thread = None
