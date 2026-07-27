from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue, Empty
from typing import Any
import atexit
import json
import os
import re
import shutil
import subprocess
import threading
import time

from .tools import Tool


DEFAULT_MCP_CONFIG_PATH = Path("config/mcp_servers.json")
MCP_PROTOCOL_VERSION = "2024-11-05"
MAX_TOOL_NAME_LENGTH = 64


class MCPProviderError(RuntimeError):
    pass


class MCPRequestError(MCPProviderError):
    pass


@dataclass(frozen=True)
class MCPServerConfig:
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    enabled: bool = True
    transport: str = "stdio"
    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    description: str = ""
    initialize_timeout_seconds: float = 20.0
    list_timeout_seconds: float = 20.0
    tool_timeout_seconds: float = 180.0
    blocked_tools: set[str] = field(default_factory=set)
    skill_id: str = ""

    @classmethod
    def from_mapping(cls, name: str, raw: dict[str, Any]) -> MCPServerConfig:
        return cls(
            name=safe_provider_name(name),
            command=str(raw.get("command") or "").strip(),
            args=[str(item) for item in raw.get("args") or []],
            enabled=bool(raw.get("enabled", True)),
            transport=str(raw.get("type") or raw.get("transport") or "stdio").strip() or "stdio",
            cwd=str(raw.get("cwd")).strip() if raw.get("cwd") else None,
            env={str(k): str(v) for k, v in (raw.get("env") or {}).items()},
            description=str(raw.get("description") or ""),
            initialize_timeout_seconds=float(raw.get("initialize_timeout_seconds") or 20.0),
            list_timeout_seconds=float(raw.get("list_timeout_seconds") or 20.0),
            tool_timeout_seconds=float(raw.get("tool_timeout_seconds") or 180.0),
            blocked_tools={
                str(item).strip().lower()
                for item in (raw.get("blocked_tools") or raw.get("deny_tools") or [])
                if str(item).strip()
            },
            skill_id=str(raw.get("skill_id") or "").strip(),
        )


@dataclass(frozen=True)
class MCPToolEntry:
    exposed_name: str
    original_name: str
    server_name: str
    description: str
    input_schema: dict[str, Any]


class MCPStdioSession:
    """Small synchronous MCP stdio client.

    The project should not depend on individual skill CLIs. This client speaks
    the MCP JSON-RPC boundary directly, so any stdio MCP server can become a
    ToolBus provider without hardcoding project logic.
    """

    def __init__(self, config: MCPServerConfig, *, workspace_root: Path) -> None:
        self.config = config
        self.workspace_root = workspace_root
        self.process: subprocess.Popen[str] | None = None
        self._request_id = 0
        self._send_lock = threading.Lock()
        self._pending: dict[int, Queue[dict[str, Any]]] = {}
        self._pending_lock = threading.Lock()
        self._stderr_tail: deque[str] = deque(maxlen=40)
        self._stdout_tail: deque[str] = deque(maxlen=20)
        self._reader_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self.initialized = False
        self.server_info: dict[str, Any] = {}

    def start(self) -> None:
        if self.is_alive() and self.initialized:
            return
        if self.config.transport != "stdio":
            raise MCPProviderError(
                f"MCP server {self.config.name!r} uses unsupported transport "
                f"{self.config.transport!r}; only stdio is supported in this provider."
            )
        if not self.config.command:
            raise MCPProviderError(f"MCP server {self.config.name!r} has no command.")

        cwd = resolve_cwd(self.config.cwd, self.workspace_root)
        command = shutil.which(self.config.command) or self.config.command
        env = os.environ.copy()
        env.update(expand_env_values(self.config.env))
        env.setdefault("WORK_AGENT_WORKSPACE", str(self.workspace_root))

        self.process = subprocess.Popen(
            [command, *self.config.args],
            cwd=cwd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._reader_thread = threading.Thread(
            target=self._stdout_loop,
            name=f"work-agent-mcp-{self.config.name}-stdout",
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._stderr_loop,
            name=f"work-agent-mcp-{self.config.name}-stderr",
            daemon=True,
        )
        self._reader_thread.start()
        self._stderr_thread.start()

        result = self.request(
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "work-agent", "version": "0.1.0"},
            },
            timeout_seconds=self.config.initialize_timeout_seconds,
        )
        if isinstance(result, dict):
            self.server_info = result.get("serverInfo") or result.get("server_info") or {}
        self.notify("notifications/initialized", {})
        self.initialized = True

    def list_tools(self) -> list[dict[str, Any]]:
        self.start()
        result = self.request("tools/list", {}, timeout_seconds=self.config.list_timeout_seconds)
        if not isinstance(result, dict):
            return []
        tools = result.get("tools") or []
        return [item for item in tools if isinstance(item, dict)]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.start()
        result = self.request(
            "tools/call",
            {"name": name, "arguments": arguments or {}},
            timeout_seconds=self.config.tool_timeout_seconds,
        )
        return result if isinstance(result, dict) else {"content": [{"type": "text", "text": str(result)}]}

    def request(self, method: str, params: dict[str, Any], *, timeout_seconds: float) -> Any:
        process = self._process_or_raise()
        with self._send_lock:
            self._request_id += 1
            request_id = self._request_id
            response_queue: Queue[dict[str, Any]] = Queue(maxsize=1)
            with self._pending_lock:
                self._pending[request_id] = response_queue
            self._write_json(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": method,
                    "params": params,
                }
            )

        deadline = time.monotonic() + timeout_seconds
        try:
            while True:
                if process.poll() is not None:
                    raise MCPProviderError(
                        f"MCP server {self.config.name!r} exited with code {process.returncode}. "
                        f"{self.diagnostics()}"
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"MCP request {self.config.name}.{method} timed out after "
                        f"{timeout_seconds:g}s. {self.diagnostics()}"
                    )
                try:
                    message = response_queue.get(timeout=min(remaining, 0.2))
                    break
                except Empty:
                    continue
        finally:
            with self._pending_lock:
                self._pending.pop(request_id, None)

        if "error" in message:
            raise MCPRequestError(
                f"MCP request {self.config.name}.{method} failed: "
                f"{json.dumps(message.get('error'), ensure_ascii=False)}"
            )
        return message.get("result")

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        with self._send_lock:
            self._write_json({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def close(self) -> None:
        process = self.process
        self.process = None
        self.initialized = False
        if not process or process.poll() is not None:
            return
        try:
            process.terminate()
            process.wait(timeout=2)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass

    def is_alive(self) -> bool:
        process = self.process
        return process is not None and process.poll() is None

    def diagnostics(self) -> str:
        parts: list[str] = []
        if self._stderr_tail:
            parts.append("stderr tail:\n" + "\n".join(self._stderr_tail))
        if self._stdout_tail:
            parts.append("stdout non-json tail:\n" + "\n".join(self._stdout_tail))
        return "\n".join(parts)

    def _process_or_raise(self) -> subprocess.Popen[str]:
        process = self.process
        if process is None:
            raise MCPProviderError(f"MCP server {self.config.name!r} is not started.")
        if process.poll() is not None:
            raise MCPProviderError(
                f"MCP server {self.config.name!r} is not running. {self.diagnostics()}"
            )
        return process

    def _write_json(self, payload: dict[str, Any]) -> None:
        process = self._process_or_raise()
        if process.stdin is None:
            raise MCPProviderError(f"MCP server {self.config.name!r} stdin is closed.")
        process.stdin.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
        process.stdin.flush()

    def _stdout_loop(self) -> None:
        process = self.process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            text = line.strip()
            if not text:
                continue
            try:
                message = json.loads(text)
            except json.JSONDecodeError:
                self._stdout_tail.append(text[-1000:])
                continue
            if not isinstance(message, dict):
                continue
            raw_id = message.get("id")
            if raw_id is None:
                continue
            try:
                request_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            with self._pending_lock:
                response_queue = self._pending.get(request_id)
            if response_queue is not None:
                response_queue.put(message)

    def _stderr_loop(self) -> None:
        process = self.process
        if process is None or process.stderr is None:
            return
        for line in process.stderr:
            text = line.rstrip()
            if text:
                self._stderr_tail.append(text[-1000:])


class MCPToolProvider:
    provider_kind = "mcp"

    def __init__(
        self,
        provider_id: str = "mcp",
        *,
        workspace_root: str | Path,
        config_path: str | Path = DEFAULT_MCP_CONFIG_PATH,
        scope_id: str = "",
        enabled_skill_ids: set[str] | None = None,
    ) -> None:
        self.provider_id = provider_id
        self.workspace_root = Path(workspace_root).resolve()
        self.config_path = Path(config_path)
        self.scope_id = str(scope_id or "").strip()
        self.enabled_skill_ids = set(enabled_skill_ids) if enabled_skill_ids is not None else None
        if not self.config_path.is_absolute():
            self.config_path = self.workspace_root / self.config_path
        self._lock = threading.RLock()
        self._configs: list[MCPServerConfig] = []
        self._sessions: dict[str, MCPStdioSession] = {}
        self._server_errors: dict[str, str] = {}
        self._tools: dict[str, MCPToolEntry] = {}
        self._loaded = False
        self._config_mtime_ns: int | None = None
        self._read_config()

    def list_tools(self) -> list[Tool]:
        with self._lock:
            self._ensure_loaded()
            return [
                Tool(
                    name=entry.exposed_name,
                    description=entry.description,
                    parameters=entry.input_schema,
                    handler=lambda arguments, name=entry.exposed_name: self.call_tool(name, arguments),
                    provider_id=self.provider_id,
                    provider_kind=self.provider_kind,
                    metadata={
                        "mcp_server": entry.server_name,
                        "mcp_tool": entry.original_name,
                    },
                )
                for entry in sorted(self._tools.values(), key=lambda item: item.exposed_name)
            ]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        with self._lock:
            self._ensure_loaded()
            entry = self._tools.get(name)
            if entry is None:
                available = ", ".join(sorted(self._tools))
                raise KeyError(f"Unknown MCP tool {name!r}. Available MCP tools: {available}")
            session = self._sessions.get(entry.server_name)
            if session is None:
                error = self._server_errors.get(entry.server_name, "not connected")
                raise MCPProviderError(f"MCP server {entry.server_name!r} unavailable: {error}")
            try:
                result = session.call_tool(entry.original_name, arguments or {})
            except MCPRequestError:
                raise
            except Exception:
                # Reconnect once for stale stdio sessions.
                session.close()
                self._sessions.pop(entry.server_name, None)
                self._server_errors.pop(entry.server_name, None)
                self._connect_server(next(cfg for cfg in self._configs if cfg.name == entry.server_name))
                session = self._sessions.get(entry.server_name)
                if session is None:
                    error = self._server_errors.get(entry.server_name, "reconnect failed")
                    raise MCPProviderError(f"MCP server {entry.server_name!r} unavailable: {error}")
                result = session.call_tool(entry.original_name, arguments or {})
            return format_mcp_call_result(result)

    def status(self) -> dict[str, Any]:
        with self._lock:
            self._reload_config_if_needed()
            servers = []
            for config in self._configs:
                session = self._sessions.get(config.name)
                error = self._server_errors.get(config.name)
                tool_count = sum(1 for item in self._tools.values() if item.server_name == config.name)
                if not config.enabled:
                    state = "disabled"
                elif config.skill_id and self.enabled_skill_ids is not None and config.skill_id not in self.enabled_skill_ids:
                    state = "disabled_by_skill"
                elif error:
                    state = "error"
                elif session and session.is_alive():
                    state = "connected"
                elif self._loaded:
                    state = "not_connected"
                else:
                    state = "configured"
                servers.append(
                    {
                        "name": config.name,
                        "transport": config.transport,
                        "enabled": config.enabled,
                        "status": state,
                        "tool_count": tool_count,
                        "description": config.description,
                        "error": error or "",
                    }
                )
            if not self.config_path.exists():
                overall = "not_configured"
            elif not self._configs:
                overall = "empty"
            elif any(item["status"] == "connected" for item in servers):
                overall = "connected"
            elif any(item["status"] == "error" for item in servers):
                overall = "error"
            else:
                overall = "configured"
            return {
                "status": overall,
                "config_path": str(self.config_path),
                "servers": servers,
            }

    def close(self) -> None:
        with self._lock:
            for session in self._sessions.values():
                session.close()
            self._sessions.clear()
            self._tools.clear()
            self._loaded = False

    def _ensure_loaded(self) -> None:
        self._reload_config_if_needed()
        if self._loaded:
            return
        self._tools.clear()
        self._server_errors.clear()
        raw_entries: list[tuple[str, dict[str, Any]]] = []
        for config in self._configs:
            if not config.enabled:
                continue
            if config.skill_id and self.enabled_skill_ids is not None and config.skill_id not in self.enabled_skill_ids:
                continue
            self._connect_server(config)
            session = self._sessions.get(config.name)
            if session is None:
                continue
            try:
                for raw_tool in session.list_tools():
                    raw_name = str(raw_tool.get("name") or "").strip().lower()
                    if raw_name in config.blocked_tools:
                        continue
                    raw_entries.append((config.name, raw_tool))
            except Exception as error:
                self._server_errors[config.name] = f"{type(error).__name__}: {error}"
                session.close()
                self._sessions.pop(config.name, None)

        self._tools = build_tool_entries(raw_entries)
        self._loaded = True

    def _connect_server(self, config: MCPServerConfig) -> None:
        if config.transport != "stdio":
            self._server_errors[config.name] = (
                f"unsupported transport {config.transport!r}; only stdio is implemented"
            )
            return
        session = self._sessions.get(config.name)
        if session and session.is_alive() and session.initialized:
            return
        session = MCPStdioSession(config, workspace_root=self.workspace_root)
        try:
            session.start()
        except Exception as error:
            session.close()
            self._server_errors[config.name] = f"{type(error).__name__}: {error}"
            return
        self._sessions[config.name] = session

    def _reload_config_if_needed(self) -> None:
        current_mtime = file_mtime_ns(self.config_path)
        if current_mtime == self._config_mtime_ns:
            return
        self.close()
        self._read_config()

    def _read_config(self) -> None:
        self._config_mtime_ns = file_mtime_ns(self.config_path)
        self._configs = load_mcp_server_configs(self.config_path)
        self._loaded = False


_PROVIDER_CACHE: dict[tuple[str, str, str, tuple[str, ...] | None], MCPToolProvider] = {}
_ATEEXIT_REGISTERED = False


def build_mcp_tool_provider(
    workspace_root: str | Path,
    *,
    config_path: str | Path | None = None,
    scope_id: str | None = None,
    enabled_skill_ids: set[str] | None = None,
) -> MCPToolProvider:
    global _ATEEXIT_REGISTERED
    workspace = Path(workspace_root).resolve()
    config = Path(
        config_path
        or os.getenv("WORK_AGENT_MCP_CONFIG")
        or DEFAULT_MCP_CONFIG_PATH
    )
    if not config.is_absolute():
        config = workspace / config
    scope = str(scope_id or "").strip()
    skill_scope = tuple(sorted(enabled_skill_ids)) if enabled_skill_ids is not None else None
    key = (str(workspace), str(config.resolve() if config.exists() else config), scope, skill_scope)
    provider = _PROVIDER_CACHE.get(key)
    if provider is None:
        provider = MCPToolProvider(
            "mcp",
            workspace_root=workspace,
            config_path=config,
            scope_id=scope,
            enabled_skill_ids=enabled_skill_ids,
        )
        _PROVIDER_CACHE[key] = provider
    if not _ATEEXIT_REGISTERED:
        atexit.register(close_cached_mcp_providers)
        _ATEEXIT_REGISTERED = True
    return provider


def close_cached_mcp_providers() -> None:
    for provider in list(_PROVIDER_CACHE.values()):
        provider.close()


def load_mcp_server_configs(config_path: Path) -> list[MCPServerConfig]:
    if not config_path.exists():
        return []
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"MCP config must be a JSON object: {config_path}")
    raw_servers = data.get("mcp_servers")
    if raw_servers is None:
        raw_servers = data.get("servers")
    if raw_servers is None:
        raw_servers = data.get("mcpServers")
    if raw_servers is None:
        raw_servers = data
    if not isinstance(raw_servers, dict):
        raise ValueError(f"MCP servers config must be an object: {config_path}")
    configs: list[MCPServerConfig] = []
    for name, raw in raw_servers.items():
        if not isinstance(raw, dict):
            continue
        configs.append(MCPServerConfig.from_mapping(str(name), raw))
    return configs


def build_tool_entries(raw_entries: list[tuple[str, dict[str, Any]]]) -> dict[str, MCPToolEntry]:
    base_names: list[str] = []
    for server_name, raw_tool in raw_entries:
        original_name = str(raw_tool.get("name") or "").strip()
        base_names.append(sanitize_tool_name(original_name or server_name))
    duplicates = {name for name in base_names if base_names.count(name) > 1}
    used: set[str] = set()
    entries: dict[str, MCPToolEntry] = {}
    for (server_name, raw_tool), base_name in zip(raw_entries, base_names):
        original_name = str(raw_tool.get("name") or "").strip()
        if not original_name:
            continue
        exposed_name = base_name
        if exposed_name in duplicates or exposed_name in used:
            exposed_name = sanitize_tool_name(f"{server_name}__{original_name}")
        exposed_name = unique_name(exposed_name, used)
        used.add(exposed_name)
        entries[exposed_name] = MCPToolEntry(
            exposed_name=exposed_name,
            original_name=original_name,
            server_name=server_name,
            description=str(raw_tool.get("description") or f"MCP tool {server_name}.{original_name}"),
            input_schema=normalize_input_schema(raw_tool.get("inputSchema") or raw_tool.get("input_schema")),
        )
    return entries


def format_mcp_call_result(result: dict[str, Any]) -> str:
    is_error = bool(result.get("isError") or result.get("is_error"))
    content = result.get("content")
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                parts.append(str(item))
                continue
            content_type = str(item.get("type") or "")
            if content_type == "text":
                parts.append(str(item.get("text") or ""))
            else:
                parts.append(json.dumps(compact_mcp_content_item(item), ensure_ascii=False))
        text = "\n".join(part for part in parts if part).strip()
        if is_error:
            return f"MCP_TOOL_ERROR: {text}" if text else "MCP_TOOL_ERROR"
        return text or json.dumps(result, ensure_ascii=False)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    return f"MCP_TOOL_ERROR: {rendered}" if is_error else rendered


def compact_mcp_content_item(item: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, value in item.items():
        if isinstance(value, str) and len(value) > 1200:
            compact[key] = value[:1200] + f"... [truncated {len(value) - 1200} chars]"
        else:
            compact[key] = value
    return compact


def normalize_input_schema(schema: Any) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}
    normalized = dict(schema)
    if "type" not in normalized and "properties" in normalized:
        normalized["type"] = "object"
    if normalized.get("type") != "object" and "properties" not in normalized:
        return {"type": "object", "properties": {}}
    normalized.setdefault("type", "object")
    normalized.setdefault("properties", {})
    return normalized


def sanitize_tool_name(name: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_-]+", "_", str(name or "").strip())
    text = re.sub(r"_+", "_", text).strip("_-")
    if not text:
        text = "mcp_tool"
    if not re.match(r"^[A-Za-z_]", text):
        text = f"mcp_{text}"
    return text[:MAX_TOOL_NAME_LENGTH]


def safe_provider_name(name: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(name or "").strip()).strip(".-")
    return text or "mcp-server"


def unique_name(base_name: str, used: set[str]) -> str:
    if base_name not in used:
        return base_name
    for index in range(2, 1000):
        suffix = f"_{index}"
        candidate = f"{base_name[: MAX_TOOL_NAME_LENGTH - len(suffix)]}{suffix}"
        if candidate not in used:
            return candidate
    raise ValueError(f"Could not allocate unique tool name for {base_name}")


def resolve_cwd(raw_cwd: str | None, workspace_root: Path) -> str:
    if not raw_cwd:
        return str(workspace_root)
    cwd = Path(raw_cwd).expanduser()
    if not cwd.is_absolute():
        cwd = workspace_root / cwd
    return str(cwd.resolve())


def expand_env_values(env: dict[str, str]) -> dict[str, str]:
    return {key: os.path.expandvars(value) for key, value in env.items()}


def file_mtime_ns(path: Path) -> int | None:
    try:
        return path.stat().st_mtime_ns
    except FileNotFoundError:
        return None
