from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any
import json
import os
import re
import threading
import time
import uuid


DEBUG_TRACE_DIR = Path("meet_files/debug_traces")
DEFAULT_STRING_LIMIT = 4000
RECENT_INDEX_NAME = "_recent.jsonl"
SENSITIVE_KEY_RE = re.compile(r"(api[_-]?key|authorization|password|passwd|secret|token)", re.I)
_WRITE_LOCK = threading.RLock()


class DebugTrace:
    """Lightweight local JSONL observability for one agent turn."""

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        conversation_id: str,
        route: str,
        profile: str = "",
        model: str = "",
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.conversation_id = safe_trace_name(conversation_id) or "unknown"
        self.route = route
        self.profile = profile
        self.model = model
        self.trace_id = f"trace-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
        self.started_monotonic = time.monotonic()
        self.enabled = debug_trace_enabled()
        self.trace_dir = self.workspace_root / DEBUG_TRACE_DIR
        self.trace_path = self.trace_dir / f"{self.conversation_id}.jsonl"
        self.index_path = self.trace_dir / RECENT_INDEX_NAME

    def emit(self, event: str, **payload: Any) -> None:
        if not self.enabled:
            return
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "ts_ms": int(time.time() * 1000),
            "elapsed_ms": int((time.monotonic() - self.started_monotonic) * 1000),
            "trace_id": self.trace_id,
            "conversation_id": self.conversation_id,
            "route": self.route,
            "profile": self.profile,
            "model": self.model,
            "event": event,
            **sanitize_debug_payload(payload),
        }
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with _WRITE_LOCK:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            append_line(self.trace_path, line)
            append_line(self.index_path, line)

    def context_payload(self) -> dict[str, str]:
        return {
            "trace_id": self.trace_id,
            "debug_trace_path": str(self.trace_path.relative_to(self.workspace_root)),
        }


def debug_trace_enabled() -> bool:
    value = str(os.getenv("WORK_AGENT_DEBUG_TRACE", "1")).strip().lower()
    return value not in {"0", "false", "no", "off"}


def append_line(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def sanitize_debug_payload(value: Any, *, string_limit: int = DEFAULT_STRING_LIMIT) -> Any:
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if SENSITIVE_KEY_RE.search(key_text):
                cleaned[key_text] = "[REDACTED]"
            else:
                cleaned[key_text] = sanitize_debug_payload(item, string_limit=string_limit)
        return cleaned
    if isinstance(value, list):
        return [sanitize_debug_payload(item, string_limit=string_limit) for item in value[:80]]
    if isinstance(value, tuple):
        return [sanitize_debug_payload(item, string_limit=string_limit) for item in value[:80]]
    if isinstance(value, str):
        return clip_debug_string(value, string_limit)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return clip_debug_string(str(value), string_limit)


def clip_debug_string(text: str, limit: int = DEFAULT_STRING_LIMIT) -> str:
    value = str(text or "")
    if len(value) <= limit:
        return value
    head = max(400, limit // 3)
    tail = max(800, limit - head - 80)
    omitted = len(value) - head - tail
    return value[:head] + f"\n…[debug truncated {omitted} chars]…\n" + value[-tail:]


def safe_trace_name(raw_value: Any) -> str:
    text = str(raw_value or "").strip()
    text = re.sub(r"[^A-Za-z0-9_.:-]+", "-", text).strip(".-")
    return text[:120]


def list_debug_traces(
    workspace_root: str | Path,
    *,
    conversation_id: str | None = None,
    trace_id: str | None = None,
    limit: int = 200,
) -> dict[str, Any]:
    root = Path(workspace_root).resolve()
    trace_dir = root / DEBUG_TRACE_DIR
    limit = max(1, min(int(limit or 200), 2000))
    if conversation_id:
        path = trace_dir / f"{safe_trace_name(conversation_id)}.jsonl"
    else:
        path = trace_dir / RECENT_INDEX_NAME
    records = read_jsonl_tail(path, limit=limit * 3 if trace_id else limit)
    if trace_id:
        records = [record for record in records if str(record.get("trace_id") or "") == trace_id]
        records = records[-limit:]
    traces: dict[str, dict[str, Any]] = {}
    for record in records:
        current_trace_id = str(record.get("trace_id") or "")
        if not current_trace_id:
            continue
        item = traces.setdefault(
            current_trace_id,
            {
                "trace_id": current_trace_id,
                "conversation_id": record.get("conversation_id") or "",
                "route": record.get("route") or "",
                "profile": record.get("profile") or "",
                "model": record.get("model") or "",
                "first_ts": record.get("ts") or "",
                "last_ts": record.get("ts") or "",
                "event_count": 0,
            },
        )
        item["last_ts"] = record.get("ts") or item["last_ts"]
        item["event_count"] += 1
        item["last_event"] = record.get("event") or ""
    return {
        "ok": True,
        "enabled": debug_trace_enabled(),
        "path": str(path.relative_to(root)) if path.exists() else str(path),
        "events": records[-limit:],
        "traces": list(traces.values())[-limit:],
    }


def read_jsonl_tail(path: Path, *, limit: int) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    lines: deque[str] = deque(maxlen=limit)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip():
                lines.append(line)
    records: list[dict[str, Any]] = []
    for line in lines:
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            records.append(parsed)
    return records


def compact_message_summary(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        item: dict[str, Any] = {
            "role": role,
            "content_chars": len(str(message.get("content") or "")),
        }
        if role == "assistant" and isinstance(message.get("tool_calls"), list):
            item["tool_calls"] = [
                {
                    "id": str(call.get("id") or "") if isinstance(call, dict) else "",
                    "name": str((call.get("function") or {}).get("name") or "")
                    if isinstance(call, dict)
                    else "",
                }
                for call in message.get("tool_calls", [])[:20]
            ]
        if role == "tool":
            item["name"] = str(message.get("name") or "")
            item["tool_call_id"] = str(message.get("tool_call_id") or "")
        summary.append(item)
    return summary
