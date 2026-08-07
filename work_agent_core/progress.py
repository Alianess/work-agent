from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import itertools
import os
import re
import selectors
import shlex
import subprocess
import threading
import time
from typing import Any


ProgressSink = Callable[[dict[str, Any]], None]
CancelCheck = Callable[[], bool]

_thread_state = threading.local()
_command_counter = itertools.count(1)
DEFAULT_PREVIEW_LIMIT = 4200


def set_tool_progress_sink(sink: ProgressSink | None) -> ProgressSink | None:
    previous = getattr(_thread_state, "sink", None)
    _thread_state.sink = sink
    return previous


def set_tool_cancel_check(check: CancelCheck | None) -> CancelCheck | None:
    """Bind the current turn cancellation probe to one tool worker thread."""
    previous = getattr(_thread_state, "cancel_check", None)
    _thread_state.cancel_check = check
    return previous


def current_tool_cancel_check() -> CancelCheck | None:
    return getattr(_thread_state, "cancel_check", None)


def emit_tool_progress(event: dict[str, Any]) -> None:
    sink = getattr(_thread_state, "sink", None)
    if not sink:
        return
    sink(event)


def compact_preview_text(text: str, *, limit: int = DEFAULT_PREVIEW_LIMIT) -> str:
    """Return a bounded preview while preserving both start and latest tail."""
    value = str(text or "")
    if len(value) <= limit:
        return value
    head = max(800, limit // 3)
    tail = max(1200, limit - head - 120)
    omitted = len(value) - head - tail
    return (
        value[:head].rstrip()
        + f"\n\n… 省略 {omitted} 字，活动栏仅保留预览，完整内容会写入文件 …\n\n"
        + value[-tail:].lstrip()
    )


def run_logged_process(
    command: list[str],
    *,
    cwd: str | Path,
    timeout_seconds: int,
    label: str,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    command_id = next_command_id()
    command_text = redact_command(shlex.join([str(part) for part in command]))
    cwd_path = Path(cwd)
    emit_tool_progress(
        {
            "event": "activity",
            "id": command_id,
            "phase": "action",
            "title": "已运行命令",
            "detail": label,
            "content": "",
            "activity_type": "command",
            "command": command_text,
            "command_status": "running",
        }
    )

    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    started_at = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=cwd_path,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        bufsize=1,
    )
    selector = selectors.DefaultSelector()
    stream_names: dict[Any, str] = {}
    if process.stdout is not None:
        selector.register(process.stdout, selectors.EVENT_READ)
        stream_names[process.stdout] = "stdout"
    if process.stderr is not None:
        selector.register(process.stderr, selectors.EVENT_READ)
        stream_names[process.stderr] = "stderr"

    timed_out = False
    last_heartbeat = started_at
    heartbeat_emitted = False
    try:
        while selector.get_map():
            if time.monotonic() - started_at > max(1, int(timeout_seconds)):
                timed_out = True
                process.kill()
                break
            events = selector.select(timeout=0.2)
            if not events:
                now = time.monotonic()
                if process.poll() is None and now - last_heartbeat >= 10:
                    last_heartbeat = now
                    emit_command_heartbeat(
                        command_id,
                        command_heartbeat_text(label, int(now - started_at)),
                    )
                    heartbeat_emitted = True
                if process.poll() is not None:
                    drain_registered_streams(selector, stream_names, stdout_parts, stderr_parts, command_id)
                continue
            for key, _ in events:
                stream = key.fileobj
                line = stream.readline()
                if line:
                    if stream_names.get(stream) == "stderr":
                        stderr_parts.append(line)
                    else:
                        stdout_parts.append(line)
                    emit_command_delta(command_id, line)
                else:
                    try:
                        selector.unregister(stream)
                    except Exception:
                        pass
    finally:
        selector.close()

    returncode = process.wait(timeout=5)
    for stream in (process.stdout, process.stderr):
        if stream is not None:
            stream.close()
    stdout = "".join(stdout_parts)
    stderr = "".join(stderr_parts)
    if timed_out:
        message = f"{label}超时，请检查任务输入、拆分处理或稍后重试。"
        if heartbeat_emitted:
            elapsed_seconds = int(time.monotonic() - started_at)
            emit_command_heartbeat(
                command_id,
                f"✗ [{elapsed_seconds}s] {label}超时\n",
                status="error",
            )
        emit_command_delta(command_id, f"\n✗ {message}\n", status="error")
        raise TimeoutError(message)

    status = "success" if returncode == 0 else "error"
    if heartbeat_emitted:
        elapsed_seconds = int(time.monotonic() - started_at)
        status_mark = "✓" if returncode == 0 else "✗"
        status_text = "完成" if returncode == 0 else "失败"
        emit_command_heartbeat(
            command_id,
            f"{status_mark} [{elapsed_seconds}s] {label}{status_text}\n",
            status=status,
        )
    final_message = "✓ 成功" if returncode == 0 else f"✗ 失败，退出码 {returncode}"
    emit_command_delta(command_id, f"\n{final_message}\n", status=status)
    result = subprocess.CompletedProcess(command, returncode, stdout, stderr)
    if check and returncode != 0:
        raise subprocess.CalledProcessError(returncode, command, output=stdout, stderr=stderr)
    return result


def drain_registered_streams(
    selector: selectors.BaseSelector,
    stream_names: dict[Any, str],
    stdout_parts: list[str],
    stderr_parts: list[str],
    command_id: str,
) -> None:
    for key in list(selector.get_map().values()):
        stream = key.fileobj
        rest = stream.read()
        if rest:
            if stream_names.get(stream) == "stderr":
                stderr_parts.append(rest)
            else:
                stdout_parts.append(rest)
            emit_command_delta(command_id, rest)
        try:
            selector.unregister(stream)
        except Exception:
            pass


def emit_command_delta(command_id: str, content: str, *, status: str | None = None) -> None:
    event: dict[str, Any] = {
        "event": "activity_delta",
        "id": command_id,
        "phase": "action",
        "title": "已运行命令",
        "content": content,
        "activity_type": "command",
    }
    if status:
        event["command_status"] = status
    emit_tool_progress(event)


def emit_command_heartbeat(command_id: str, content: str, *, status: str = "running") -> None:
    emit_tool_progress(
        {
            "event": "activity_delta",
            "id": f"{command_id}-heartbeat",
            "phase": "action",
            "title": "运行状态",
            "content": content,
            "append_mode": "replace",
            "activity_type": "command",
            "command_status": status,
        }
    )


def command_heartbeat_text(label: str, elapsed_seconds: int) -> str:
    if "Qwen3-ASR" in label or "转写" in label:
        return f"[{elapsed_seconds}s] {label}处理中...\n"
    if "音频预处理" in label:
        return f"[{elapsed_seconds}s] {label}处理中...\n"
    return f"[{elapsed_seconds}s] {label} 执行中...\n"


def next_command_id() -> str:
    return f"cmd-{int(time.time() * 1000)}-{next(_command_counter)}"


def redact_command(command: str) -> str:
    redacted = re.sub(r"sk-[A-Za-z0-9_-]{10,}", "sk-***", command)
    redacted = re.sub(
        r"(?i)(api[_-]?key|token|secret|password)=('[^']*'|\"[^\"]*\"|\S+)",
        r"\1=***",
        redacted,
    )
    redacted = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._-]{12,}", r"\1***", redacted)
    return redacted


def safe_process_environment(env: dict[str, str] | None = None) -> dict[str, str]:
    base = dict(os.environ)
    if env:
        base.update(env)
    return base
