"""macOS 通知中心投递。

观察者的发现和到期提醒原本只落在 Web 铃铛和 Friday 会话里——用户不盯着页面
就等于没有主动性。这里用 osascript 把同一条消息投到系统通知中心；CLI 进程发
系统通知是 macOS 支持的路径，首次触发会弹一次通知授权。

通知是最不重要的出口：限流、失败、未授权都静默降级为返回 False，绝不影响
主流程。
"""

from __future__ import annotations

import subprocess
import threading
import time

_RATE_LOCK = threading.Lock()
_RATE_WINDOW: list[float] = []
_MAX_PER_60S = 8
_TIMEOUT_SECONDS = 6
_MAX_BODY_CHARS = 500


def _within_rate_budget(now: float) -> bool:
    with _RATE_LOCK:
        while _RATE_WINDOW and now - _RATE_WINDOW[0] > 60:
            _RATE_WINDOW.pop(0)
        if len(_RATE_WINDOW) >= _MAX_PER_60S:
            return False
        _RATE_WINDOW.append(now)
        return True


def notify(title: str, body: str, *, subtitle: str = "") -> bool:
    """投一条系统通知。返回是否真的送达；任何失败都只返回 False。"""

    title = str(title or "").strip() or "Friday"
    body = str(body or "").strip()[:_MAX_BODY_CHARS]
    subtitle = str(subtitle or "").strip()
    if not body:
        return False
    if not _within_rate_budget(time.monotonic()):
        return False
    line = "display notification (item 1 of argv) with title (item 2 of argv)"
    if subtitle:
        line += " subtitle (item 3 of argv)"
    args: list[str] = []
    for script_line in ("on run argv", line, "end run"):
        args += ["-e", script_line]
    argv = [body, title] + ([subtitle] if subtitle else [])
    try:
        proc = subprocess.run(
            ["/usr/bin/osascript", *args, "--", *argv],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0
