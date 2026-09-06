"""macOS 通知投递的契约：正文必填、限流、失败静默、argv 传参没有注入面。"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from work_agent_core.host_services import desktop_notify


class DesktopNotifyTests(unittest.TestCase):
    def setUp(self) -> None:
        desktop_notify._RATE_WINDOW.clear()

    def test_empty_body_is_dropped_without_spawning_a_process(self) -> None:
        with patch.object(desktop_notify.subprocess, "run") as run:
            self.assertFalse(desktop_notify.notify("Friday", "   "))
            run.assert_not_called()

    def test_body_title_subtitle_travel_as_argv_not_as_script_source(self) -> None:
        body = '正文"; display dialog "pwned'
        with patch.object(desktop_notify.subprocess, "run") as run:
            run.return_value = SimpleNamespace(returncode=0)
            self.assertTrue(desktop_notify.notify("Friday 提醒", body, subtitle="Friday"))

        argv = run.call_args.args[0]
        self.assertEqual(argv[0], "/usr/bin/osascript")
        self.assertIn("--", argv)
        # 正文整段作为一个 argv 元素传给 AppleScript 的 argv，不进脚本源码。
        self.assertIn(body, argv)
        script = " ".join(argv[1 : argv.index("--")])
        self.assertNotIn("pwned", script)

    def test_failure_and_timeout_return_false(self) -> None:
        with patch.object(desktop_notify.subprocess, "run") as run:
            run.return_value = SimpleNamespace(returncode=1)
            self.assertFalse(desktop_notify.notify("t", "b"))
            run.side_effect = OSError("blocked")
            self.assertFalse(desktop_notify.notify("t", "b"))
            run.side_effect = subprocess_timeout()
            self.assertFalse(desktop_notify.notify("t", "b"))

    def test_rate_limit_drops_excess_notifications(self) -> None:
        with patch.object(desktop_notify.subprocess, "run") as run:
            run.return_value = SimpleNamespace(returncode=0)
            results = [desktop_notify.notify("t", f"b{i}") for i in range(10)]

        self.assertEqual(results, [True] * 8 + [False, False])
        self.assertEqual(run.call_count, 8)


def subprocess_timeout():
    return subprocess.TimeoutExpired(cmd="osascript", timeout=6)


if __name__ == "__main__":
    unittest.main()
