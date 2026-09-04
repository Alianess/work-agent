"""工作区外只读白名单：读放行、写不行、不复制。

用户在 agent 设置里声明 extra_read_roots（桌面、下载……）。设计动机：
读一张截图几乎没有风险，写才有——而读和写原来共用 resolve() 的一条边界，
导致"把桌面截图路径给我"都要先复制进工作区，磁盘被只读操作越用越大。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from work_agent_core import web_server
from work_agent_core.execution.backends.seatbelt import SeatbeltBackend
from work_agent_core.tools import WorkspaceFiles


class ExtraReadRootsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        base = Path(self._temporary.name)
        self.workspace = (base / "workspace").resolve()
        self.workspace.mkdir()
        self.extra = (base / "Desktop").resolve()
        self.extra.mkdir()
        (self.extra / "note.md").write_text("桌面上的笔记", encoding="utf-8")
        (self.extra / "shots").mkdir()
        (self.extra / "shots" / "a.png").write_bytes(b"png")
        self.files = WorkspaceFiles(self.workspace, extra_read_roots=[self.extra])
        self.addCleanup(self._temporary.cleanup)

    def test_reading_a_text_file_under_an_extra_root_works(self) -> None:
        result = self.files.read_text({"path": str(self.extra / "note.md")})
        self.assertEqual(result, "桌面上的笔记")

    def test_listing_a_directory_under_an_extra_root_works(self) -> None:
        result = self.files.read_text({"path": str(self.extra / "shots")})
        self.assertIn("a.png", result)

    def test_writing_under_an_extra_root_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            self.files.write_text({"path": str(self.extra / "evil.md"), "content": "x"})
        with self.assertRaises(ValueError):
            self.files.edit_text(
                {"path": str(self.extra / "note.md"), "old_text": "笔记", "new_text": "篡改"}
            )

    def test_outside_any_root_is_still_refused(self) -> None:
        elsewhere = Path(self._temporary.name) / "elsewhere.md"
        elsewhere.write_text("x", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.files.read_text({"path": str(elsewhere)})

    def test_without_extra_roots_reads_stay_workspace_only(self) -> None:
        strict = WorkspaceFiles(self.workspace)
        with self.assertRaises(ValueError):
            strict.read_text({"path": str(self.extra / "note.md")})

    def test_seatbelt_profile_includes_extra_roots_as_read_only(self) -> None:
        backend = SeatbeltBackend(
            sandbox_exec="/usr/bin/true",
            runtime_workspace_root=self.workspace,
            readable_source_root=self.workspace,
            extra_read_roots=(self.extra,),
        )
        profile = backend._profile(self.workspace, backend._read_roots())
        self.assertIn(f'(subpath "{self.extra}")', profile)


class ExtraReadRootsSettingsTests(unittest.TestCase):
    """设置页保存白名单：提交列表就更新，不带键时原样保留。"""

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.settings_path = Path(temporary.name) / "agent_settings.json"
        patcher = patch.object(
            web_server, "user_agent_settings_path", return_value=self.settings_path
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_settings_save_updates_the_whitelist(self) -> None:
        web_server.save_agent_settings_payload(
            {"extra_read_roots": ["~/Desktop", "  ", "~/Downloads"]}
        )
        self.assertEqual(
            web_server.load_agent_settings()["extra_read_roots"],
            ["~/Desktop", "~/Downloads"],
        )

    def test_settings_save_without_the_key_keeps_current_roots(self) -> None:
        web_server.save_agent_settings_payload({"extra_read_roots": ["~/Desktop"]})
        web_server.save_agent_settings_payload({"details": "只改背景，不该冲掉白名单"})
        self.assertEqual(
            web_server.load_agent_settings()["extra_read_roots"],
            ["~/Desktop"],
        )

    def test_settings_save_preserves_and_updates_auto_approve(self) -> None:
        web_server.save_agent_settings_payload({"auto_approve": False})
        self.assertFalse(web_server.load_agent_settings()["auto_approve"])

        web_server.save_agent_settings_payload({"details": "只改背景，不该重置审查模式"})
        self.assertFalse(web_server.load_agent_settings()["auto_approve"])

        web_server.save_agent_settings_payload({"auto_approve": True})
        self.assertTrue(web_server.load_agent_settings()["auto_approve"])


if __name__ == "__main__":
    unittest.main()
