from __future__ import annotations

import base64
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from work_agent_core import web_server


class TemporarySyncTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name).resolve()
        self.root_patch = patch.object(
            web_server,
            "temporary_sync_root",
            side_effect=self._temporary_sync_root,
        )
        self.active_account = "first"
        self.root_patch.start()

    def tearDown(self) -> None:
        self.root_patch.stop()
        self.temp_dir.cleanup()

    def _temporary_sync_root(self) -> Path:
        root = self.root / self.active_account
        (root / "files").mkdir(parents=True, exist_ok=True)
        return root

    def test_text_overwrites_previous_value_without_expiry(self) -> None:
        first = web_server.save_temporary_sync_text_payload({"content": "电脑 A 的文字"})
        second = web_server.save_temporary_sync_text_payload({"content": "电脑 B 的新文字"})
        payload = web_server.temporary_sync_payload(now=second["text"]["updated_at"] + 86400 * 365)

        self.assertEqual(first["message"], "文字已同步")
        self.assertEqual(payload["text"]["content"], "电脑 B 的新文字")
        self.assertEqual(payload["text"]["updated_at"], second["text"]["updated_at"])

    def test_files_are_available_until_one_hour_then_removed(self) -> None:
        with patch.object(web_server.time, "time", return_value=1_000):
            uploaded = web_server.add_temporary_sync_file_payload(
                {
                    "name": "传输文件.txt",
                    "mime_type": "text/plain",
                    "content_base64": base64.b64encode("临时内容".encode("utf-8")).decode("ascii"),
                }
            )["file"]

        self.assertEqual(uploaded["expires_at"], 1_000 + 3_600)
        data_path, metadata = web_server.resolve_temporary_sync_file(uploaded["id"], now=4_599)
        self.assertEqual(data_path.read_text(encoding="utf-8"), "临时内容")
        self.assertEqual(metadata["name"], "传输文件.txt")

        expired_payload = web_server.temporary_sync_payload(now=4_600)
        self.assertEqual(expired_payload["files"], [])
        self.assertFalse(data_path.exists())

    def test_each_account_has_an_independent_sync_area(self) -> None:
        web_server.save_temporary_sync_text_payload({"content": "仅第一个账号可见"})
        self.active_account = "second"

        self.assertEqual(web_server.temporary_sync_payload()["text"]["content"], "")
        web_server.save_temporary_sync_text_payload({"content": "第二个账号的文字"})

        self.active_account = "first"
        self.assertEqual(
            web_server.temporary_sync_payload()["text"]["content"],
            "仅第一个账号可见",
        )

    def test_invalid_file_ids_and_oversized_text_are_rejected(self) -> None:
        self.assertIsNone(web_server.parse_temporary_sync_file_route("/api/temporary-sync/files/../x"))
        self.assertEqual(web_server.sanitize_filename("a\r\nb.txt"), "a__b.txt")
        self.assertEqual(web_server.normalize_mime_type("text/plain; charset=utf-8"), "text/plain")
        self.assertEqual(web_server.normalize_mime_type("text/plain\r\nX-Test: yes"), "application/octet-stream")
        with self.assertRaisesRegex(ValueError, "文字内容不能超过"):
            web_server.save_temporary_sync_text_payload(
                {"content": "x" * (web_server.TEMP_SYNC_MAX_TEXT_CHARS + 1)}
            )


if __name__ == "__main__":
    unittest.main()
