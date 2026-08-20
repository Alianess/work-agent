from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from work_agent_core import web_server
from work_agent_core.web_server import meeting_time_from_manifest


class MeetingArchiveTests(unittest.TestCase):
    def test_explicit_meeting_time_wins(self) -> None:
        explicit = {"display": "2026年7月16日上午", "source": "transcript"}

        result = meeting_time_from_manifest(
            {
                "meeting_time": explicit,
                "recording_metadata": {"recording_started_at": "2026-07-16T10:06:00+08:00"},
            }
        )

        self.assertEqual(result, explicit)

    def test_valid_recording_start_supplies_date_fallback(self) -> None:
        result = meeting_time_from_manifest(
            {
                "recording_metadata": {
                    "recording_started_at": "2026-07-16T10:06:00+08:00",
                    "recording_ended_at": "2026-07-16T10:34:09+08:00",
                }
            }
        )

        self.assertEqual(result["display"], "2026年7月16日")
        self.assertEqual(result["start"], "2026-07-16T10:06:00+08:00")
        self.assertEqual(result["source"], "recording_metadata_fallback")

    def test_missing_or_invalid_recording_start_has_no_fallback(self) -> None:
        self.assertIsNone(meeting_time_from_manifest({"recording_metadata": {}}))
        self.assertIsNone(
            meeting_time_from_manifest({"recording_metadata": {"recording_started_at": "not-a-time"}})
        )

    def test_archive_accepts_iso_created_at(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_dir = root / "meet_files" / "会议项目" / "测试会议"
            archive_dir.mkdir(parents=True)
            manifest = archive_dir / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "meeting_id": "meeting-test",
                        "title": "测试会议",
                        "created_at": "2026-08-07T16:53:56+08:00",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with patch.object(web_server, "WORKSPACE_ROOT", root):
                result = web_server.meeting_archive_from_manifest(manifest)

        self.assertEqual(result["created_at"], 1786092836)
        self.assertIsInstance(result["updated_at"], int)


if __name__ == "__main__":
    unittest.main()
