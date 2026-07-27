from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from work_agent_core.audio_metadata import (
    format_duration,
    parse_media_datetime,
    probe_audio_metadata,
    recording_metadata_summary,
)


class AudioMetadataTests(unittest.TestCase):
    def test_probe_reads_embedded_start_and_duration(self) -> None:
        ffprobe_payload = {
            "format": {
                "duration": "6587.178667",
                "tags": {"creation_time": "2026-07-07T01:10:06.000000Z"},
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "meeting.m4a"
            audio_path.touch()
            completed = subprocess.CompletedProcess(
                args=["ffprobe"],
                returncode=0,
                stdout=json.dumps(ffprobe_payload),
                stderr="",
            )
            with patch("work_agent_core.audio_metadata.find_runtime_executable", return_value="/usr/bin/ffprobe"):
                with patch("work_agent_core.audio_metadata.subprocess.run", return_value=completed):
                    metadata = probe_audio_metadata(audio_path)

        self.assertEqual(metadata["recording_started_at_utc"], "2026-07-07T01:10:06+00:00")
        self.assertEqual(metadata["recording_started_at_epoch"], 1783386606)
        self.assertEqual(metadata["duration_seconds"], 6587.179)
        self.assertEqual(metadata["recording_time_source"], "embedded_media_creation_time")
        self.assertTrue(metadata["recording_time_timezone_known"])
        self.assertEqual(metadata["recording_time_validation"], "plausible_file_timeline")
        self.assertIn("recording_ended_at", metadata)

    def test_probe_does_not_use_file_mtime_when_creation_time_is_missing(self) -> None:
        ffprobe_payload = {"format": {"duration": "60.0", "tags": {}}}
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "meeting.m4a"
            audio_path.touch()
            completed = subprocess.CompletedProcess(
                args=["ffprobe"],
                returncode=0,
                stdout=json.dumps(ffprobe_payload),
                stderr="",
            )
            with patch("work_agent_core.audio_metadata.find_runtime_executable", return_value="/usr/bin/ffprobe"):
                with patch("work_agent_core.audio_metadata.subprocess.run", return_value=completed):
                    metadata = probe_audio_metadata(audio_path)

        self.assertEqual(metadata, {"duration_seconds": 60.0})
        self.assertNotIn("recording_started_at", metadata)

    def test_probe_rejects_creation_time_that_conflicts_with_file_timeline(self) -> None:
        ffprobe_payload = {
            "format": {
                "duration": "3600.0",
                "tags": {"creation_time": "2026-07-09T09:57:00.000000Z"},
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "exported.m4a"
            audio_path.touch()
            os.utime(audio_path, (1783591140, 1783591140))
            completed = subprocess.CompletedProcess(
                args=["ffprobe"],
                returncode=0,
                stdout=json.dumps(ffprobe_payload),
                stderr="",
            )
            with patch("work_agent_core.audio_metadata.find_runtime_executable", return_value="/usr/bin/ffprobe"):
                with patch("work_agent_core.audio_metadata.subprocess.run", return_value=completed):
                    metadata = probe_audio_metadata(audio_path)

        self.assertEqual(metadata["media_created_at_utc"], "2026-07-09T09:57:00+00:00")
        self.assertEqual(metadata["recording_time_validation"], "inconsistent_with_file_timeline")
        self.assertNotIn("recording_started_at", metadata)

    def test_summary_keeps_recording_time_distinct_from_meeting_time(self) -> None:
        summary = recording_metadata_summary(
            {
                "recording_started_at": "2026-07-07T09:10:06+08:00",
                "duration_seconds": 6587.179,
            }
        )

        self.assertIn("录音文件内嵌开始时间", summary)
        self.assertIn("1小时49分47秒", summary)
        self.assertIn("不自动等同于会议正式开始时间", summary)

    def test_datetime_and_duration_helpers(self) -> None:
        parsed = parse_media_datetime("2026-07-07T01:10:06Z")

        self.assertIsNotNone(parsed)
        self.assertEqual(format_duration(1847.4), "30分47秒")


if __name__ == "__main__":
    unittest.main()
