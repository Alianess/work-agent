from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()
