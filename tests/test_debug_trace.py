from __future__ import annotations

import unittest

from work_agent_core.debug_trace import sanitize_debug_payload


class DebugTraceSanitizationTests(unittest.TestCase):
    def test_usage_counts_are_visible_but_credentials_remain_redacted(self) -> None:
        cleaned = sanitize_debug_payload(
            {
                "prompt_tokens": 123_456,
                "completion_tokens": 7_890,
                "total_tokens": 131_346,
                "context_trigger_tokens": 217_600,
                "access_token": "secret-session-value",
                "api_key": "secret-api-value",
            }
        )

        self.assertEqual(cleaned["prompt_tokens"], 123_456)
        self.assertEqual(cleaned["completion_tokens"], 7_890)
        self.assertEqual(cleaned["total_tokens"], 131_346)
        self.assertEqual(cleaned["context_trigger_tokens"], 217_600)
        self.assertEqual(cleaned["access_token"], "[REDACTED]")
        self.assertEqual(cleaned["api_key"], "[REDACTED]")

    def test_string_token_fields_are_never_unredacted(self) -> None:
        cleaned = sanitize_debug_payload({"prompt_tokens": "123456"})
        self.assertEqual(cleaned["prompt_tokens"], "[REDACTED]")


if __name__ == "__main__":
    unittest.main()
