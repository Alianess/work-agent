from __future__ import annotations

import unittest

from work_agent_core.web_server import (
    extract_workspace_file_references,
    extract_workspace_paths,
    normalize_workspace_reference_path,
    sanitize_context_file_paths,
)


class WorkspaceFileReferenceTests(unittest.TestCase):
    def test_nul_separated_text_paths_are_extracted_individually(self) -> None:
        text = (
            "meet_files/first/qwen3-asr/transcript.txt"
            "\x00"
            "meet_files/second/qwen3-asr/transcript.txt"
        )

        self.assertEqual(
            extract_workspace_paths(text),
            [
                "meet_files/first/qwen3-asr/transcript.txt",
                "meet_files/second/qwen3-asr/transcript.txt",
            ],
        )

    def test_serialized_nul_context_path_is_split_before_normalization(self) -> None:
        joined = (
            "meet_files/first/qwen3-asr/transcript.txt"
            "\\u0000"
            "meet_files/second/qwen3-asr/transcript.txt"
        )

        self.assertEqual(
            sanitize_context_file_paths([joined]),
            [
                "meet_files/first/qwen3-asr/transcript.txt",
                "meet_files/second/qwen3-asr/transcript.txt",
            ],
        )
        self.assertEqual(
            [item["path"] for item in extract_workspace_file_references("", [joined])],
            [
                "meet_files/first/qwen3-asr/transcript.txt",
                "meet_files/second/qwen3-asr/transcript.txt",
            ],
        )

    def test_overlong_path_is_ignored_without_stat_error(self) -> None:
        overlong = "meet_files/" + ("x" * 5000)

        self.assertEqual(normalize_workspace_reference_path(overlong), "")
        self.assertEqual(sanitize_context_file_paths([overlong]), [])


if __name__ == "__main__":
    unittest.main()
