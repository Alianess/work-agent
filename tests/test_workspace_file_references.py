from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from work_agent_core.config import ModelProfile
from work_agent_core import web_server
from work_agent_core.web_server import (
    enrich_image_attachments_for_model,
    extract_workspace_file_references,
    extract_workspace_paths,
    image_fallback_final_content,
    normalize_workspace_reference_path,
    sanitize_context_file_paths,
)


class WorkspaceFileReferenceTests(unittest.TestCase):
    @staticmethod
    def profile(*, supports_vision: bool) -> ModelProfile:
        return ModelProfile(
            name="test-profile",
            provider="openai-compatible",
            base_url="https://api.example.com/v1",
            model="test-model",
            api_key_env="TEST_KEY",
            supports_vision=supports_vision,
        )

    def test_text_only_model_skips_historical_images_without_removing_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            image_path = root / "tmp" / "history.jpg"
            image_path.parent.mkdir(parents=True)
            image_path.write_bytes(b"not-a-real-jpeg")
            messages = [
                {"role": "user", "content": "请看 tmp/history.jpg"},
                {"role": "assistant", "content": "好的"},
                {"role": "user", "content": "继续，只处理文字"},
                {"role": "user", "content": "重复引用 tmp/history.jpg"},
            ]

            with patch.object(web_server, "WORKSPACE_ROOT", root):
                prepared = enrich_image_attachments_for_model(
                    messages,
                    self.profile(supports_vision=False),
                    workspace_root=root,
                )

            self.assertEqual(prepared.skipped_count, 1)
            self.assertEqual(prepared.attached_count, 0)
            self.assertIn("不支持图片识别", prepared.notice)
            self.assertEqual(prepared.messages, messages)
            self.assertEqual(prepared.messages[0]["content"], "请看 tmp/history.jpg")

    def test_vision_model_attaches_each_historical_image_only_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            image_path = root / "tmp" / "history.png"
            image_path.parent.mkdir(parents=True)
            image_path.write_bytes(b"png-bytes")
            messages = [
                {"role": "user", "content": "图片 tmp/history.png"},
                {"role": "user", "content": "还是 tmp/history.png"},
            ]

            with patch.object(web_server, "WORKSPACE_ROOT", root):
                prepared = enrich_image_attachments_for_model(
                    messages,
                    self.profile(supports_vision=True),
                    workspace_root=root,
                )

            self.assertEqual(prepared.attached_count, 1)
            self.assertEqual(prepared.skipped_count, 0)
            self.assertEqual(prepared.notice, "")
            first_content = prepared.messages[0]["content"]
            self.assertIsInstance(first_content, list)
            self.assertEqual(first_content[1]["type"], "image_url")
            self.assertTrue(first_content[1]["image_url"]["url"].startswith("data:image/png;base64,"))
            self.assertEqual(prepared.messages[1], messages[1])

    def test_image_fallback_notice_is_visible_in_final_reply(self) -> None:
        content = image_fallback_final_content("继续处理文字。", "当前模型不支持图片识别。")

        self.assertIn("⚠️ 当前模型不支持图片识别。", content)
        self.assertTrue(content.endswith("继续处理文字。"))

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
