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
            attachment_block = "参考附件：\n- [图片] history.jpg: tmp/history.jpg"
            messages = [
                {"role": "user", "content": "继续，只处理文字"},
                {"role": "assistant", "content": "好的"},
                {"role": "user", "content": f"请看\n\n{attachment_block}"},
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
            self.assertEqual(prepared.messages[2]["content"], f"请看\n\n{attachment_block}")

    def test_vision_model_attaches_each_historical_image_only_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            image_path = root / "tmp" / "history.png"
            image_path.parent.mkdir(parents=True)
            image_path.write_bytes(b"png-bytes")
            messages = [
                {"role": "user", "content": "图片\n\n参考附件：\n- [图片] history.png: tmp/history.png"},
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

    def test_typed_path_stays_text_and_never_attaches(self) -> None:
        # 用户只给路径，智能体就只能看到路径；要看图得自己调 read_file。
        # 把路径自动附图，模型就会宣称「给我路径我就能看到」——契约就错了。
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            image_path = root / "tmp" / "typed.png"
            image_path.parent.mkdir(parents=True)
            image_path.write_bytes(b"png-bytes")
            messages = [
                {"role": "user", "content": "帮我看看 tmp/typed.png 这张图"},
            ]

            with patch.object(web_server, "WORKSPACE_ROOT", root):
                prepared = enrich_image_attachments_for_model(
                    messages,
                    self.profile(supports_vision=True),
                    workspace_root=root,
                )

            self.assertEqual(prepared.attached_count, 0)
            self.assertEqual(prepared.skipped_count, 0)
            self.assertEqual(prepared.messages, messages)

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


class SpacedFileNameTests(unittest.TestCase):
    """文件名里带空格的附件，图不许被静默丢掉。

    macOS 截屏永远叫「截屏2026-08-20 19.29.13.png」。路径正则原来以空格为硬边界，
    于是它被截成「…/20260820-192915-截屏2026-08-20」，不是文件，图就没了——
    模型只好去读文件属性，答出「86% 是白色像素」这种话。
    """

    @staticmethod
    def profile(*, supports_vision: bool) -> ModelProfile:
        return ModelProfile(
            name="spaced-name-test",
            provider="openai-compatible",
            base_url="https://example.invalid/v1",
            model="test-model",
            api_key_env="UNUSED",
            supports_vision=supports_vision,
        )

    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        (self.root / "meet_files" / "attachments").mkdir(parents=True)
        self.image = self.root / "meet_files" / "attachments" / "截屏2026-08-20 19.29.13.png"
        # 一个真的能被 Pillow 打开的最小 PNG
        from PIL import Image

        Image.new("RGB", (8, 8), (10, 20, 30)).save(self.image, "PNG")
        (self.root / "config").mkdir()
        (self.root / "config" / "model_profiles.json").write_text("{}", encoding="utf-8")
        self.addCleanup(self._temporary.cleanup)

    @property
    def reference(self) -> str:
        return "meet_files/attachments/截屏2026-08-20 19.29.13.png"

    def test_a_space_in_the_name_no_longer_truncates_the_path(self) -> None:
        text = f"这是什么页面 参考附件： - [图片] {self.reference}"

        self.assertEqual(
            extract_workspace_paths(text, workspace_root=self.root), [self.reference]
        )

    def test_prose_after_the_path_is_not_swallowed(self) -> None:
        text = f"{self.reference} 你看看这是什么"

        self.assertEqual(
            extract_workspace_paths(text, workspace_root=self.root), [self.reference]
        )

    def test_a_second_path_after_a_spaced_one_is_still_found(self) -> None:
        # 允许空格后，一次匹配可能横跨两条路径；扫描必须从裁剪后的终点继续。
        text = f"两个：{self.reference} 和 config/model_profiles.json"

        self.assertEqual(
            extract_workspace_paths(text, workspace_root=self.root),
            [self.reference, "config/model_profiles.json"],
        )

    def test_a_path_that_does_not_exist_still_resolves_to_its_first_segment(self) -> None:
        # 盘上没有的路径无从裁剪，行为必须和以前一致，不能把后面的话吞进来。
        text = "看一下 meet_files/不存在的文件.md 这个"

        self.assertEqual(
            extract_workspace_paths(text, workspace_root=self.root),
            ["meet_files/不存在的文件.md"],
        )

    def test_the_image_actually_reaches_a_vision_model(self) -> None:
        messages = [
            {
                "role": "user",
                "content": (
                    "这是什么页面\n\n参考附件：\n"
                    f"- [图片] 截屏2026-08-20 19.29.13.png: {self.reference}"
                ),
            }
        ]

        prepared = enrich_image_attachments_for_model(
            messages, self.profile(supports_vision=True), workspace_root=self.root
        )

        self.assertEqual(prepared.attached_count, 1)
        parts = prepared.messages[0]["content"]
        self.assertEqual([part["type"] for part in parts], ["text", "image_url"])
        self.assertTrue(parts[1]["image_url"]["url"].startswith("data:image/"))

    def test_a_text_only_model_reports_the_image_as_skipped(self) -> None:
        messages = [
            {
                "role": "user",
                "content": (
                    "这是什么页面\n\n参考附件：\n"
                    f"- [图片] 截屏2026-08-20 19.29.13.png: {self.reference}"
                ),
            }
        ]

        prepared = enrich_image_attachments_for_model(
            messages, self.profile(supports_vision=False), workspace_root=self.root
        )

        self.assertEqual(prepared.attached_count, 0)
        self.assertEqual(prepared.skipped_count, 1)


if __name__ == "__main__":
    unittest.main()
