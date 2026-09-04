from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from work_agent_core.config import ModelProfile
from work_agent_core.file_reference_index import PersistentFileReferenceIndex
from work_agent_core import web_server


class PersistentFileReferenceIndexTests(unittest.TestCase):
    def make_index(self, root: Path, index_path: Path) -> PersistentFileReferenceIndex:
        return PersistentFileReferenceIndex(
            workspace_root=root,
            index_path=index_path,
            scan_roots=[root / "meet_files"],
            is_visible=lambda path: path.suffix.lower() in {".png", ".md"},
            refresh_interval_seconds=3600,
        )

    def test_persisted_snapshot_is_reused_without_rescanning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            image = root / "meet_files" / "attachments" / "history.png"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"png")
            index_path = root / "state" / "file_reference_index.json"

            first = self.make_index(root, index_path)
            self.assertEqual(first.snapshot()["history.png"], ["meet_files/attachments/history.png"])
            self.assertTrue(index_path.is_file())

            second = self.make_index(root, index_path)
            with patch(
                "work_agent_core.file_reference_index.os.walk",
                side_effect=AssertionError("persisted index should not rescan"),
            ):
                second.warm_async()
                self.assertEqual(
                    second.snapshot()["history.png"],
                    ["meet_files/attachments/history.png"],
                )

    def test_index_never_crosses_its_account_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            first_root = base / "u1" / "workspace"
            second_root = base / "u2" / "workspace"
            first_file = first_root / "meet_files" / "attachments" / "private.png"
            second_file = second_root / "meet_files" / "attachments" / "private.png"
            first_file.parent.mkdir(parents=True)
            second_file.parent.mkdir(parents=True)
            first_file.write_bytes(b"u1")
            second_file.write_bytes(b"u2")

            index = self.make_index(first_root, base / "u1" / "file_reference_index.json")

            self.assertEqual(
                index.snapshot()["private.png"],
                ["meet_files/attachments/private.png"],
            )
            payload = (base / "u1" / "file_reference_index.json").read_text(encoding="utf-8")
            self.assertNotIn(str(second_root), payload)

    def test_attachment_visibility_uses_explicit_account_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "member-workspace"
            image = root / "meet_files" / "attachments" / "private.png"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"member")

            self.assertTrue(
                web_server.is_file_library_visible(image, workspace_root=root)
            )

    def test_known_write_updates_and_delete_removes_persistent_entry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            index = self.make_index(root, root / "state" / "file_reference_index.json")
            self.assertEqual(index.snapshot(), {})
            image = root / "meet_files" / "attachments" / "new.png"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"new")

            index.upsert(image)
            self.assertEqual(index.snapshot()["new.png"], ["meet_files/attachments/new.png"])

            image.unlink()
            index.remove(image)
            self.assertNotIn("new.png", index.snapshot())


class HistoricalVisionContextTests(unittest.TestCase):
    @staticmethod
    def profile(*, supports_vision: bool = True) -> ModelProfile:
        return ModelProfile(
            name="vision-test",
            provider="openai-compatible",
            base_url="https://api.example.com/v1",
            model="vision-test",
            api_key_env="TEST_KEY",
            supports_vision=supports_vision,
        )

    def test_a_typed_filename_is_text_and_never_loads_the_file_index(self) -> None:
        # 光提文件名不是附件：不附图，也不碰文件索引。
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            image = root / "meet_files" / "attachments" / "history.png"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"png")
            messages = [
                {"role": "user", "content": "请看 history.png"},
                {"role": "assistant", "content": "我看不到它。"},
                {"role": "user", "content": "继续看 history.png 的右下角"},
            ]

            with patch.object(
                web_server,
                "visible_file_reference_index",
                side_effect=AssertionError("filename mentions should not load the file index"),
            ):
                prepared = web_server.enrich_image_attachments_for_model(
                    messages,
                    self.profile(),
                    workspace_root=root,
                )

            self.assertEqual(prepared.attached_count, 0)
            self.assertEqual(prepared.messages, messages)

    def test_later_question_keeps_original_image_pixels_in_model_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            image = root / "meet_files" / "attachments" / "history.png"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"png")
            messages = [
                {
                    "role": "user",
                    "content": "图片\n\n参考附件：\n- [图片] history.png: meet_files/attachments/history.png",
                },
                {"role": "assistant", "content": "已经看到了。"},
                {"role": "user", "content": "右下角那个很小的图标是什么？"},
            ]

            prepared = web_server.enrich_image_attachments_for_model(
                messages,
                self.profile(),
                workspace_root=root,
            )

            first_content = prepared.messages[0]["content"]
            self.assertIsInstance(first_content, list)
            self.assertTrue(first_content[1]["image_url"]["url"].startswith("data:image/png;base64,"))
            self.assertEqual(prepared.messages[2], messages[2])

    def test_compacted_history_does_not_rehydrate_conversation_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            image = root / "meet_files" / "attachments" / "history.png"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"png")
            prepared = web_server.enrich_image_attachments_for_model(
                [{"role": "user", "content": "右下角的小图标是什么？"}],
                self.profile(),
                workspace_root=root,
            )

            self.assertEqual(prepared.messages[0]["content"], "右下角的小图标是什么？")
            self.assertEqual(prepared.attached_count, 0)

    def test_an_uncompressed_old_image_stays_on_its_original_message(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            image = root / "meet_files" / "attachments" / "history.png"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"png")
            messages = [
                {
                    "role": "user",
                    "content": "图片\n\n参考附件：\n- [图片] history.png: meet_files/attachments/history.png",
                },
                {"role": "assistant", "content": "已经看到了。"},
                {"role": "user", "content": "第二轮"},
                {"role": "assistant", "content": "好。"},
                {"role": "user", "content": "第三轮继续。"},
            ]
            prepared = web_server.enrich_image_attachments_for_model(
                messages,
                self.profile(),
                workspace_root=root,
            )

            self.assertIsInstance(prepared.messages[0]["content"], list)
            self.assertEqual(prepared.messages[0]["content"][1]["type"], "image_url")
            self.assertEqual(prepared.messages[-1], messages[-1])
            self.assertEqual(prepared.attached_count, 1)

    def test_plain_text_history_does_not_touch_the_file_index(self) -> None:
        messages = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好。"},
            {"role": "user", "content": "继续"},
        ]

        with patch.object(
            web_server,
            "visible_file_reference_index",
            side_effect=AssertionError("plain text should not load the file index"),
        ):
            prepared = web_server.enrich_image_attachments_for_model(
                messages,
                self.profile(supports_vision=False),
            )

        self.assertEqual(prepared.messages, messages)


class ImagePayloadTests(unittest.TestCase):
    """图片是请求里最贵的部分，两处浪费都实测撞过 TPM 上限。"""

    @staticmethod
    def _photo(path, size=(4000, 3000)):
        from PIL import Image

        Image.new("RGB", size, (120, 140, 160)).save(path, format="JPEG", quality=95)
        return path

    def test_a_large_photo_is_downscaled_before_encoding(self) -> None:
        import base64

        with tempfile.TemporaryDirectory() as directory:
            path = self._photo(Path(directory) / "photo.jpeg")
            raw = len(base64.b64encode(path.read_bytes()))

            mime, encoded = web_server.encode_image_for_model(path, "image/jpeg")

            self.assertEqual(mime, "image/jpeg")
            self.assertTrue(encoded)
            # 模型端用不到 4000px，多出来的分辨率只是账单
            self.assertLess(len(encoded), raw / 2)

    def test_a_small_image_is_sent_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._photo(Path(directory) / "small.jpeg", size=(320, 240))
            mime, encoded = web_server.encode_image_for_model(path, "image/jpeg")
            self.assertEqual(mime, "image/jpeg")
            self.assertTrue(encoded)

    def test_an_unreadable_file_yields_nothing_rather_than_raising(self) -> None:
        mime, encoded = web_server.encode_image_for_model(Path("/nope/missing.jpeg"), "image/jpeg")
        self.assertEqual((mime, encoded), ("", ""))

    def test_read_file_hands_an_image_to_the_loop_not_to_the_tool_result(self) -> None:
        """一个 read 吃下文本和图片——和 Claude、Pi 一致。

        图片不能进 tool 结果（OpenAI 兼容的 tool 消息只能是字符串），所以它走
        附件通道由 harness 注入。对模型来说仍然只有一个 read_file。
        """

        from work_agent_core.progress import set_tool_attachment_sink
        from work_agent_core.tools import WorkspaceFiles

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._photo(root / "note.jpeg", size=(2400, 1800))
            (root / "note.md").write_text("这是文本。", encoding="utf-8")
            files = WorkspaceFiles(root)
            blocks: list[dict] = []
            previous = set_tool_attachment_sink(blocks.append)
            try:
                image_result = files.read_text({"path": "note.jpeg"})
                text_result = files.read_text({"path": "note.md"})
            finally:
                set_tool_attachment_sink(previous)

            self.assertEqual(text_result, "这是文本。")
            self.assertEqual(len(blocks), 1)
            self.assertEqual(blocks[0]["type"], "image_url")
            self.assertTrue(blocks[0]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
            self.assertIn("下一步", image_result)

    def test_reading_an_image_says_so_when_nothing_can_receive_it(self) -> None:
        from work_agent_core.tools import WorkspaceFiles

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._photo(root / "note.jpeg", size=(600, 400))

            result = WorkspaceFiles(root).read_text({"path": "note.jpeg"})

            # 不静默丢弃：说清楚这张图没有进入本次请求
            self.assertIn("未进入本次请求", result)

    def test_reading_a_missing_file_reports_it(self) -> None:
        from work_agent_core.tools import WorkspaceFiles

        with tempfile.TemporaryDirectory() as directory:
            self.assertIn("没有这个文件", WorkspaceFiles(Path(directory)).read_text({"path": "nope.md"}))

if __name__ == "__main__":
    unittest.main()
