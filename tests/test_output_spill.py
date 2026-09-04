"""截断必须留有回头路：模型能知道自己没看到什么，并且能拿到。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from work_agent_core.shell_tools import spill_output
from work_agent_core.tools import WorkspaceFiles
from work_agent_core import web_server


class ShellOutputSpillTests(unittest.TestCase):
    def test_output_within_limit_is_returned_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            text, path = spill_output(
                "short",
                20000,
                workspace_root=Path(directory),
                execution_id="exec-1",
                stream="stdout",
            )
            self.assertEqual(text, "short")
            self.assertEqual(path, "")
            self.assertFalse((Path(directory) / "tmp").exists())

    def test_oversized_output_is_written_whole_and_path_returned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = "x" * 50000

            text, path = spill_output(
                payload,
                20000,
                workspace_root=root,
                execution_id="exec-1",
                stream="stdout",
            )

            self.assertEqual(path, "tmp/shell_output/exec-1.stdout.txt")
            self.assertEqual((root / path).read_text(encoding="utf-8"), payload)
            self.assertIn(path, text)
            self.assertIn("30000", text)

    def test_unwritable_spill_still_returns_the_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # A file where the spill directory should go makes mkdir fail.
            (root / "tmp").write_text("not a directory", encoding="utf-8")

            text, path = spill_output(
                "y" * 50000,
                20000,
                workspace_root=root,
                execution_id="exec-2",
                stream="stderr",
            )

            self.assertEqual(path, "")
            self.assertTrue(text.startswith("y"))
            self.assertIn("truncated", text)


class ReadTextOffsetTests(unittest.TestCase):
    def test_short_file_has_no_position_note(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.txt").write_text("hello", encoding="utf-8")

            self.assertEqual(WorkspaceFiles(root).read_text({"path": "a.txt"}), "hello")

    def test_paging_covers_the_whole_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = "".join(str(index % 10) for index in range(30000))
            (root / "a.txt").write_text(payload, encoding="utf-8")
            files = WorkspaceFiles(root)

            collected = ""
            offset = 0
            for _ in range(10):
                chunk = files.read_text({"path": "a.txt", "offset": offset})
                body, _, note = chunk.rpartition("\n\n[")
                collected += body
                if "read again with offset=" not in note:
                    break
                offset = int(note.split("read again with offset=")[1].rstrip("]"))

            self.assertEqual(collected, payload)

    def test_first_page_reports_how_much_remains(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.txt").write_text("y" * 30000, encoding="utf-8")

            result = WorkspaceFiles(root).read_text({"path": "a.txt"})

            self.assertIn("chars 0-12000 of 30000", result)
            self.assertIn("read again with offset=12000", result)


class SkillCatalogTests(unittest.TestCase):
    def test_prompt_keeps_only_skill_ids_and_descriptions(self) -> None:
        block = web_server.render_chat_skill_catalog()

        self.assertIn("- meeting-minutes：", block)
        self.assertIn("- anysearch：", block)
        self.assertNotIn("任务明确匹配某技能时", block)
        self.assertNotIn("无法判断对应技能时", block)
        # when_to_use remains lazy and is available through list/open.
        self.assertNotIn("当用户提到联网、Web、网页", block)
        self.assertNotIn("mention", block)
        self.assertNotIn('"enabled"', block)


if __name__ == "__main__":
    unittest.main()
