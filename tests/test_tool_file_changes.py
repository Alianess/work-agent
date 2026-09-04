from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from work_agent_core.tools import WorkspaceFiles, count_patch_file_changes


class ToolFileChangeTests(unittest.TestCase):
    def test_successful_write_notifies_file_index_handler(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            changed: list[Path] = []
            workspace = WorkspaceFiles(directory, on_file_changed=changed.append)

            workspace.write_text({"path": "meet_files/result.md", "content": "done"})

            self.assertEqual(
                changed,
                [Path(directory).resolve() / "meet_files" / "result.md"],
            )

    def test_directory_listing_carries_a_summary_and_truncation_note(self) -> None:
        """计数类问题靠统计行回答；截断必须说明，否则模型把前 N 个当全部。"""

        with tempfile.TemporaryDirectory() as directory:
            workspace = WorkspaceFiles(directory)
            (Path(directory) / "a.png").write_bytes(b"1")
            (Path(directory) / "b.png").write_bytes(b"2")
            (Path(directory) / "notes.md").write_text("x", encoding="utf-8")
            (Path(directory) / "sub").mkdir()
            (Path(directory) / "sub" / "c.docx").write_bytes(b"3")

            full = workspace.read_text({"path": directory, "max_files": 10})
            self.assertIn("共 4 个文件", full)
            self.assertIn("png×2", full)
            self.assertIn("sub/c.docx", full)  # 递归含子目录
            self.assertNotIn("未列出", full)

            truncated = workspace.read_text({"path": directory, "max_files": 2})
            self.assertIn("另有 2 个未列出", truncated)
            self.assertIn("a.png", truncated)
            self.assertNotIn("sub/c.docx", truncated)

    def test_counts_changes_per_file_in_unified_patch(self) -> None:
        patch = """--- a/web_frontend/src/App.tsx
+++ b/web_frontend/src/App.tsx
@@ -1,2 +1,3 @@
-old
+new
+extra
 keep
--- a/work_agent_core/web_server.py
+++ b/work_agent_core/web_server.py
@@ -1 +1 @@
-before
+after
"""

        self.assertEqual(
            count_patch_file_changes(patch),
            [
                {"file_path": "web_frontend/src/App.tsx", "additions": 2, "deletions": 1},
                {"file_path": "work_agent_core/web_server.py", "additions": 1, "deletions": 1},
            ],
        )


if __name__ == "__main__":
    unittest.main()
