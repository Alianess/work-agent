from __future__ import annotations

import unittest

from work_agent_core.tools import count_patch_file_changes


class ToolFileChangeTests(unittest.TestCase):
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
