from __future__ import annotations

import json
import tempfile
import unittest

from pathlib import Path

from work_agent_core.shell_tools import ShellExecutionTools, approval_action_id, issue_internal_approval_grant


class ShellAutoApprovalTests(unittest.TestCase):
    def test_workspace_artifact_command_is_delegatable(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            payload = json.loads(
                ShellExecutionTools(workspace).execute({"command": "mkdir generated"})
            )

        self.assertEqual(payload["status"], "approval_required")
        self.assertTrue(payload["auto_approvable"])
        self.assertTrue(payload["reviewable_by_model"])

    def test_general_python_script_is_not_delegatable(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            payload = json.loads(
                ShellExecutionTools(workspace).execute({"command": "python script.py"})
            )

        self.assertEqual(payload["status"], "approval_required")
        self.assertFalse(payload["auto_approvable"])
        self.assertFalse(payload["reviewable_by_model"])

    def test_package_install_is_not_delegatable(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            payload = json.loads(
                ShellExecutionTools(workspace).execute({"command": "npm install"})
            )

        self.assertEqual(payload["status"], "approval_required")
        self.assertFalse(payload["auto_approvable"])
        self.assertFalse(payload["reviewable_by_model"])

    def test_public_approved_by_user_flag_cannot_bypass_policy(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            payload = json.loads(
                ShellExecutionTools(workspace).execute(
                    {"command": "mkdir generated", "approved_by_user": True}
                )
            )

        self.assertEqual(payload["status"], "approval_required")

    def test_internal_grant_is_bound_to_exact_action(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            tools = ShellExecutionTools(workspace)
            wrong = json.loads(
                tools.execute(
                    {
                        "command": "mkdir generated",
                        "_approval_source": "reviewer",
                        "_approval_action_id": "approval-wrong",
                    }
                )
            )
            action_id = approval_action_id(
                command="mkdir generated",
                cwd=str(Path(workspace).resolve()),
                timeout_seconds=120,
            )
            approved = json.loads(
                tools.execute(
                    {
                        "command": "mkdir generated",
                        "_approval_source": "reviewer",
                        "_approval_action_id": action_id,
                        "_approval_grant": issue_internal_approval_grant(
                            action_id=action_id,
                            source="reviewer",
                        ),
                    }
                )
            )

        self.assertEqual(wrong["status"], "approval_required")
        self.assertTrue(approved["ok"])

    def test_model_supplied_action_id_cannot_forge_an_internal_grant(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            action_id = approval_action_id(
                command="mkdir generated",
                cwd=str(Path(workspace).resolve()),
                timeout_seconds=120,
            )
            payload = json.loads(
                ShellExecutionTools(workspace).execute(
                    {
                        "command": "mkdir generated",
                        "_approval_source": "user",
                        "_approval_action_id": action_id,
                    }
                )
            )

        self.assertEqual(payload["status"], "approval_required")

    def test_unknown_command_cannot_reference_path_outside_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            payload = json.loads(
                ShellExecutionTools(workspace).execute({"command": "unknown-tool /etc/passwd"})
            )

        self.assertEqual(payload["status"], "denied")
        self.assertEqual(payload["risk_category"], "SYSTEM")

    def test_single_scoped_file_delete_requires_explicit_user_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            target = Path(workspace) / "generated.txt"
            target.write_text("temporary", encoding="utf-8")
            tools = ShellExecutionTools(workspace)
            result = json.loads(tools.execute({"command": "rm generated.txt"}))

            self.assertEqual(result["status"], "approval_required")
            self.assertEqual(result["risk_category"], "DELETE")
            self.assertFalse(result["reviewable_by_model"])
            self.assertTrue(target.exists())

    def test_multiple_scoped_file_deletes_are_reviewable(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            first = Path(workspace) / "first.txt"
            second = Path(workspace) / "second.txt"
            first.write_text("first", encoding="utf-8")
            second.write_text("second", encoding="utf-8")
            payload = json.loads(
                ShellExecutionTools(workspace).execute({"command": "rm first.txt second.txt"})
            )

        self.assertEqual(payload["status"], "approval_required")
        self.assertEqual(payload["risk_category"], "DELETE")
        self.assertFalse(payload["reviewable_by_model"])

    def test_find_side_effect_predicates_are_hard_denied(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            tools = ShellExecutionTools(workspace)
            payload = json.loads(tools.execute({"command": "find . -delete"}))

        self.assertEqual(payload["status"], "denied")
        self.assertEqual(payload["risk_category"], "DELETE")

    def test_read_only_find_remains_auto_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            payload = json.loads(
                ShellExecutionTools(workspace).execute({"command": "find . -name '*.txt'"})
            )

        self.assertEqual(payload["status"], "executed")
        self.assertEqual(payload["permission"], "allow")

    def test_recursive_or_broad_delete_is_hard_denied(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            tools = ShellExecutionTools(workspace)
            recursive = json.loads(tools.execute({"command": "rm -rf generated"}))
            wildcard = json.loads(tools.execute({"command": "rm *.tmp"}))
            root = json.loads(tools.execute({"command": "rm ."}))

        for payload in (recursive, wildcard, root):
            self.assertEqual(payload["status"], "denied")
            self.assertEqual(payload["risk_category"], "DELETE")

    def test_delete_outside_workspace_is_hard_denied(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            payload = json.loads(
                ShellExecutionTools(workspace).execute({"command": "rm /tmp/outside.txt"})
            )

        self.assertEqual(payload["status"], "denied")
        self.assertEqual(payload["risk_category"], "SYSTEM")


if __name__ == "__main__":
    unittest.main()
