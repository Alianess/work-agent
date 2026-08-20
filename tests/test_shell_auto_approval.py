from __future__ import annotations

import json
import tempfile
import unittest

from pathlib import Path

from work_agent_core.shell_tools import ShellExecutionTools, approval_action_id, issue_internal_approval_grant
from tests.execution_test_support import trusted_shell_tools_for_test


class ShellAutoApprovalTests(unittest.TestCase):
    def test_workspace_artifact_command_is_delegatable(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            payload = json.loads(
                trusted_shell_tools_for_test(workspace).execute({"command": "mkdir generated"})
            )

        self.assertEqual(payload["status"], "approval_required")
        self.assertTrue(payload["auto_approvable"])
        self.assertTrue(payload["reviewable_by_model"])

    def test_general_python_script_needs_a_grant_but_a_reviewer_may_issue_it(self) -> None:
        """The delegate and the reviewer are deliberately different boundaries.

        A script under the project's own interpreter is bounded by the sandbox
        and the workspace check, so refusing to let the reviewer clear it made
        "review for me" unable to handle ordinary work. It still never runs
        without some grant.
        """

        with tempfile.TemporaryDirectory() as workspace:
            payload = json.loads(
                trusted_shell_tools_for_test(workspace).execute({"command": "python script.py"})
            )

        self.assertEqual(payload["status"], "approval_required")
        self.assertFalse(payload["auto_approvable"])
        self.assertTrue(payload["reviewable_by_model"])

    def test_package_install_is_not_delegatable(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            payload = json.loads(
                trusted_shell_tools_for_test(workspace).execute({"command": "npm install"})
            )

        self.assertEqual(payload["status"], "approval_required")
        self.assertFalse(payload["auto_approvable"])
        self.assertFalse(payload["reviewable_by_model"])

    def test_public_approved_by_user_flag_cannot_bypass_policy(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            payload = json.loads(
                trusted_shell_tools_for_test(workspace).execute(
                    {"command": "mkdir generated", "approved_by_user": True}
                )
            )

        self.assertEqual(payload["status"], "approval_required")

    def test_internal_grant_is_bound_to_exact_action(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            tools = trusted_shell_tools_for_test(workspace)
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
                trusted_shell_tools_for_test(workspace).execute(
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
                trusted_shell_tools_for_test(workspace).execute({"command": "unknown-tool /etc/passwd"})
            )

        self.assertEqual(payload["status"], "denied")
        self.assertEqual(payload["risk_category"], "SYSTEM")

    def test_single_scoped_file_delete_requires_explicit_user_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            target = Path(workspace) / "generated.txt"
            target.write_text("temporary", encoding="utf-8")
            tools = trusted_shell_tools_for_test(workspace)
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
                trusted_shell_tools_for_test(workspace).execute({"command": "rm first.txt second.txt"})
            )

        self.assertEqual(payload["status"], "approval_required")
        self.assertEqual(payload["risk_category"], "DELETE")
        self.assertFalse(payload["reviewable_by_model"])

    def test_find_side_effect_predicates_are_hard_denied(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            tools = trusted_shell_tools_for_test(workspace)
            payload = json.loads(tools.execute({"command": "find . -delete"}))

        self.assertEqual(payload["status"], "denied")
        self.assertEqual(payload["risk_category"], "DELETE")

    def test_read_only_find_remains_auto_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            payload = json.loads(
                trusted_shell_tools_for_test(workspace).execute({"command": "find . -name '*.txt'"})
            )

        self.assertEqual(payload["status"], "executed")
        self.assertEqual(payload["permission"], "allow")

    def test_native_tool_call_id_is_idempotent_within_a_turn(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            tools = trusted_shell_tools_for_test(
                workspace,
                account_id="account-1",
                turn_id="turn-1",
            )
            first = json.loads(
                tools.execute({"command": "find . -name '*.txt'", "_execution_tool_call_id": "call-1"})
            )
            second = json.loads(
                tools.execute({"command": "find . -name '*.txt'", "_execution_tool_call_id": "call-1"})
            )

        self.assertTrue(first["ok"])
        self.assertTrue(second["ok"])
        self.assertEqual(first["execution_id"], second["execution_id"])

    def test_recursive_or_broad_delete_is_hard_denied(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            tools = trusted_shell_tools_for_test(workspace)
            recursive = json.loads(tools.execute({"command": "rm -rf generated"}))
            wildcard = json.loads(tools.execute({"command": "rm *.tmp"}))
            root = json.loads(tools.execute({"command": "rm ."}))

        for payload in (recursive, wildcard, root):
            self.assertEqual(payload["status"], "denied")
            self.assertEqual(payload["risk_category"], "DELETE")

    def test_delete_outside_workspace_is_hard_denied(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            payload = json.loads(
                trusted_shell_tools_for_test(workspace).execute({"command": "rm /tmp/outside.txt"})
            )

        self.assertEqual(payload["status"], "denied")
        self.assertEqual(payload["risk_category"], "SYSTEM")

    def test_command_naming_an_unsnapshotted_directory_is_flagged(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            (Path(workspace) / "meet_files/attachments").mkdir(parents=True)
            tools = trusted_shell_tools_for_test(workspace)
            quoted = json.loads(
                tools.execute(
                    {
                        "command": (
                            "python -c \"import pdfplumber; "
                            "pdfplumber.open('meet_files/attachments/a.pdf')\""
                        )
                    }
                )
            )
            unrelated = json.loads(tools.execute({"command": "python script.py"}))

        self.assertIn("meet_files", quoted["isolated_workspace_note"])
        self.assertNotIn("isolated_workspace_note", unrelated)


if __name__ == "__main__":
    unittest.main()


class InlinePythonReadOnlyTests(unittest.TestCase):
    """A policy that only sees the program name charges a click per probe."""

    def test_read_only_probes_run_without_asking(self) -> None:
        commands = [
            '.venv/bin/python -c "import pypdfium2; print(\'ok\')"',
            'python -c "import importlib.util; print(importlib.util.find_spec(\'x\'))"',
            'python -c "import json; print(json.load(open(\'a.json\')))"',
        ]
        with tempfile.TemporaryDirectory() as workspace:
            tools = trusted_shell_tools_for_test(workspace)
            for command in commands:
                payload = json.loads(tools.execute({"command": command}))
                self.assertNotEqual(payload["status"], "approval_required", command)
                # READ keeps delivery on discard, so a misjudgement cannot persist.
                self.assertEqual(payload["risk_category"], "READ", command)

    def test_snippets_with_effects_still_require_approval(self) -> None:
        commands = [
            'python -c "open(\'x.txt\',\'w\').write(\'boom\')"',
            'python -c "import subprocess; subprocess.run([\'ls\'])"',
            'python -c "import os; os.remove(\'a\')"',
            'python -c "import urllib.request; urllib.request.urlopen(\'http://x\')"',
            'python -c "exec(open(\'evil.py\').read())"',
            'python -c "import json; json.dump({}, open(\'a.json\',\'w\'))"',
            'python -c "def ("',
        ]
        with tempfile.TemporaryDirectory() as workspace:
            tools = trusted_shell_tools_for_test(workspace)
            for command in commands:
                payload = json.loads(tools.execute({"command": command}))
                self.assertEqual(payload["status"], "approval_required", command)

    def test_reviewer_covers_project_scripts_but_not_installs_or_unknown_binaries(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            tools = trusted_shell_tools_for_test(workspace)
            reviewable = {
                command: json.loads(tools.execute({"command": command})).get("reviewable_by_model")
                for command in (
                    "python script.py",
                    "node build.js",
                    "pip install requests",
                    "npm install",
                    "pdftotext a.pdf -",
                )
            }
        self.assertTrue(reviewable["python script.py"])
        self.assertTrue(reviewable["node build.js"])
        self.assertFalse(reviewable["pip install requests"])
        self.assertFalse(reviewable["npm install"])
        self.assertFalse(reviewable["pdftotext a.pdf -"])


class ShellPipelineTests(unittest.TestCase):
    """The parser is not the boundary; the sandbox is."""

    def test_pipes_chains_and_globs_run(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            Path(workspace, "a.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
            tools = trusted_shell_tools_for_test(workspace)
            for command, expected in (
                ("cat a.txt | wc -l", "3"),
                ("ls | head -1", "a.txt"),
                ("echo hi && echo there", "hi"),
            ):
                payload = json.loads(tools.execute({"command": command}))
                self.assertEqual(payload["status"], "executed", command)
                self.assertIn(expected, payload["stdout"], command)

    def test_every_stage_is_checked_not_just_the_first(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            tools = trusted_shell_tools_for_test(workspace)
            payload = json.loads(tools.execute({"command": "echo hi | sudo tee /etc/passwd"}))
        self.assertEqual(payload["status"], "denied")

    def test_redirect_outside_the_workspace_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            tools = trusted_shell_tools_for_test(workspace)
            payload = json.loads(tools.execute({"command": "ls > /tmp/escape-probe.txt"}))
        self.assertEqual(payload["status"], "denied")
        self.assertIn("工作区之外", payload["reason"])

    def test_command_substitution_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            tools = trusted_shell_tools_for_test(workspace)
            for command in ("cat $(echo a.txt)", "cat `echo a.txt`"):
                payload = json.loads(tools.execute({"command": command}))
                self.assertEqual(payload["status"], "denied", command)

    def test_redirect_into_the_workspace_still_asks(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            tools = trusted_shell_tools_for_test(workspace)
            tools.sandbox_auto_allow = False
            payload = json.loads(tools.execute({"command": "ls > listing.txt"}))
        self.assertEqual(payload["status"], "approval_required")
