from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from work_agent_core import web_server
from work_agent_core.auth import AuthUser
from work_agent_core.execution.backends import TrustedHostBackend
from work_agent_core.execution.models import BackendKind, CommandSpec, ExecutionClass, ExecutionMode, ExecutionRequest
from work_agent_core.execution.orchestrator import ExecutionOrchestrator
from work_agent_core.execution.policy import PolicyEngine


class ExecutionApiReadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name).resolve()
        self.user = AuthUser(id=42, username="execution-test", role="member", created_at=0)
        self.patches = [
            patch.object(web_server, "account_workspace_root", return_value=self.root),
            patch.object(web_server, "current_auth_user", return_value=self.user),
        ]
        for item in self.patches:
            item.start()
        self.runner = ExecutionOrchestrator(
            workspace_root=self.root,
            execution_root=self.root / "meet_files" / "execution",
            policy=PolicyEngine(default_backend=BackendKind.TRUSTED_HOST),
            backends={BackendKind.TRUSTED_HOST: TrustedHostBackend()},
        )

    def tearDown(self) -> None:
        for item in reversed(self.patches):
            item.stop()
        self.temporary_directory.cleanup()

    def test_account_scoped_execution_read_endpoints_return_durable_receipt_events_and_changes(self) -> None:
        result = self.runner.submit(
            ExecutionRequest(
                request_id="request-api",
                idempotency_key="idempotency-api",
                account_id="42",
                turn_id="turn-api",
                conversation_id="conversation-api",
                project_id="",
                tool_call_id="call-api",
                tool_name="shell_exec",
                execution_class=ExecutionClass.ISOLATED_PROCESS,
                mode=ExecutionMode.ISOLATED,
                command=CommandSpec(
                    argv=(
                        "/usr/bin/python3",
                        "-c",
                        "from pathlib import Path; Path('proof.txt').write_text('ok', encoding='utf-8')",
                    )
                ),
            ),
            source_root=self.root,
        )

        detail = web_server.execution_payload(result.execution_id)
        events = web_server.execution_events_payload(result.execution_id)
        changes = web_server.execution_changes_payload(result.execution_id)
        receipt = web_server.execution_receipt_payload(result.execution_id)

        self.assertTrue(detail["ok"])
        self.assertEqual(detail["execution"]["account_id"], "42")
        self.assertEqual(detail["execution"]["execution_id"], result.execution_id)
        self.assertGreaterEqual(len(events["events"]), 3)
        self.assertTrue(receipt["receipt"])
        self.assertEqual(changes["change_set"]["changes"][0]["path"], "proof.txt")

    def test_execution_route_accepts_only_strict_execution_ids(self) -> None:
        self.assertEqual(
            web_server.parse_execution_route("/api/executions/exe_" + "a" * 32 + "/events"),
            ("exe_" + "a" * 32, "events"),
        )
        self.assertIsNone(web_server.parse_execution_route("/api/executions/../../etc/passwd"))
        self.assertIsNone(web_server.parse_execution_route("/api/executions/exe_short/receipt"))


if __name__ == "__main__":
    unittest.main()
