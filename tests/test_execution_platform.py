from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from work_agent_core.execution.backends import TrustedHostBackend
from work_agent_core.execution.models import (
    BackendKind,
    CapabilitySet,
    CommandSpec,
    ExecutionClass,
    ExecutionMode,
    ExecutionRequest,
    ExecutionStatus,
    NetworkScope,
    PermissionDecision,
    PermissionDecisionValue,
)
from work_agent_core.execution.orchestrator import ExecutionOrchestrator
from work_agent_core.execution.policy import PolicyEngine


class ExecutionPlatformTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name).resolve()
        self.execution_root = self.root / "meet_files" / "execution"
        self.orchestrator = ExecutionOrchestrator(
            workspace_root=self.root,
            execution_root=self.execution_root,
            policy=PolicyEngine(default_backend=BackendKind.TRUSTED_HOST),
            backends={BackendKind.TRUSTED_HOST: TrustedHostBackend()},
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def request(
        self,
        *,
        key: str,
        command: CommandSpec,
        capabilities: CapabilitySet | None = None,
        delivery_mode: str = "apply_after_validation",
    ) -> ExecutionRequest:
        return ExecutionRequest(
            request_id=f"request-{key}",
            idempotency_key=f"idempotency-{key}",
            account_id="test-account",
            turn_id="turn-test",
            conversation_id="conversation-test",
            project_id="project-test",
            tool_call_id=f"call-{key}",
            tool_name="test_execution",
            execution_class=ExecutionClass.ISOLATED_PROCESS,
            mode=ExecutionMode.ISOLATED,
            command=command,
            requested_capabilities=capabilities or CapabilitySet(),
            delivery_mode=delivery_mode,
            reason="执行平台测试",
        )

    def test_private_snapshot_applies_verified_file_change_without_recursing_execution_data(self) -> None:
        (self.root / "source.txt").write_text("before\n", encoding="utf-8")
        result = self.orchestrator.submit(
            self.request(
                key="snapshot",
                command=CommandSpec(
                    argv=(
                        "/usr/bin/python3",
                        "-c",
                        "from pathlib import Path; Path('result.txt').write_text('after\\n', encoding='utf-8')",
                    )
                ),
            ),
            source_root=self.root,
        )

        self.assertEqual(result.status, ExecutionStatus.SUCCEEDED)
        self.assertEqual((self.root / "result.txt").read_text(encoding="utf-8"), "after\n")
        self.assertEqual(result.delivery_status.value, "applied")
        self.assertTrue(result.change_set_id)
        self.assertTrue((self.execution_root / "receipts").exists() or self.orchestrator.store.receipt(result.execution_id))

    def test_domain_permission_is_recorded_then_fails_closed_without_a_network_broker(self) -> None:
        orchestrator = ExecutionOrchestrator(
            workspace_root=self.root,
            execution_root=self.execution_root,
            policy=PolicyEngine(),
        )
        request = self.request(
            key="network",
            command=CommandSpec(argv=("/usr/bin/python3", "-c", "print('ok')")),
            capabilities=CapabilitySet(network=NetworkScope(mode="domain_allowlist", allowed_domains=("pypi.org",))),
        )
        waiting = orchestrator.submit(request, source_root=self.root)

        self.assertEqual(waiting.status, ExecutionStatus.WAITING_PERMISSION)
        record = orchestrator.store.get(waiting.execution_id)
        permission_events = [event for event in orchestrator.store.events(waiting.execution_id) if event["type"] == "permission.requested"]
        self.assertEqual(len(permission_events), 1)
        permission = permission_events[0]["payload"]["permission_request"]
        resumed = orchestrator.resume_after_permission(
            waiting.execution_id,
            PermissionDecision(
                permission_request_id=permission["permission_request_id"],
                decision=PermissionDecisionValue.ALLOW_ONCE,
                decided_by="user:test-account",
                decided_at_ms=1,
                expected_contract_digest=record.contract["digest"],
                client_nonce="test-nonce",
            ),
            source_root=self.root,
        )

        self.assertEqual(resumed.status, ExecutionStatus.FAILED)
        self.assertEqual(resumed.error.code, "NETWORK_BROKER_UNAVAILABLE")
        stored_permission = orchestrator.store.permission(permission["permission_request_id"])
        self.assertEqual(stored_permission["status"], "allowed")

    def test_apply_conflict_never_overwrites_newer_source_file(self) -> None:
        (self.root / "source.txt").write_text("before\n", encoding="utf-8")
        result = self.orchestrator.submit(
            self.request(
                key="conflict",
                command=CommandSpec(
                    argv=(
                        "/usr/bin/python3",
                        "-c",
                        "from pathlib import Path; Path('source.txt').write_text('sandbox\\n', encoding='utf-8')",
                    )
                ),
                delivery_mode="review_then_apply",
            ),
            source_root=self.root,
        )
        self.assertEqual(result.status, ExecutionStatus.AWAITING_APPLY)
        self.assertEqual((self.root / "source.txt").read_text(encoding="utf-8"), "before\n")
        (self.root / "source.txt").write_text("newer source\n", encoding="utf-8")
        deferred = self.orchestrator.apply_changes(
            result.execution_id,
            change_set_id=str(result.change_set_id),
            expected_digest=self.orchestrator.workspace.load_change_set(str(result.change_set_id)).digest,
        )
        self.assertEqual(deferred.status, ExecutionStatus.FAILED)
        self.assertEqual(deferred.error.code, "WORKSPACE_CONFLICT")
        self.assertEqual((self.root / "source.txt").read_text(encoding="utf-8"), "newer source\n")

    def test_cancel_kills_process_before_delayed_side_effect(self) -> None:
        result = self.orchestrator.submit(
            self.request(
                key="cancel",
                command=CommandSpec(
                    argv=(
                        "/usr/bin/python3",
                        "-c",
                        "import time; time.sleep(5); open('should-not-exist.txt', 'w').write('no')",
                    )
                ),
            ),
            source_root=self.root,
            cancel_check=lambda: True,
        )

        self.assertEqual(result.status, ExecutionStatus.CANCELLED)
        self.assertFalse((self.root / "should-not-exist.txt").exists())

    def test_default_policy_selects_native_seatbelt(self) -> None:
        policy = PolicyEngine()
        decision = policy.evaluate(
            self.request(key="default", command=CommandSpec(argv=("/usr/bin/true",)))
        )
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.backend, BackendKind.MACOS_SEATBELT)


if __name__ == "__main__":
    unittest.main()
