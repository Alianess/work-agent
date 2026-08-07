from __future__ import annotations

from pathlib import Path

from work_agent_core.execution.backends import TrustedHostBackend
from work_agent_core.execution.models import BackendKind
from work_agent_core.execution.orchestrator import ExecutionOrchestrator
from work_agent_core.execution.policy import PolicyEngine
from work_agent_core.shell_tools import ShellExecutionTools


def trusted_shell_tools_for_test(
    workspace_root: str | Path,
    *,
    account_id: str = "local",
    turn_id: str = "",
    conversation_id: str = "",
) -> ShellExecutionTools:
    """Explicit test-only backend injection.

    Unit tests must not turn a missing macOS Seatbelt boundary into an implicit
    production fallback.  The production constructor still selects Seatbelt and
    fails closed; this helper names the deliberately non-isolated test backend.
    """
    root = Path(workspace_root).resolve()
    runner = ExecutionOrchestrator(
        workspace_root=root,
        execution_root=root / ".execution-test-data",
        policy=PolicyEngine(default_backend=BackendKind.TRUSTED_HOST),
        backends={BackendKind.TRUSTED_HOST: TrustedHostBackend()},
    )
    return ShellExecutionTools(
        root,
        execution_orchestrator=runner,
        account_id=account_id,
        turn_id=turn_id,
        conversation_id=conversation_id,
    )
