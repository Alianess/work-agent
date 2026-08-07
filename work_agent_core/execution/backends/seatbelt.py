from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

from ..errors import failure
from ..models import BackendKind, CommandSpec, ExecutionContract
from ...runtime_env import apply_project_agent_environment, runtime_bin_directories, runtime_search_path
from .base import BackendHealth, ExecutionEnvironment, ProcessExecutionBackend


class SeatbeltBackend(ProcessExecutionBackend):
    """Native macOS isolated execution backend with no background daemon."""

    kind = BackendKind.MACOS_SEATBELT

    def __init__(
        self,
        sandbox_exec: str = "/usr/bin/sandbox-exec",
        *,
        runtime_workspace_root: str | Path | None = None,
    ) -> None:
        self.sandbox_exec = sandbox_exec
        self.runtime_workspace_root = Path(runtime_workspace_root or Path.cwd()).resolve()

    def health(self) -> BackendHealth:
        executable = shutil.which(self.sandbox_exec) if "/" not in self.sandbox_exec else self.sandbox_exec
        if not executable or not Path(executable).is_file():
            return BackendHealth(False, "未找到 macOS Seatbelt sandbox-exec。")
        probe = subprocess.run(
            [executable, "-p", "(version 1) (deny default) (allow process-exec)", "/usr/bin/true"],
            capture_output=True,
            text=True,
            timeout=4,
            check=False,
        )
        if probe.returncode != 0:
            detail = (probe.stderr or probe.stdout or "Seatbelt 不可用").strip()
            return BackendHealth(False, detail[-500:])
        return BackendHealth(True, "macOS Seatbelt 可用")

    def prepare(
        self,
        contract: ExecutionContract,
        *,
        workspace_path: Path,
        log_dir: Path,
    ) -> ExecutionEnvironment:
        health = self.health()
        if not health.available:
            raise failure(
                "BACKEND_UNAVAILABLE",
                f"macOS Seatbelt 不可用：{health.detail}。系统不会改用宿主执行。",
                retryable=True,
                phase="preparing",
                user_action="repair_backend",
            )
        environment = super().prepare(contract, workspace_path=workspace_path, log_dir=log_dir)
        profile_path = environment.log_dir / "seatbelt.sb"
        profile_path.write_text(self._profile(workspace_path, self._runtime_read_roots()), encoding="utf-8")
        return ExecutionEnvironment(
            environment_id=environment.environment_id,
            backend=environment.backend,
            workspace_path=environment.workspace_path,
            log_dir=environment.log_dir,
            backend_handle=str(profile_path),
        )

    def command_argv(self, environment: ExecutionEnvironment, command: CommandSpec) -> list[str]:
        return [self.sandbox_exec, "-f", environment.backend_handle, "--", *command.argv]

    def environment(self, environment: ExecutionEnvironment, command: CommandSpec) -> dict[str, str]:
        base = super().environment(environment, command)
        base["PATH"] = runtime_search_path(self.runtime_workspace_root, base.get("PATH", ""))
        return apply_project_agent_environment(base, self.runtime_workspace_root)

    def _runtime_read_roots(self) -> tuple[Path, ...]:
        """Return only managed program/runtime trees required by allowed commands.

        The private snapshot remains the only writable project tree.  The source
        virtual environment is read-only so `python` retains the project's
        dependencies without granting the command read access to the source
        workspace, its configuration, or the account home directory.
        """
        candidates: list[Path] = [self.runtime_workspace_root / ".venv"]
        for binary_dir in runtime_bin_directories(self.runtime_workspace_root):
            resolved = binary_dir.resolve()
            if resolved.name == "bin":
                candidates.append(resolved.parent)
            else:
                candidates.append(resolved)
        roots: list[Path] = []
        for candidate in candidates:
            if not candidate.is_dir():
                continue
            resolved = candidate.resolve()
            if resolved not in roots:
                roots.append(resolved)
        return tuple(roots)

    @staticmethod
    def _profile(workspace_path: Path, runtime_roots: tuple[Path, ...]) -> str:
        workspace = _seatbelt_path(workspace_path)
        temporary = _seatbelt_path(workspace_path / ".work-agent-tmp")
        lines = [
            "(version 1)",
            "(deny default)",
            "(allow process-exec)",
            "(allow process-fork)",
            "(allow signal (target self))",
            "(allow file-read* (subpath \"/System\") (subpath \"/usr\") (subpath \"/bin\") (subpath \"/sbin\") (subpath \"/Library\"))",
        ]
        lines.extend(f"(allow file-read* (subpath \"{_seatbelt_path(root)}\"))" for root in runtime_roots)
        lines.extend(
            [
                f"(allow file-read* (subpath \"{workspace}\"))",
                f"(allow file-write* (subpath \"{workspace}\") (subpath \"{temporary}\"))",
                "(deny network*)",
            ]
        )
        return "\n".join(lines)


def _seatbelt_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "\\\\").replace('"', '\\"')
