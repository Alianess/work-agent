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
        readable_source_root: str | Path | None = None,
    ) -> None:
        self.sandbox_exec = sandbox_exec
        self.runtime_workspace_root = Path(runtime_workspace_root or Path.cwd()).resolve()
        # Reads are wide, writes stay narrow. Denying reads of the account's own
        # data made every shell command report the user's files as missing —
        # the snapshot does not contain them — which reads as data loss rather
        # than as isolation. Writes still land only in the private snapshot.
        self.readable_source_root = (
            Path(readable_source_root).resolve() if readable_source_root else None
        )

    def health(self) -> BackendHealth:
        executable = shutil.which(self.sandbox_exec) if "/" not in self.sandbox_exec else self.sandbox_exec
        if not executable or not Path(executable).is_file():
            return BackendHealth(False, "未找到 macOS Seatbelt sandbox-exec。")
        profile = self._profile(self.runtime_workspace_root, self._read_roots())
        try:
            probe = subprocess.run(
                [executable, "-p", profile, "/usr/bin/true"],
                capture_output=True,
                text=True,
                timeout=4,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return BackendHealth(False, "Seatbelt 健康探测在 4 秒内没有完成。")
        except OSError as error:
            detail = str(error).strip() or type(error).__name__
            return BackendHealth(False, f"无法启动 Seatbelt 健康探测：{detail}")
        if probe.returncode != 0:
            detail = (probe.stderr or probe.stdout or "").strip()
            if "sandbox_apply: Operation not permitted" in detail:
                detail = (
                    "当前服务进程处于不允许嵌套 Seatbelt 的父级沙箱；"
                    "请通过本机 launchd 独立重新安装并启动 Work Agent。"
                )
            elif not detail:
                detail = f"Seatbelt 健康探测失败（退出码 {probe.returncode}，未返回错误详情）。"
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
        profile_path.write_text(self._profile(workspace_path, self._read_roots()), encoding="utf-8")
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

    def _read_roots(self) -> tuple[Path, ...]:
        """Managed runtimes plus the account data the command is working on."""
        roots = list(self._runtime_read_roots())
        source = self.readable_source_root
        if source is not None and source.is_dir() and source not in roots:
            roots.append(source)
        return tuple(roots)

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
        readable_roots = (
            Path("/System"),
            Path("/usr"),
            Path("/bin"),
            Path("/sbin"),
            Path("/Library"),
            # /bin/sh consults this to pick the concrete shell; without it the
            # sandbox denies every pipeline before the command even starts.
            Path("/private/var/select"),
            Path("/etc"),
            *runtime_roots,
            workspace_path,
        )
        readable_ancestors = _seatbelt_ancestor_paths(readable_roots)
        read_filters = [f'(literal "{_seatbelt_path(path)}")' for path in readable_ancestors]
        read_filters.extend(f'(subpath "{_seatbelt_path(path)}")' for path in readable_roots)
        read_filters.extend(
            [
                '(literal "/dev/null")',
                '(literal "/dev/random")',
                '(literal "/dev/urandom")',
                '(literal "/dev/zero")',
                '(regex #"^/dev/fd/[0-9]+$")',
            ]
        )
        lines = [
            "(version 1)",
            # Modern macOS performs sysctl and service initialization before
            # the target command starts. A global ``deny default`` aborts the
            # target before filesystem rules can take effect. Keep system
            # initialization compatible, then apply explicit file, network,
            # preference, device, and Apple-event denials below.
            "(allow default)",
            "(deny file-read*)",
            "(allow file-read*\n    " + "\n    ".join(read_filters) + ")",
            "(deny file-write*)",
            "(allow file-write*\n"
            "    (literal \"/dev/ptmx\")\n"
            "    (literal \"/dev/dtracehelper\")\n"
            "    (literal \"/dev/null\")\n"
            "    (literal \"/dev/random\")\n"
            "    (literal \"/dev/urandom\")\n"
            "    (literal \"/dev/zero\")\n"
            "    (regex #\"^/dev/fd/[0-9]+$\")\n"
            "    (regex #\"^/dev/tty[a-z0-9]*$\")\n"
            f"    (subpath \"{workspace}\")\n"
            f"    (subpath \"{temporary}\"))",
            "(deny appleevent-send)",
            "(deny user-preference-read user-preference-write)",
            "(deny distributed-notification-post)",
            "(deny iokit-open*)",
            "(deny network*)",
        ]
        return "\n".join(lines)


def _seatbelt_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "\\\\").replace('"', '\\"')


def _seatbelt_ancestor_paths(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    """Allow only metadata traversal for parents of explicitly readable roots."""
    result: list[Path] = [Path("/")]
    for path in paths:
        resolved = path.resolve()
        for parent in reversed(resolved.parents):
            if parent not in result:
                result.append(parent)
    return tuple(result)
