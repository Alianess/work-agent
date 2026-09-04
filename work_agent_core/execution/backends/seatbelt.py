from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import threading

from ..errors import failure
from ..models import BackendKind, CommandSpec, ExecutionContract
from ..network_broker import NetworkBroker
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
        extra_read_roots: tuple[Path, ...] = (),
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
        # 用户在 agent 设置里声明的只读目录（桌面、下载……）：shell 里的读与
        # read_file 的读保持一致，不然同一条路径两个工具两种答案。
        self.extra_read_roots = extra_read_roots
        self._broker_lock = threading.RLock()
        self._brokers: dict[str, tuple[NetworkBroker, int]] = {}

    def _read_roots(self) -> tuple[Path, ...]:
        """Managed runtimes plus the account data the command is working on."""
        roots = list(self._runtime_read_roots())
        source = self.readable_source_root
        if source is not None and source.is_dir() and source not in roots:
            roots.append(source)
        for extra in self.extra_read_roots:
            resolved = Path(extra).resolve()
            if resolved.is_dir() and resolved not in roots:
                roots.append(resolved)
        return tuple(roots)

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
        broker: NetworkBroker | None = None
        broker_port: int | None = None
        try:
            if contract.capabilities.network.mode == "domain_allowlist":
                broker = NetworkBroker(
                    allowed_domains=frozenset(contract.capabilities.network.allowed_domains),
                    max_bytes_in=contract.capabilities.network.max_bytes_in,
                    max_bytes_out=contract.capabilities.network.max_bytes_out,
                )
                broker_port = broker.start()
                with self._broker_lock:
                    self._brokers[environment.environment_id] = (broker, broker_port)
            profile_path = environment.log_dir / "seatbelt.sb"
            profile_path.write_text(
                self._profile(
                    workspace_path,
                    self._read_roots(),
                    network_broker_port=broker_port,
                    dependency_write_root=(self.runtime_workspace_root / ".venv") if broker_port else None,
                ),
                encoding="utf-8",
            )
        except Exception:
            with self._broker_lock:
                self._brokers.pop(environment.environment_id, None)
            if broker is not None:
                broker.close()
            raise
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
        with self._broker_lock:
            broker_entry = self._brokers.get(environment.environment_id)
        if broker_entry is not None:
            proxy_url = f"http://127.0.0.1:{broker_entry[1]}"
            base.update(
                {
                    "HTTP_PROXY": proxy_url,
                    "HTTPS_PROXY": proxy_url,
                    "http_proxy": proxy_url,
                    "https_proxy": proxy_url,
                    "NO_PROXY": "",
                    "no_proxy": "",
                    "PIP_CONFIG_FILE": "/dev/null",
                    "PIP_INDEX_URL": "https://pypi.org/simple",
                    "PIP_EXTRA_INDEX_URL": "",
                    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                    "SSL_CERT_FILE": "/private/etc/ssl/cert.pem",
                    "REQUESTS_CA_BUNDLE": "/private/etc/ssl/cert.pem",
                    "CURL_CA_BUNDLE": "/private/etc/ssl/cert.pem",
                    "NPM_CONFIG_REGISTRY": "https://registry.npmjs.org/",
                    "NPM_CONFIG_USERCONFIG": "/dev/null",
                    "NPM_CONFIG_AUDIT": "false",
                    "NPM_CONFIG_FUND": "false",
                }
            )
        return apply_project_agent_environment(base, self.runtime_workspace_root)

    def destroy(self, environment: ExecutionEnvironment) -> None:
        with self._broker_lock:
            broker_entry = self._brokers.pop(environment.environment_id, None)
        if broker_entry is not None:
            broker_entry[0].close()

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
    def _profile(
        workspace_path: Path,
        runtime_roots: tuple[Path, ...],
        *,
        network_broker_port: int | None = None,
        dependency_write_root: Path | None = None,
    ) -> str:
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
            # macOS exposes /etc through a symlink into /private/etc; TLS
            # clients resolve the certificate path before opening it.
            Path("/private/etc"),
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
        write_filters = [
            '(literal "/dev/ptmx")',
            '(literal "/dev/dtracehelper")',
            '(literal "/dev/null")',
            '(literal "/dev/random")',
            '(literal "/dev/urandom")',
            '(literal "/dev/zero")',
            '(regex #"^/dev/fd/[0-9]+$")',
            '(regex #"^/dev/tty[a-z0-9]*$")',
            f'(subpath "{workspace}")',
            f'(subpath "{temporary}")',
        ]
        if dependency_write_root is not None:
            write_filters.append(f'(subpath "{_seatbelt_path(dependency_write_root)}")')
        network_rule = (
            f'(allow network-outbound (remote ip "localhost:{network_broker_port}"))'
            if network_broker_port
            else ""
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
            "(allow file-write*\n    " + "\n    ".join(write_filters) + ")",
            "(deny appleevent-send)",
            "(deny user-preference-read user-preference-write)",
            "(deny distributed-notification-post)",
            "(deny iokit-open*)",
            "(deny network*)",
            network_rule,
        ]
        return "\n".join(line for line in lines if line)


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
