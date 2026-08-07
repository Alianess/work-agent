from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol
import os
import selectors
import signal
import subprocess
import time
import uuid

from ..events import ExecutionEventSink
from ..models import BackendKind, CapabilitySet, CommandSpec, ExecutionContract, ProcessOutcome


CancelCheck = Callable[[], bool]


@dataclass(frozen=True)
class BackendHealth:
    available: bool
    detail: str = ""
    version: str = ""


@dataclass(frozen=True)
class ExecutionEnvironment:
    environment_id: str
    backend: BackendKind
    workspace_path: Path
    log_dir: Path
    backend_handle: str = ""


class ExecutionBackend(Protocol):
    kind: BackendKind

    def health(self) -> BackendHealth: ...

    def prepare(
        self,
        contract: ExecutionContract,
        *,
        workspace_path: Path,
        log_dir: Path,
    ) -> ExecutionEnvironment: ...

    def run(
        self,
        environment: ExecutionEnvironment,
        command: CommandSpec,
        capabilities: CapabilitySet,
        events: ExecutionEventSink,
        cancel_check: CancelCheck | None = None,
    ) -> ProcessOutcome: ...

    def destroy(self, environment: ExecutionEnvironment) -> None: ...


class ProcessExecutionBackend:
    """Common process lifecycle with process-group cancellation and output caps."""

    kind: BackendKind

    def health(self) -> BackendHealth:
        return BackendHealth(available=True)

    def prepare(
        self,
        contract: ExecutionContract,
        *,
        workspace_path: Path,
        log_dir: Path,
    ) -> ExecutionEnvironment:
        log_dir.mkdir(parents=True, exist_ok=True)
        return ExecutionEnvironment(
            environment_id=f"env_{uuid.uuid4().hex}",
            backend=self.kind,
            workspace_path=workspace_path,
            log_dir=log_dir,
        )

    def run(
        self,
        environment: ExecutionEnvironment,
        command: CommandSpec,
        capabilities: CapabilitySet,
        events: ExecutionEventSink,
        cancel_check: CancelCheck | None = None,
    ) -> ProcessOutcome:
        if not command.argv:
            raise ValueError("执行命令不能为空。")
        if command.shell:
            raise ValueError("执行后端不接受 Shell 解释命令。")
        cwd = _safe_cwd(environment.workspace_path, command.cwd)
        argv = self.command_argv(environment, command)
        env = self.environment(environment, command)
        started_at_ms = int(time.time() * 1000)
        started_at_monotonic = time.monotonic()
        stdout_path = environment.log_dir / "stdout.log"
        stderr_path = environment.log_dir / "stderr.log"
        events.emit(
            "process.started",
            phase="running",
            summary="已在安全执行环境启动命令。",
            payload={"argv": list(command.argv), "cwd": command.cwd, "backend": self.kind.value},
        )
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            stdin=subprocess.PIPE if command.stdin_text is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            bufsize=1,
            start_new_session=True,
            preexec_fn=_resource_limiter(capabilities) if os.name != "nt" else None,
        )
        if command.stdin_text is not None and process.stdin is not None:
            try:
                process.stdin.write(command.stdin_text)
                process.stdin.close()
            except BrokenPipeError:
                pass

        selector = selectors.DefaultSelector()
        streams: dict[object, str] = {}
        for stream, stream_name in ((process.stdout, "stdout"), (process.stderr, "stderr")):
            if stream is not None:
                selector.register(stream, selectors.EVENT_READ)
                streams[stream] = stream_name
        output_sizes = {"stdout": 0, "stderr": 0}
        output_caps = {"stdout": capabilities.resources.stdout_bytes, "stderr": capabilities.resources.stderr_bytes}
        output_handles = {
            "stdout": stdout_path.open("w", encoding="utf-8"),
            "stderr": stderr_path.open("w", encoding="utf-8"),
        }
        timed_out = False
        cancelled = False
        cap_warned: set[str] = set()
        try:
            while selector.get_map():
                now = time.monotonic()
                if cancel_check is not None and cancel_check():
                    cancelled = True
                    self._terminate_tree(process)
                    break
                if now - started_at_monotonic > capabilities.resources.wall_timeout_seconds:
                    timed_out = True
                    self._terminate_tree(process)
                    break
                for key, _ in selector.select(timeout=0.2):
                    stream = key.fileobj
                    line = stream.readline()
                    if not line:
                        try:
                            selector.unregister(stream)
                        except Exception:
                            pass
                        continue
                    stream_name = streams[stream]
                    encoded_size = len(line.encode("utf-8", errors="replace"))
                    if output_sizes[stream_name] < output_caps[stream_name]:
                        remaining = output_caps[stream_name] - output_sizes[stream_name]
                        clipped = line.encode("utf-8", errors="replace")[:remaining].decode("utf-8", errors="replace")
                        output_handles[stream_name].write(clipped)
                        output_handles[stream_name].flush()
                        output_sizes[stream_name] += len(clipped.encode("utf-8", errors="replace"))
                        events.emit(
                            f"process.{stream_name}.delta",
                            phase="running",
                            summary="命令输出更新。",
                            payload={"content": clipped, "stream": stream_name},
                            visibility="debug",
                        )
                    elif stream_name not in cap_warned:
                        cap_warned.add(stream_name)
                        events.emit(
                            "resource.warning",
                            phase="running",
                            summary=f"{stream_name} 输出达到上限，后续内容仅丢弃并继续排空。",
                            payload={"stream": stream_name, "limit_bytes": output_caps[stream_name]},
                        )
                if process.poll() is not None and not selector.get_map():
                    break
            returncode = process.wait(timeout=5)
        finally:
            selector.close()
            for handle in output_handles.values():
                handle.close()
            for stream in (process.stdout, process.stderr, process.stdin):
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:
                        pass
        finished_at_ms = int(time.time() * 1000)
        outcome = ProcessOutcome(
            exit_code=process.returncode,
            signal=-process.returncode if process.returncode is not None and process.returncode < 0 else None,
            timed_out=timed_out,
            cancelled=cancelled,
            stdout_ref=str(stdout_path),
            stderr_ref=str(stderr_path),
            started_at_ms=started_at_ms,
            finished_at_ms=finished_at_ms,
        )
        if cancelled:
            events.emit("execution.cancelled", phase="cancelled", summary="执行已取消，进程树已终止。")
        elif timed_out:
            events.emit("execution.failed", phase="failed", summary="执行超时，进程树已终止。", payload={"code": "PROCESS_TIMEOUT"})
        elif process.returncode == 0:
            events.emit("process.completed", phase="running", summary="命令已完成。", payload={"exit_code": 0})
        else:
            events.emit(
                "process.failed",
                phase="running",
                summary=f"命令以退出码 {process.returncode} 结束。",
                payload={"exit_code": process.returncode},
            )
        return outcome

    def command_argv(self, environment: ExecutionEnvironment, command: CommandSpec) -> list[str]:
        return list(command.argv)

    def environment(self, environment: ExecutionEnvironment, command: CommandSpec) -> dict[str, str]:
        allowed = {key: value for key, value in os.environ.items() if key in {"PATH", "LANG", "LC_ALL", "LC_CTYPE", "TERM"}}
        allowed["HOME"] = str(environment.workspace_path / ".work-agent-home")
        allowed["TMPDIR"] = str(environment.workspace_path / ".work-agent-tmp")
        (environment.workspace_path / ".work-agent-home").mkdir(exist_ok=True)
        (environment.workspace_path / ".work-agent-tmp").mkdir(exist_ok=True)
        for key, value in command.env.items():
            if key.upper() in {"HOME", "PATH", "DYLD_INSERT_LIBRARIES", "LD_PRELOAD"}:
                continue
            allowed[key] = value
        return allowed

    def destroy(self, environment: ExecutionEnvironment) -> None:
        return None

    @staticmethod
    def _terminate_tree(process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        try:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            return


def _safe_cwd(workspace_root: Path, raw_cwd: str) -> Path:
    candidate = (workspace_root / raw_cwd).resolve(strict=False)
    if workspace_root.resolve() not in (candidate, *candidate.parents):
        raise ValueError("命令工作目录越出执行工作区。")
    if not candidate.is_dir():
        raise FileNotFoundError(f"执行工作目录不存在：{raw_cwd}")
    return candidate


def _resource_limiter(capabilities: CapabilitySet) -> Callable[[], None]:
    def apply() -> None:
        try:
            import resource

            resource.setrlimit(resource.RLIMIT_CPU, (capabilities.resources.cpu_seconds, capabilities.resources.cpu_seconds + 1))
            resource.setrlimit(resource.RLIMIT_AS, (capabilities.resources.memory_bytes, capabilities.resources.memory_bytes))
            resource.setrlimit(resource.RLIMIT_NPROC, (capabilities.resources.pids, capabilities.resources.pids))
            resource.setrlimit(resource.RLIMIT_NOFILE, (capabilities.resources.open_files, capabilities.resources.open_files))
            resource.setrlimit(resource.RLIMIT_FSIZE, (capabilities.filesystem.max_written_bytes, capabilities.filesystem.max_written_bytes))
        except Exception:
            # Seatbelt remains the filesystem/network boundary. Some rlimits
            # are unavailable to an unprivileged macOS process, so the caller
            # must treat an unavailable limit as a backend-health concern, not
            # as permission to use a less restricted host process.
            pass

    return apply
