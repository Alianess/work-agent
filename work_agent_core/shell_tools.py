from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
import json
import hashlib
import os
import secrets
import shlex
import threading
import time
import uuid

from .execution import (
    CapabilitySet,
    CommandSpec,
    ExecutionClass,
    ExecutionMode,
    ExecutionOrchestrator,
    ExecutionRequest,
)
from .execution.events import ExecutionEvent
from .progress import current_tool_cancel_check, emit_tool_progress
from .runtime_env import apply_project_agent_environment, project_agent_python, project_node
from .tools import Tool, ToolRegistry, WorkspaceFiles


AUTO_ALLOW_COMMANDS = {
    "pwd",
    "ls",
    "rg",
    "cat",
    "head",
    "tail",
    "wc",
    "file",
    "stat",
    "du",
    "which",
}
AUTO_ALLOW_GIT_SUBCOMMANDS = {
    "status",
    "log",
    "diff",
    "show",
    "branch",
}
ASK_COMMANDS = {
    "python",
    "python3",
    "node",
    "npm",
    "pnpm",
    "npx",
    "pip",
    "pip3",
    "conda",
    "ffmpeg",
    "soffice",
    "libreoffice",
    "pandoc",
    "mkdir",
    "cp",
    "mv",
    "touch",
    "rm",
}
VERSION_FLAGS = {"--version", "-V", "-v", "version"}
DENY_COMMANDS = {
    "sudo",
    "su",
    "chmod",
    "chown",
    "curl",
    "wget",
    "ssh",
    "scp",
    "rsync",
}
SHELL_CONTROL_TOKENS = {
    "|",
    "||",
    "&",
    "&&",
    ";",
    ">",
    ">>",
    "<",
    "$(",
    "`",
}
SENSITIVE_PATH_PARTS = {
    ".env",
    ".ssh",
    "id_rsa",
    "id_ed25519",
    "private_key",
    "secret",
    "secrets",
    "token",
    "api_key",
}

_APPROVAL_GRANT_LOCK = threading.RLock()
_APPROVAL_GRANTS: dict[str, tuple[str, str, int]] = {}
APPROVAL_GRANT_TTL_SECONDS = 5 * 60


@dataclass(frozen=True)
class ShellDecision:
    status: str
    reason: str
    risk_category: str


class ShellExecutionTools:
    def __init__(
        self,
        workspace_root: str | Path,
        *,
        execution_orchestrator: ExecutionOrchestrator | None = None,
        runtime_workspace_root: str | Path | None = None,
        account_id: str = "local",
        turn_id: str = "",
        conversation_id: str = "",
        project_id: str = "",
    ) -> None:
        self.workspace = WorkspaceFiles(workspace_root)
        self.workspace_root = self.workspace.workspace_root
        self.runtime_workspace_root = Path(runtime_workspace_root or self.workspace_root).resolve()
        self.execution_orchestrator = execution_orchestrator or ExecutionOrchestrator(
            workspace_root=self.workspace_root,
            runtime_workspace_root=self.runtime_workspace_root,
        )
        self.account_id = str(account_id or "local")
        self.turn_id = str(turn_id or "")
        self.conversation_id = str(conversation_id or "")
        self.project_id = str(project_id or "")

    def execute(self, args: dict[str, Any]) -> str:
        internal_tool_call_id = str(args.get("_execution_tool_call_id") or "").strip()
        command_text = str(args.get("command") or "").strip()
        if not command_text:
            raise ValueError("缺少 command。")
        cwd = self._resolve_cwd(str(args.get("cwd") or "."))
        timeout_seconds = min(max(int(args.get("timeout_seconds") or 120), 1), 900)
        argv = parse_command(command_text)
        decision = self._decide(argv, cwd)
        action_id = approval_action_id(
            command=command_text,
            cwd=str(cwd),
            timeout_seconds=timeout_seconds,
        )
        approval_granted = consume_internal_approval_grant(
            token=str(args.get("_approval_grant") or ""),
            action_id=action_id,
            source=str(args.get("_approval_source") or ""),
        )

        if decision.status == "deny":
            return json.dumps(
                {
                    "ok": False,
                    "status": "denied",
                    "risk_category": decision.risk_category,
                    "reason": decision.reason,
                    "command": command_text,
                    "cwd": str(cwd),
                },
                ensure_ascii=False,
                indent=2,
            )
        if decision.status == "ask" and not approval_granted:
            return json.dumps(
                {
                    "ok": False,
                    "status": "approval_required",
                    "risk_category": decision.risk_category,
                    "reason": decision.reason,
                    "command": command_text,
                    "cwd": str(cwd),
                    "timeout_seconds": timeout_seconds,
                    "auto_approvable": is_auto_approvable_command(argv, decision),
                    "reviewable_by_model": is_model_reviewable_command(argv, decision),
                    "action_id": action_id,
                    "preview": build_command_preview(
                        command_text=command_text,
                        cwd=cwd,
                        timeout_seconds=timeout_seconds,
                        risk_category=decision.risk_category,
                        reason=decision.reason,
                    ),
                    "next_step": "由独立审查智能体或用户确认当前精确动作后，系统使用内部审批凭证重试。",
                },
                ensure_ascii=False,
                indent=2,
            )

        # Replaying a streamed/recovered native tool call returns the same
        # execution receipt instead of creating a second host process.
        execution_id_seed = internal_tool_call_id or uuid.uuid4().hex
        execution_identity = (
            f"toolcall:{self.turn_id or self.conversation_id or 'standalone'}:{execution_id_seed}"
        )
        managed_argv = self._managed_runtime_argv(argv)
        default_capabilities = CapabilitySet()
        execution = self.execution_orchestrator.submit(
            ExecutionRequest(
                request_id=f"shell_{execution_identity}",
                idempotency_key=execution_identity,
                account_id=self.account_id,
                turn_id=self.turn_id,
                conversation_id=self.conversation_id,
                project_id=self.project_id,
                tool_call_id=execution_id_seed,
                tool_name="shell_exec",
                execution_class=ExecutionClass.ISOLATED_PROCESS,
                mode=ExecutionMode.ISOLATED,
                command=CommandSpec(
                    argv=tuple(managed_argv),
                    cwd=".",
                    env={"WORK_AGENT_RUNTIME": "isolated"},
                ),
                requested_capabilities=CapabilitySet(
                    resources=replace(
                        default_capabilities.resources,
                        wall_timeout_seconds=timeout_seconds,
                    )
                ),
                delivery_mode="apply_after_validation",
                reason=f"运行工作区内命令：{Path(argv[0]).name}",
            ),
            source_root=cwd,
            on_event=self._on_execution_event,
            cancel_check=current_tool_cancel_check(),
        )
        self._emit_execution_result(execution)
        stdout, stderr = self._execution_output(execution)
        succeeded = execution.status.value == "succeeded"

        return json.dumps(
            {
                "ok": succeeded,
                "status": "executed" if succeeded else "failed",
                "permission": decision.status,
                "risk_category": decision.risk_category,
                "command": command_text,
                "cwd": str(cwd),
                "returncode": execution.process.exit_code if execution.process else None,
                "stdout": truncate(stdout, 20000),
                "stderr": truncate(stderr, 12000),
                "execution_id": execution.execution_id,
                "execution_status": execution.status.value,
                "delivery_status": execution.delivery_status.value,
                "change_set_id": execution.change_set_id,
                "receipt_id": execution.receipt_id,
                "error": execution.error.code if execution.error else "",
                "reason": execution.error.message if execution.error else "",
            },
            ensure_ascii=False,
            indent=2,
        )

    def _managed_runtime_argv(self, argv: list[str]) -> list[str]:
        """Pin managed interpreters instead of inheriting an arbitrary host PATH."""
        if not argv:
            return argv
        executable = Path(argv[0]).name
        resolved: Path | None = None
        if executable in {"python", "python3"}:
            resolved = project_agent_python(self.runtime_workspace_root)
        elif executable == "node":
            resolved = project_node(self.runtime_workspace_root)
        if resolved is None:
            return list(argv)
        return [str(resolved), *argv[1:]]

    def _execution_output(self, execution: Any) -> tuple[str, str]:
        process = getattr(execution, "process", None)
        if process is None:
            return "", ""
        return self._read_execution_log(process.stdout_ref), self._read_execution_log(process.stderr_ref)

    @staticmethod
    def _read_execution_log(raw_path: str) -> str:
        if not raw_path:
            return ""
        try:
            return Path(raw_path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def _on_execution_event(self, event: ExecutionEvent) -> None:
        payload = event.payload
        if event.type == "process.stdout.delta" or event.type == "process.stderr.delta":
            emit_tool_progress(
                {
                    "event": "activity_delta",
                    "id": f"execution-{event.execution_id}",
                    "phase": "action",
                    "title": "安全执行环境",
                    "content": str(payload.get("content") or ""),
                    "append_mode": "append",
                    "activity_type": "command",
                    "command_status": "running",
                    "execution_id": event.execution_id,
                    "execution_status": event.phase,
                }
            )
            return
        status = "running"
        if event.type in {"execution.completed", "delivery.applied", "process.completed"}:
            status = "success"
        elif event.type in {"execution.failed", "execution.cancelled", "process.failed"}:
            status = "error"
        emit_tool_progress(
            {
                "event": "activity",
                "id": f"execution-{event.execution_id}",
                "phase": normalized_activity_phase(event.phase, status),
                "title": "安全执行环境",
                "detail": event.summary,
                "content": "",
                "activity_type": "runtime_summary",
                "command_status": status,
                "execution_id": event.execution_id,
                "execution_status": event.phase,
                "execution_event": event.type,
            }
        )

    @staticmethod
    def _emit_execution_result(execution: Any) -> None:
        status = str(getattr(getattr(execution, "status", None), "value", "failed"))
        delivery = str(getattr(getattr(execution, "delivery_status", None), "value", "none"))
        error = getattr(execution, "error", None)
        if error is not None:
            detail = str(getattr(error, "message", "安全执行失败。"))
            command_status = "error"
            phase = "error"
        elif delivery == "applied":
            detail = "验证通过，变更已原子写回当前工作目录。"
            command_status = "success"
            phase = "complete"
        elif delivery == "validated":
            detail = "命令和产物验证完成，未产生需要写回的文件变更。"
            command_status = "success"
            phase = "complete"
        elif delivery == "changes_ready":
            detail = "变更已准备好，等待显式写回确认。"
            command_status = "running"
            phase = "action"
        else:
            detail = "安全执行已结束。"
            command_status = "success" if status == "succeeded" else "error"
            phase = "complete" if command_status == "success" else "error"
        emit_tool_progress(
            {
                "event": "activity_delta",
                "id": f"execution-{execution.execution_id}",
                "phase": phase,
                "title": "安全执行环境",
                "content": "",
                "append_mode": "replace",
                "detail": detail,
                "activity_type": "command",
                "command_status": command_status,
                "execution_id": execution.execution_id,
                "execution_status": status,
                "delivery_status": delivery,
                "change_set_id": getattr(execution, "change_set_id", None),
                "receipt_id": getattr(execution, "receipt_id", ""),
            }
        )

    def _resolve_cwd(self, raw_cwd: str) -> Path:
        cwd = self.workspace.resolve(raw_cwd)
        if not cwd.exists():
            raise FileNotFoundError(f"cwd 不存在：{raw_cwd}")
        if not cwd.is_dir():
            raise NotADirectoryError(f"cwd 不是目录：{raw_cwd}")
        return cwd

    def _decide(self, argv: list[str], cwd: Path) -> ShellDecision:
        executable = Path(argv[0]).name
        if executable == "rm":
            return decide_rm(argv, cwd, self.workspace_root)
        if executable == "find":
            return decide_find(argv, cwd, self.workspace_root)
        if executable in DENY_COMMANDS:
            return ShellDecision("deny", f"{executable} 属于高风险命令，当前策略直接拒绝。", risk_category_for_executable(executable))
        if has_shell_control_token(argv):
            return ShellDecision("deny", "当前 shell_exec 只支持单个命令 argv，不支持管道、重定向、命令替换或多命令串联。", "SYSTEM")
        if contains_sensitive_path(argv):
            return ShellDecision("deny", "命令参数疑似访问密钥、环境变量或敏感路径。", "SYSTEM")
        skill_hint = detect_skill_script_argument(argv, self.workspace_root)
        if skill_hint is not None:
            skill_id, script_path = skill_hint
            return ShellDecision(
                "ask",
                (
                    f"该命令调用技能 {skill_id} 目录内的脚本 {script_path}。"
                    "技能脚本应通过 run_skill_script 运行，这样会自动使用技能对应的 office_python 解释器，"
                    "并避免误用系统 python3 缺失依赖。如果确需用 shell_exec，请向用户说明原因。"
                ),
                "EXECUTE",
            )
        if executable == "git":
            subcommand = argv[1] if len(argv) > 1 else ""
            if subcommand in AUTO_ALLOW_GIT_SUBCOMMANDS and paths_stay_in_workspace(argv[2:], cwd, self.workspace_root):
                return ShellDecision("allow", f"git {subcommand} 是只读或低风险查看命令。", "READ")
            return ShellDecision("ask", "git 写入、网络或不明确子命令需要用户确认。", "EXECUTE")
        if executable in AUTO_ALLOW_COMMANDS:
            if paths_stay_in_workspace(argv[1:], cwd, self.workspace_root):
                return ShellDecision("allow", f"{executable} 属于只读查看白名单。", "READ")
            return ShellDecision("deny", "命令参数包含工作区外路径。", "SYSTEM")
        if executable in ASK_COMMANDS and len(argv) == 2 and argv[1] in VERSION_FLAGS:
            return ShellDecision("allow", f"{executable} 版本查看属于低风险命令。", "READ")
        if executable in ASK_COMMANDS:
            if paths_stay_in_workspace(argv[1:], cwd, self.workspace_root):
                return ShellDecision("ask", f"{executable} 可能执行脚本、生成文件、安装依赖或修改文件，需要用户确认。", risk_category_for_executable(executable))
            return ShellDecision("deny", "命令参数包含工作区外路径。", "SYSTEM")
        if not paths_stay_in_workspace(argv[1:], cwd, self.workspace_root):
            return ShellDecision("deny", "未知命令的参数包含工作区外路径。", "SYSTEM")
        return ShellDecision("ask", f"{executable} 不在白名单内，需要用户确认。", risk_category_for_executable(executable))


def normalized_activity_phase(execution_phase: str, command_status: str) -> str:
    """Map durable execution states to the small public Activity phase set."""
    if command_status == "error" or execution_phase in {"failed", "cancelled"}:
        return "error"
    if execution_phase in {"succeeded", "complete", "applied", "validated"}:
        return "complete"
    if execution_phase == "waiting_permission":
        return "thinking"
    return "action"


def detect_skill_script_argument(
    argv: list[str], workspace_root: Path
) -> tuple[str, str] | None:
    """Return (skill_id, script_path) if any argv element is a script inside a
    skill folder. Used to nudge the model toward run_skill_script instead of a
    bare shell_exec (which would use the system python3 and miss office deps).
    """
    skills_roots = [
        workspace_root / "work_agent_skills",
        workspace_root / "meeting_audio_minutes" / "skills",
    ]
    for token in argv:
        candidate = Path(token)
        if not candidate.is_absolute():
            candidate = workspace_root / token
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        for skills_root in skills_roots:
            try:
                rel = resolved.relative_to(skills_root.resolve())
            except ValueError:
                continue
            parts = rel.parts
            if len(parts) < 2:
                continue
            if parts[1] == "scripts" and resolved.suffix.lower() in {".py", ".js", ".sh"}:
                return parts[0], "/".join(parts[1:])
    return None


def register_shell_tools(
    registry: ToolRegistry,
    workspace_root: str | Path,
    *,
    execution_orchestrator: ExecutionOrchestrator | None = None,
    runtime_workspace_root: str | Path | None = None,
    account_id: str = "local",
    turn_id: str = "",
    conversation_id: str = "",
    project_id: str = "",
) -> None:
    shell = ShellExecutionTools(
        workspace_root,
        execution_orchestrator=execution_orchestrator,
        runtime_workspace_root=runtime_workspace_root,
        account_id=account_id,
        turn_id=turn_id,
        conversation_id=conversation_id,
        project_id=project_id,
    )
    registry.register(
        Tool(
            name="shell_exec",
            description=(
                "Run a controlled terminal command inside a private macOS Seatbelt workspace. The response includes a risk_category: "
                "READ, MODIFY, EXECUTE, NETWORK, DELETE, or SYSTEM. Safe read-only commands "
                "(pwd/ls/find/rg/cat/head/tail/wc/file/stat/du and read-only git subcommands) run automatically. "
                "Commands that may write files, delete a specific workspace target, run scripts, install packages, use the network, "
                "or take a long time return approval_required with a preview. Broad deletion, sensitive access, and boundary escapes "
                "are denied by fixed policy. "
                "Do not use this tool to read, concatenate, create, or edit text/Markdown files; use the dedicated workspace file tools. "
                "This is argv execution, not a shell: never use pipes, redirection such as 2>&1, or shell globs with ls. "
                "For filename-pattern checks use find or rg --files. A nonzero returncode is a failed command; do not report "
                "a verification suite as fully passed unless every required check succeeded. If native isolation is unavailable, "
                "the command fails closed and never silently runs on the host."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Single command line. Pipes, redirection, subshells, command chaining, and shell glob expansion are not supported. Use find or rg --files for pattern checks; treat any nonzero returncode as a failed check.",
                    },
                    "cwd": {"type": "string", "default": "."},
                    "timeout_seconds": {"type": "integer", "default": 120},
                },
                "required": ["command"],
            },
            handler=shell.execute,
        )
    )


def parse_command(command_text: str) -> list[str]:
    try:
        argv = shlex.split(command_text)
    except ValueError as error:
        raise ValueError(f"命令无法解析：{error}") from error
    if not argv:
        raise ValueError("命令为空。")
    return argv


def has_shell_control_token(argv: list[str]) -> bool:
    for item in argv:
        if item in SHELL_CONTROL_TOKENS:
            return True
        if "$(" in item or "`" in item:
            return True
    return False


def contains_sensitive_path(argv: list[str]) -> bool:
    lowered = " ".join(argv).lower()
    return any(part in lowered for part in SENSITIVE_PATH_PARTS)


def paths_stay_in_workspace(args: list[str], cwd: Path, workspace_root: Path) -> bool:
    for item in args:
        if not item or item.startswith("-"):
            continue
        if looks_like_non_path_argument(item):
            continue
        path = Path(item).expanduser()
        if not path.is_absolute():
            path = cwd / path
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path.absolute()
        if workspace_root not in (resolved, *resolved.parents):
            return False
    return True


def looks_like_non_path_argument(value: str) -> bool:
    if value in {".", ".."}:
        return False
    if value.startswith(("http://", "https://")):
        return True
    if any(char in value for char in "*?[]{}"):
        return True
    if "/" in value or value.startswith("~"):
        return False
    if "." in value and len(value) > 1:
        return False
    return True


def risk_category_for_executable(executable: str) -> str:
    if executable in {"rm"}:
        return "DELETE"
    if executable in {"curl", "wget", "ssh", "scp", "rsync", "pip", "pip3", "npm", "pnpm", "npx"}:
        return "NETWORK"
    if executable in {"mkdir", "cp", "mv", "touch", "soffice", "libreoffice", "pandoc", "ffmpeg"}:
        return "MODIFY"
    if executable in {"python", "python3", "node", "conda", "git"}:
        return "EXECUTE"
    if executable in {"sudo", "su", "chmod", "chown"}:
        return "SYSTEM"
    return "EXECUTE"


def is_auto_approvable_command(argv: list[str], decision: ShellDecision) -> bool:
    """Return whether the optional approval delegate may approve this command.

    This is deliberately narrower than the ordinary ``ask`` bucket.  The
    delegate can approve local, workspace-confined artifact operations and a
    small set of verification commands, but never package installation,
    networking, git mutation, unknown programs, or general-purpose scripts.
    """
    if decision.status != "ask" or not argv:
        return False
    executable = Path(argv[0]).name
    if executable in {"mkdir", "touch", "cp", "mv", "ffmpeg", "soffice", "libreoffice", "pandoc"}:
        return True
    if executable in {"python", "python3"} and len(argv) >= 3 and argv[1] == "-m":
        return argv[2] in {"pytest", "unittest", "compileall"}
    if executable in {"npm", "pnpm"}:
        if len(argv) >= 2 and argv[1] == "test":
            return True
        return len(argv) >= 3 and argv[1] == "run" and argv[2] in {
            "build",
            "test",
            "lint",
            "typecheck",
            "check",
        }
    return False


def is_model_reviewable_command(argv: list[str], decision: ShellDecision) -> bool:
    """Return the fixed, narrow reviewer boundary.

    ``ask`` means the user may approve an exact action.  It does *not* mean a
    model reviewer may approve it: the latter is limited to deterministic,
    workspace-confined build and artifact operations from
    :func:`is_auto_approvable_command`.
    """
    return is_auto_approvable_command(argv, decision)


def decide_find(argv: list[str], cwd: Path, workspace_root: Path) -> ShellDecision:
    """Allow read-only find expressions and reject predicates with side effects."""
    side_effect_predicates = {
        "-delete",
        "-exec",
        "-execdir",
        "-ok",
        "-okdir",
        "-fprint",
        "-fprint0",
        "-fprintf",
        "-fls",
    }
    if any(item in side_effect_predicates for item in argv[1:]):
        return ShellDecision(
            "deny",
            "find 的该表达式会删除文件、执行子命令或写入文件，不属于只读查看。",
            "DELETE" if "-delete" in argv else "SYSTEM",
        )
    if paths_stay_in_workspace(argv[1:], cwd, workspace_root):
        return ShellDecision("allow", "find 仅执行工作区内的只读查找。", "READ")
    return ShellDecision("deny", "命令参数包含工作区外路径。", "SYSTEM")


def decide_rm(argv: list[str], cwd: Path, workspace_root: Path) -> ShellDecision:
    """Require an explicit user approval for every bounded deletion.

    The shell tool cannot prove that the user asked to remove this exact file.
    It therefore only validates scope here; it never silently authorizes the
    deletion and never delegates DELETE approval to a model reviewer.
    """
    if len(argv) < 2:
        return ShellDecision("deny", "rm 没有明确删除目标。", "DELETE")
    recursive_flags = {"-r", "-R", "--recursive"}
    dangerous_flags = {"--no-preserve-root", "-rf", "-fr", "-rF", "-Rf", "-fR"}
    if any(item in recursive_flags or item in dangerous_flags for item in argv[1:]):
        return ShellDecision("deny", "递归或宽范围删除不交给审查模型批准。", "DELETE")
    allowed_flags = {"-f", "--force", "-v", "--verbose", "--"}
    unknown_flags = [item for item in argv[1:] if item.startswith("-") and item not in allowed_flags]
    if unknown_flags:
        return ShellDecision("deny", f"rm 参数不在安全白名单内：{' '.join(unknown_flags)}", "DELETE")
    targets = [item for item in argv[1:] if item != "--" and not item.startswith("-")]
    if not targets:
        return ShellDecision("deny", "rm 没有明确删除目标。", "DELETE")
    if len(targets) > 20:
        return ShellDecision("deny", "单次删除目标过多，范围不够收敛。", "DELETE")
    resolved_targets: list[Path] = []
    for target in targets:
        if target in {".", ".."} or any(char in target for char in "*?[]{}"):
            return ShellDecision("deny", "删除目标不能是工作区根目录、相对上级或通配模式。", "DELETE")
        path = Path(target).expanduser()
        if not path.is_absolute():
            path = cwd / path
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path.absolute()
        if resolved == workspace_root or workspace_root not in resolved.parents:
            return ShellDecision("deny", "删除目标不在工作区内或指向工作区根目录。", "SYSTEM")
        if resolved.exists() and not resolved.is_file():
            return ShellDecision("deny", "直接删除仅支持明确的普通文件，不支持目录或其他文件类型。", "DELETE")
        resolved_targets.append(resolved)
    return ShellDecision(
        "ask",
        "删除目标明确且位于工作区内，但删除需要用户逐次确认。",
        "DELETE",
    )


def approval_action_id(*, command: str, cwd: str, timeout_seconds: int) -> str:
    normalized = json.dumps(
        {
            "command": str(command or "").strip(),
            "cwd": str(cwd or ".").strip() or ".",
            "timeout_seconds": min(max(int(timeout_seconds or 120), 1), 900),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return "approval-" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def issue_internal_approval_grant(*, action_id: str, source: str) -> str:
    """Create a short-lived, single-use capability for one approved action.

    The action ID is intentionally deterministic for audit, so it is not an
    authorization credential.  Only the runtime can mint this random grant;
    model-supplied ``_approval_source`` / ``_approval_action_id`` fields alone
    must never execute a command.
    """
    if source not in {"user", "reviewer"}:
        raise ValueError("approval grant source is invalid")
    clean_action_id = str(action_id or "").strip()
    if not clean_action_id:
        raise ValueError("approval grant action_id is required")
    token = secrets.token_urlsafe(32)
    now = int(time.time())
    with _APPROVAL_GRANT_LOCK:
        _prune_expired_approval_grants(now)
        _APPROVAL_GRANTS[token] = (clean_action_id, source, now + APPROVAL_GRANT_TTL_SECONDS)
    return token


def consume_internal_approval_grant(*, token: str, action_id: str, source: str) -> bool:
    """Consume an exact, short-lived approval grant; unknown grants fail closed."""
    clean_token = str(token or "").strip()
    if source not in {"user", "reviewer"} or not clean_token:
        return False
    now = int(time.time())
    with _APPROVAL_GRANT_LOCK:
        _prune_expired_approval_grants(now)
        grant = _APPROVAL_GRANTS.pop(clean_token, None)
    if grant is None:
        return False
    granted_action_id, granted_source, expires_at = grant
    return (
        now <= expires_at
        and secrets.compare_digest(granted_action_id, str(action_id or ""))
        and secrets.compare_digest(granted_source, source)
    )


def _prune_expired_approval_grants(now: int) -> None:
    expired = [token for token, grant in _APPROVAL_GRANTS.items() if grant[2] <= now]
    for token in expired:
        _APPROVAL_GRANTS.pop(token, None)


def build_command_preview(
    *,
    command_text: str,
    cwd: Path,
    timeout_seconds: int,
    risk_category: str,
    reason: str,
) -> str:
    return (
        f"风险类别：{risk_category}\n"
        f"工作目录：{cwd}\n"
        f"超时：{timeout_seconds}s\n"
        f"原因：{reason}\n"
        f"命令：{command_text}"
    )


def safe_environment(workspace_root: str | Path) -> dict[str, str]:
    allowed_keys = {
        "PATH",
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "PYTHONPATH",
        "WORK_AGENT_OFFICE_PYTHON",
        "WORK_AGENT_RUNTIME_BIN",
    }
    environment = {key: value for key, value in os.environ.items() if key in allowed_keys}
    return apply_project_agent_environment(environment, workspace_root)


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + f"\n...[truncated {len(text) - limit} chars]"
