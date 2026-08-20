from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
import ast
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
from .execution.workspace import DEFAULT_EXCLUDED_NAMES
from .progress import current_tool_cancel_check, emit_tool_progress
from .runtime_env import apply_project_agent_environment, project_agent_python, project_node
from .tools import Tool, ToolRegistry, WorkspaceFiles


AUTO_ALLOW_COMMANDS = {
    "pwd", "ls", "rg", "cat", "head", "tail", "wc", "file", "stat", "du", "which",
    # Everyday read-only shell vocabulary. Leaving these out did not add a
    # boundary — the sandbox already denies writes outside the workspace — it
    # only made the terminal unusable for the small jobs it exists for.
    "grep", "egrep", "fgrep", "sed", "awk", "sort", "uniq", "cut", "tr", "nl",
    "echo", "printf", "date", "basename", "dirname", "realpath", "readlink",
    "diff", "cmp", "md5", "shasum", "seq", "expr", "env", "uname", "id",
    "tree", "column", "jq", "xxd", "od", "strings", "command", "type", "true", "false",
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
        sandbox_auto_allow: bool = False,
        sandbox_in_place: bool = True,
    ) -> None:
        self.sandbox_in_place = bool(sandbox_in_place)
        # When the command will run under an OS-enforced sandbox that already
        # denies network, denies writes outside the private snapshot and denies
        # reads outside the runtime, a second gate keyed on the program name
        # adds no boundary — it only adds a click. Auto-allow keeps the fixed
        # circuit breakers and lets everything the sandbox contains just run.
        self.sandbox_auto_allow = bool(sandbox_auto_allow)
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
        try:
            stages = split_shell_pipeline(command_text)
        except ValueError as error:
            return json.dumps(
                {
                    "ok": False,
                    "status": "denied",
                    "risk_category": "SYSTEM",
                    "reason": str(error),
                    "command": command_text,
                    "cwd": str(cwd),
                },
                ensure_ascii=False,
                indent=2,
            )
        argv = stages[0]
        decision = self._decide_pipeline(stages, cwd)
        if self.sandbox_auto_allow:
            decision = self._sandbox_auto_allowed_pipeline(stages, decision)
        isolation_note = snapshot_isolation_note(command_text, cwd)
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
                    **({"isolated_workspace_note": isolation_note} if isolation_note else {}),
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
        managed_argv = self._shell_argv(command_text, stages)
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
                delivery_mode=(
                    "discard_changes"
                    if decision.risk_category == "READ"
                    else "apply_after_validation"
                ),
                # Run where the data is. The sandbox still denies network and
                # every write outside the workspace; copying the tree first only
                # hid the account's own files from the command.
                in_place=self.sandbox_in_place,
                reason=f"运行工作区内命令：{Path(argv[0]).name}",
            ),
            source_root=cwd,
            on_event=self._on_execution_event,
            cancel_check=current_tool_cancel_check(),
        )
        self._emit_execution_result(execution)
        stdout, stderr = self._execution_output(execution)
        succeeded = execution.status.value == "succeeded"
        stdout_text, stdout_path = spill_output(
            stdout,
            20000,
            workspace_root=self.workspace_root,
            execution_id=execution.execution_id,
            stream="stdout",
        )
        stderr_text, stderr_path = spill_output(
            stderr,
            12000,
            workspace_root=self.workspace_root,
            execution_id=execution.execution_id,
            stream="stderr",
        )

        return json.dumps(
            {
                "ok": succeeded,
                "status": "executed" if succeeded else "failed",
                **({"isolated_workspace_note": isolation_note} if isolation_note else {}),
                "permission": decision.status,
                "risk_category": decision.risk_category,
                "command": command_text,
                "cwd": str(cwd),
                "returncode": execution.process.exit_code if execution.process else None,
                "stdout": stdout_text,
                "stderr": stderr_text,
                **({"stdout_full_path": stdout_path} if stdout_path else {}),
                **({"stderr_full_path": stderr_path} if stderr_path else {}),
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
        snippet = inline_python_snippet(argv)
        if snippet is not None:
            read_only, reason = read_only_python_snippet(snippet)
            if read_only and paths_stay_in_workspace(argv[3:], cwd, self.workspace_root):
                return ShellDecision("allow", reason + "。", "READ")
        if executable in ASK_COMMANDS and len(argv) == 2 and argv[1] in VERSION_FLAGS:
            return ShellDecision("allow", f"{executable} 版本查看属于低风险命令。", "READ")
        if executable in ASK_COMMANDS:
            if paths_stay_in_workspace(argv[1:], cwd, self.workspace_root):
                return ShellDecision("ask", f"{executable} 可能执行脚本、生成文件、安装依赖或修改文件，需要用户确认。", risk_category_for_executable(executable))
            return ShellDecision("deny", "命令参数包含工作区外路径。", "SYSTEM")
        if not paths_stay_in_workspace(argv[1:], cwd, self.workspace_root):
            return ShellDecision("deny", "未知命令的参数包含工作区外路径。", "SYSTEM")
        return ShellDecision("ask", f"{executable} 不在白名单内，需要用户确认。", risk_category_for_executable(executable))

    def _decide_pipeline(self, stages: list[list[str]], cwd: Path) -> ShellDecision:
        """Apply the per-program policy to every stage and take the strictest.

        A pipeline is only as safe as its most dangerous stage, and a redirect
        is a write no matter which stage carries it.
        """

        strictest = ShellDecision("allow", "全部命令均为只读白名单。", "READ")
        rank = {"allow": 0, "ask": 1, "deny": 2}
        for stage in stages:
            for target in redirect_targets(stage):
                if not paths_stay_in_workspace([target], cwd, self.workspace_root):
                    return ShellDecision("deny", f"重定向目标 {target} 在工作区之外。", "SYSTEM")
            cleaned = strip_redirects(stage)
            if not cleaned:
                continue
            decision = self._decide(cleaned, cwd)
            if rank[decision.status] > rank[strictest.status]:
                strictest = decision
            elif decision.status == strictest.status and decision.risk_category != "READ":
                strictest = decision
        # A redirect writes a file whatever else the line does, so a pipeline of
        # otherwise read-only stages is still a modification.
        if strictest.status == "allow" and any(redirect_targets(stage) for stage in stages):
            return ShellDecision("ask", "命令通过重定向写入工作区文件，需要确认。", "MODIFY")
        return strictest

    def _sandbox_auto_allowed_pipeline(
        self, stages: list[list[str]], decision: ShellDecision
    ) -> ShellDecision:
        for stage in stages:
            cleaned = strip_redirects(stage)
            if cleaned:
                decision = self._sandbox_auto_allowed(cleaned, decision)
                if decision.status != "allow":
                    return decision
        return decision

    def _shell_argv(self, command_text: str, stages: list[list[str]]) -> list[str]:
        """Run the line through a real shell, with project runtimes on PATH.

        A single stage keeps its pinned interpreter so ``python`` still means the
        project venv; anything with shell syntax goes to ``/bin/sh`` and relies
        on the sandbox environment's PATH for the same guarantee.
        """

        if len(stages) == 1 and not redirect_targets(stages[0]):
            return self._managed_runtime_argv(stages[0])
        return ["/bin/sh", "-c", command_text]

    def _sandbox_auto_allowed(self, argv: list[str], decision: ShellDecision) -> ShellDecision:
        """Let the sandbox be the permission, keeping the fixed circuit breakers.

        Deletion stays behind a prompt no matter what: a wrong delete inside the
        snapshot still gets applied back, and that is the one class the sandbox
        does not make recoverable. Package managers stay behind a prompt because
        their whole purpose is to persist state.
        """

        if decision.status != "ask":
            return decision
        executable = Path(argv[0]).name
        if decision.risk_category in {"DELETE", "SYSTEM"}:
            return decision
        if executable in NEVER_SANDBOX_AUTO_ALLOWED:
            return decision
        return ShellDecision(
            "allow",
            f"{executable} 在隔离沙箱内运行：禁网络、禁写工作区外、改动经校验后才回写。",
            decision.risk_category,
        )


# Modules whose mere import hands the snippet a way out of the read-only story.
EFFECTFUL_MODULES = frozenset({
    "subprocess", "shutil", "socket", "urllib", "urllib2", "requests", "httpx",
    "http", "ftplib", "smtplib", "telnetlib", "ctypes", "multiprocessing",
    "webbrowser", "pip", "setuptools", "distutils", "venv", "signal", "pty",
})

# Attribute names that write, delete, execute or reach the network. Unknown
# attributes stay allowed: this is a denylist over observed effects, backed by
# the sandbox and by classifying the snippet READ so its changes are discarded.
EFFECTFUL_ATTRIBUTES = frozenset({
    "write", "writelines", "write_text", "write_bytes", "truncate", "flush",
    "unlink", "remove", "removedirs", "rmdir", "rmtree", "mkdir", "makedirs",
    "rename", "replace", "chmod", "chown", "symlink_to", "hardlink_to", "touch",
    "system", "popen", "spawn", "spawnl", "spawnv", "execv", "execve",
    "run", "call", "check_call", "check_output", "Popen",
    "urlopen", "connect", "sendall", "send", "request",
    "save", "to_csv", "to_excel", "to_json", "dump", "savefig", "commit",
})

EFFECTFUL_BUILTINS = frozenset({"exec", "eval", "compile", "__import__", "input", "breakpoint"})

READ_FILE_MODES = frozenset({"r", "rb", "rt", "br", "tr"})


def read_only_python_snippet(code: str) -> tuple[bool, str]:
    """Decide whether a ``python -c`` body can only read.

    The policy otherwise sees nothing but the program name, so every inline
    probe — ``import x; print(x.__version__)`` — lands in the same bucket as an
    arbitrary script and costs the user a click. Reading the snippet lets the
    decision follow what the code actually does. Anything not provably
    read-only keeps its existing ``ask`` treatment.
    """

    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        return False, f"无法解析为 Python 代码：{error.msg}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in EFFECTFUL_MODULES:
                    return False, f"导入了可能产生副作用的模块 {root}"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in EFFECTFUL_MODULES:
                return False, f"导入了可能产生副作用的模块 {root}"
        elif isinstance(node, ast.Attribute):
            if node.attr in EFFECTFUL_ATTRIBUTES:
                return False, f"调用了可能写入或执行的方法 {node.attr}"
        elif isinstance(node, ast.Name):
            if node.id in EFFECTFUL_BUILTINS:
                return False, f"使用了动态执行内建函数 {node.id}"
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "open":
                mode = _literal_open_mode(node)
                if mode is None:
                    return False, "open() 的模式不是字面量，无法确认是只读"
                if mode not in READ_FILE_MODES:
                    return False, f"open() 使用了写入模式 {mode!r}"
    return True, "该 python -c 代码体只读取，不写入、不执行子进程、不联网"


def _literal_open_mode(node: ast.Call) -> str | None:
    if len(node.args) >= 2:
        second = node.args[1]
        return second.value if isinstance(second, ast.Constant) and isinstance(second.value, str) else None
    for keyword in node.keywords:
        if keyword.arg == "mode":
            value = keyword.value
            return value.value if isinstance(value, ast.Constant) and isinstance(value.value, str) else None
    return "r"


def inline_python_snippet(argv: list[str]) -> str | None:
    """Return the code body of a ``python -c <code>`` invocation."""
    if len(argv) < 3 or Path(argv[0]).name not in {"python", "python3"}:
        return None
    return argv[2] if argv[1] == "-c" else None



PIPELINE_SEPARATORS = ("&&", "||", "|", ";", "\n")
REDIRECT_OPERATORS = (">>", ">", "<", "2>", "2>>", "&>")


def split_shell_pipeline(command_text: str) -> list[list[str]]:
    """Decompose a shell line into the simple commands it will run.

    The policy needs to see every program an invocation reaches, but refusing
    the shell outright turned ordinary work — ``grep ... | head``, ``ls | wc -l``
    — into a denial. Splitting on the operators and checking each stage keeps
    the same per-program policy while letting the shell be a shell.

    Command substitution stays refused: its contents are decided at run time,
    so no static check can see the program it would reach.
    """

    if "$(" in command_text or "`" in command_text:
        raise ValueError("命令替换 $(...) 或反引号会在运行时才决定实际程序，静态策略无法审查。")
    lexer = shlex.shlex(command_text, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    try:
        tokens = list(lexer)
    except ValueError as error:
        raise ValueError(f"命令无法解析：{error}") from error

    stages: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        if token in PIPELINE_SEPARATORS or token in {"&"}:
            if current:
                stages.append(current)
            current = []
            continue
        current.append(token)
    if current:
        stages.append(current)
    if not stages:
        raise ValueError("命令为空。")
    return stages


def redirect_targets(argv: list[str]) -> list[str]:
    """Paths a stage would write through a redirect."""
    targets: list[str] = []
    for index, token in enumerate(argv):
        if token in REDIRECT_OPERATORS and index + 1 < len(argv):
            targets.append(argv[index + 1])
        else:
            for operator in (">>", ">"):
                if token.startswith(operator) and len(token) > len(operator):
                    targets.append(token[len(operator):])
                    break
    return targets


def strip_redirects(argv: list[str]) -> list[str]:
    """Drop redirect operators and their targets from a stage's argv."""
    cleaned: list[str] = []
    skip_next = False
    for token in argv:
        if skip_next:
            skip_next = False
            continue
        if token in REDIRECT_OPERATORS:
            skip_next = True
            continue
        if any(token.startswith(op) and len(token) > len(op) for op in (">>", ">")):
            continue
        cleaned.append(token)
    return cleaned


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
    sandbox_auto_allow: bool = False,
) -> None:
    shell = ShellExecutionTools(
        workspace_root,
        execution_orchestrator=execution_orchestrator,
        runtime_workspace_root=runtime_workspace_root,
        account_id=account_id,
        turn_id=turn_id,
        conversation_id=conversation_id,
        project_id=project_id,
        sandbox_auto_allow=sandbox_auto_allow,
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
                "The command runs in the real workspace, so user data under meet_files/ is visible; the sandbox denies network "
                "access and every write outside the workspace. "
                "Do not use this tool to read, concatenate, create, or edit text/Markdown files; use the dedicated workspace file tools. "
                "Prefer an existing core file tool or a skill tool whenever one can do the job; reach for this tool only when a "
                "terminal is genuinely required. When approval is needed, call the tool and let it return approval_required — never "
                "ask the user in prose to reply with 确认 or 允许执行, never simulate an approval in a content-only message, never "
                "describe a command you have not actually issued as pending approval, and never claim or retry around an approval "
                "decision. A denied command stays denied. If the isolation backend is unavailable, do not request approval for a "
                "terminal action that is certain to fail; if a file tool can deliver the same result, switch to it and continue. "
                "Pipes, redirection, && / || / ; chains and globs are supported and run through a real shell; every stage is "
                "checked against the same policy and a redirect target outside the workspace is refused. Command substitution "
                "$(...) and backticks are refused because the program they reach is only decided at run time. "
                "stdout is capped at 20000 chars and stderr at 12000; when a stream is cut, the full text is written to a "
                "workspace file and its path is returned as stdout_full_path / stderr_full_path — read that file when you need "
                "the part that was cut. "
                "A nonzero returncode is a failed command; do not report "
                "a verification suite as fully passed unless every required check succeeded. If native isolation is unavailable, "
                "the command fails closed and never silently runs on the host."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "A shell command line. Pipes, redirection, && / || / ; chains and globs work. Command substitution $(...) and backticks are refused. Treat any nonzero returncode as a failed check.",
                    },
                    "cwd": {"type": "string", "default": "."},
                    "timeout_seconds": {"type": "integer", "default": 120},
                },
                "required": ["command"],
            },
            handler=shell.execute,
        )
    )


def snapshot_isolation_note(command_text: str, cwd: Path) -> str:
    """Warn when a command names a directory the isolated copy will not contain.

    ``shell_exec`` runs against a private snapshot of the project tree, and the
    snapshot skips heavy or private trees such as ``meet_files/``. Inside the
    sandbox those paths simply do not exist, so the command comes back with a
    bare ``No such file or directory`` that reads as if the data were missing
    from the project. Naming the skipped directories keeps the failure legible
    and points at the workspace file tools that do see them.
    """

    hidden = [
        name
        for name in sorted(DEFAULT_EXCLUDED_NAMES)
        if not name.startswith(".")
        and (cwd / name).is_dir()
        and mentions_path_segment(command_text, name)
    ]
    if not hidden:
        return ""
    return (
        f"命令引用了 {', '.join(hidden)}，但这些目录不会同步进隔离执行副本，"
        "在沙箱内表现为“文件不存在”。读写用户数据请改用 read_text_file、write_text_file、"
        "list_workspace_files 或对应技能工具。"
    )


def mentions_path_segment(command_text: str, name: str) -> bool:
    """Match ``name`` used as a leading path segment, including inside quotes."""
    start = 0
    while True:
        index = command_text.find(name, start)
        if index < 0:
            return False
        start = index + len(name)
        before = command_text[index - 1] if index else " "
        after = command_text[start] if start < len(command_text) else " "
        if before not in "/-_." and not before.isalnum() and (after == "/" or not (after.isalnum() or after in "-_.")):
            return True


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


# Interpreters the project already runs. A script under these is bounded by the
# sandbox and the workspace check; an unknown binary is not.
REVIEWABLE_PROJECT_RUNTIMES = frozenset({"python", "python3", "node"})
# Installers persist state on purpose, so the sandbox containing them is not the
# same as the change being wanted.
NEVER_SANDBOX_AUTO_ALLOWED = frozenset({"pip", "pip3", "conda", "npm", "pnpm", "npx", "rm", "rmdir"})
NEVER_REVIEWABLE_EXECUTABLES = frozenset({"pip", "pip3", "conda", "npm", "pnpm", "npx"})


def is_model_reviewable_command(argv: list[str], decision: ShellDecision) -> bool:
    """Return the reviewer boundary, which is wider than the delegate's.

    ``ask`` means the user may approve an exact action; it does not mean a
    model reviewer may. The reviewer additionally gets workspace-confined
    scripts under the project's own interpreters, because refusing those made
    "review for me" unable to clear the ordinary work it exists for. Package
    installation, networking, deletion and unknown binaries stay outside: those
    are the cases where a wrong call is not recoverable by discarding changes.
    """

    if is_auto_approvable_command(argv, decision):
        return True
    if decision.status != "ask" or not argv:
        return False
    if decision.risk_category in {"DELETE", "SYSTEM", "NETWORK"}:
        return False
    executable = Path(argv[0]).name
    if executable in NEVER_REVIEWABLE_EXECUTABLES:
        return False
    return executable in REVIEWABLE_PROJECT_RUNTIMES


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


SHELL_OUTPUT_DIR = Path("tmp") / "shell_output"


def spill_output(
    text: str,
    limit: int,
    *,
    workspace_root: Path,
    execution_id: str,
    stream: str,
) -> tuple[str, str]:
    """Truncate for the model, but keep the rest reachable.

    A cut that leaves no way back is worse than a long result: the model cannot
    tell what it did not see. Everything past the limit goes to a file under the
    workspace, so the returned path can be read with the ordinary file tools.
    """

    if len(text) <= limit:
        return text, ""
    relative = SHELL_OUTPUT_DIR / f"{execution_id or 'run'}.{stream}.txt"
    target = workspace_root / relative
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    except OSError:
        # Losing the spill file must not lose the command result.
        return truncate(text, limit), ""
    kept = text[:limit].rstrip()
    path_text = relative.as_posix()
    marker = (
        f"\n...[truncated {len(text) - limit} chars of {len(text)}. "
        f"Full output saved to {path_text} — read it if you need the rest.]"
    )
    return kept + marker, path_text
