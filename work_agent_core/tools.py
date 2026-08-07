from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
import difflib
import json
import subprocess

from .progress import emit_tool_progress, next_command_id


ToolHandler = Callable[[dict[str, Any]], str]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler
    provider_id: str = "local"
    provider_kind: str = "local"
    metadata: dict[str, Any] = field(default_factory=dict)

    def render_for_prompt(self) -> str:
        return json.dumps(
            {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
            ensure_ascii=False,
        )


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as error:
            available = ", ".join(sorted(self._tools))
            raise KeyError(f"Unknown tool {name!r}. Available tools: {available}") from error

    def list(self) -> list[Tool]:
        return [self._tools[name] for name in sorted(self._tools)]

    def prompt_block(self) -> str:
        return "\n".join(tool.render_for_prompt() for tool in self.list())


class WorkspaceFiles:
    def __init__(self, workspace_root: str | Path) -> None:
        self.workspace_root = Path(workspace_root).resolve()

    def resolve(self, raw_path: str) -> Path:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = self.workspace_root / path
        resolved = path.resolve()
        if self.workspace_root not in (resolved, *resolved.parents):
            raise ValueError(f"Path is outside workspace: {raw_path}")
        return resolved

    def read_text(self, args: dict[str, Any]) -> str:
        path = self.resolve(str(args["path"]))
        max_chars = int(args.get("max_chars") or 12000)
        text = path.read_text(encoding=args.get("encoding") or "utf-8")
        if len(text) > max_chars:
            return text[:max_chars] + f"\n\n[truncated to {max_chars} chars from {len(text)}]"
        return text

    def write_text(self, args: dict[str, Any]) -> str:
        path = self.resolve(str(args["path"]))
        content = str(args["content"])
        encoding = str(args.get("encoding") or "utf-8")
        result = self._write_text_with_activity(
            path=path,
            content=content,
            encoding=encoding,
            tool_name="write_text_file",
            command_label=f"write_text_file --path {self._display_path(path)} --chars {len(content)}",
        )
        return f"Wrote {result}"

    def edit_text(self, args: dict[str, Any]) -> str:
        path = self.resolve(str(args["path"]))
        encoding = str(args.get("encoding") or "utf-8")
        old_text = str(args["old_text"])
        new_text = str(args.get("new_text") or "")
        replace_all = bool(args.get("replace_all"))
        expected_replacements = int(args.get("expected_replacements") or 1)
        if not old_text:
            raise ValueError("old_text 不能为空。")
        if expected_replacements < 1:
            raise ValueError("expected_replacements 必须大于等于 1。")
        if not path.exists():
            raise FileNotFoundError(f"文件不存在：{args['path']}")
        previous_content = path.read_text(encoding=encoding, errors="replace")
        actual_matches = previous_content.count(old_text)
        if actual_matches == 0:
            raise ValueError("没有找到 old_text，未修改文件。请先读取目标片段，确保完全匹配。")
        if actual_matches != expected_replacements:
            raise ValueError(
                f"old_text 匹配到 {actual_matches} 处，但 expected_replacements={expected_replacements}。"
                "为避免误改，已停止。"
            )
        if replace_all:
            next_content = previous_content.replace(old_text, new_text)
            replacements = actual_matches
        else:
            next_content = previous_content.replace(old_text, new_text, 1)
            replacements = 1
        self._write_text_with_activity(
            path=path,
            content=next_content,
            encoding=encoding,
            tool_name="edit_text_file",
            command_label=(
                f"edit_text_file --path {self._display_path(path)} "
                f"--replacements {replacements}"
            ),
        )
        return f"Edited {path}: {replacements} replacement(s)"

    def apply_unified_patch(self, args: dict[str, Any]) -> str:
        patch_text = str(args["patch"])
        if not patch_text.strip():
            raise ValueError("patch 不能为空。")
        touched_paths = validate_unified_patch_paths(patch_text, self.workspace_root)
        additions, deletions = count_patch_changes(patch_text)
        file_changes = count_patch_file_changes(patch_text)
        activity_id = next_command_id()
        display_paths = ", ".join(touched_paths) or "unknown"
        emit_tool_progress(
            {
                "event": "activity",
                "id": activity_id,
                "phase": "action",
                "title": "应用补丁",
                "detail": display_paths,
                "content": clip_text(patch_text, 12000, suffix="\n… patch 预览已截断。"),
                "activity_type": "file_edit",
                "command": f"apply_unified_patch --files {display_paths}",
                "command_status": "running",
                "tool_name": "apply_unified_patch",
                "file_path": display_paths,
                "additions": additions,
                "deletions": deletions,
                "file_changes": file_changes,
            }
        )
        result = subprocess.run(
            ["git", "apply", "--whitespace=nowarn"],
            input=patch_text,
            cwd=self.workspace_root,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            emit_tool_progress(
                {
                    "event": "activity_delta",
                    "id": activity_id,
                    "phase": "error",
                    "title": "应用补丁",
                    "content": f"\n✗ patch 应用失败：{clip_text(result.stderr or result.stdout, 2000)}\n",
                    "activity_type": "file_edit",
                    "command_status": "error",
                    "tool_name": "apply_unified_patch",
                    "file_path": display_paths,
                    "additions": additions,
                    "deletions": deletions,
                    "file_changes": file_changes,
                }
            )
            raise RuntimeError(f"patch 应用失败：{result.stderr or result.stdout}")
        emit_tool_progress(
            {
                "event": "activity_delta",
                "id": activity_id,
                "phase": "action",
                "title": "应用补丁",
                "content": "\n✓ patch 已应用。\n",
                "activity_type": "file_edit",
                "command_status": "success",
                "tool_name": "apply_unified_patch",
                "file_path": display_paths,
                "additions": additions,
                "deletions": deletions,
                "file_changes": file_changes,
            }
        )
        return json.dumps(
            {
                "ok": True,
                "files": touched_paths,
                "additions": additions,
                "deletions": deletions,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
            ensure_ascii=False,
            indent=2,
        )

    def _write_text_with_activity(
        self,
        *,
        path: Path,
        content: str,
        encoding: str,
        tool_name: str,
        command_label: str,
    ) -> Path:
        try:
            display_path = self._display_path(path)
        except ValueError:
            display_path = str(path)
        previous_content = ""
        if path.exists():
            previous_content = path.read_text(encoding=encoding, errors="replace")
        additions, deletions, diff_preview = build_text_edit_preview(
            previous_content=previous_content,
            next_content=content,
            display_path=display_path,
        )
        activity_id = next_command_id()
        emit_tool_progress(
            {
                "event": "activity",
                "id": activity_id,
                "phase": "action",
                "title": f"已编辑 {Path(display_path).name}",
                "detail": display_path,
                "content": diff_preview,
                "activity_type": "file_edit",
                "command": command_label,
                "command_status": "running",
                "tool_name": tool_name,
                "file_path": display_path,
                "additions": additions,
                "deletions": deletions,
            }
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_text(content, encoding=encoding)
        except Exception as error:
            emit_tool_progress(
                {
                    "event": "activity_delta",
                    "id": activity_id,
                    "phase": "error",
                    "title": f"已编辑 {Path(display_path).name}",
                    "content": f"\n✗ 写入失败：{type(error).__name__}: {error}\n",
                    "activity_type": "file_edit",
                    "command_status": "error",
                    "tool_name": tool_name,
                    "file_path": display_path,
                    "additions": additions,
                    "deletions": deletions,
                }
            )
            raise
        emit_tool_progress(
            {
                "event": "activity_delta",
                "id": activity_id,
                "phase": "action",
                "title": f"已编辑 {Path(display_path).name}",
                "content": "",
                "activity_type": "file_edit",
                "command_status": "success",
                "tool_name": tool_name,
                "file_path": display_path,
                "additions": additions,
                "deletions": deletions,
            }
        )
        return path

    def _display_path(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.workspace_root))
        except ValueError:
            return str(path)

    def list_files(self, args: dict[str, Any]) -> str:
        directory = self.resolve(str(args.get("path") or "meet_files"))
        max_files = int(args.get("max_files") or 80)
        files = []
        for item in sorted(directory.rglob("*")):
            if item.is_file():
                files.append(str(item.relative_to(self.workspace_root)))
            if len(files) >= max_files:
                break
        return "\n".join(files)


def build_text_edit_preview(
    *,
    previous_content: str,
    next_content: str,
    display_path: str,
    max_lines: int = 220,
    max_chars: int = 12000,
) -> tuple[int, int, str]:
    previous_lines = previous_content.splitlines()
    next_lines = next_content.splitlines()
    diff_lines = list(
        difflib.unified_diff(
            previous_lines,
            next_lines,
            fromfile=f"a/{display_path}",
            tofile=f"b/{display_path}",
            lineterm="",
            n=3,
        )
    )
    additions = sum(1 for line in diff_lines if line.startswith("+") and not line.startswith("+++"))
    deletions = sum(1 for line in diff_lines if line.startswith("-") and not line.startswith("---"))
    if not diff_lines:
        return 0, 0, "文件内容没有变化。"
    clipped = False
    if len(diff_lines) > max_lines:
        diff_lines = diff_lines[:max_lines]
        clipped = True
    preview = "\n".join(diff_lines)
    if len(preview) > max_chars:
        preview = preview[:max_chars].rstrip()
        clipped = True
    if clipped:
        preview += "\n… diff 预览已截断，完整内容已写入文件。"
    return additions, deletions, preview


def validate_unified_patch_paths(patch_text: str, workspace_root: Path) -> list[str]:
    paths: list[str] = []
    for line in patch_text.splitlines():
        if not line.startswith(("--- ", "+++ ")):
            continue
        raw_path = line[4:].strip().split("\t", 1)[0]
        if line.startswith("+++ ") and raw_path == "/dev/null":
            raise ValueError("apply_unified_patch 不允许删除整个文件；如确需删除，请使用带用户确认的终端操作。")
        if raw_path == "/dev/null":
            continue
        if raw_path.startswith(("a/", "b/")):
            raw_path = raw_path[2:]
        candidate = Path(raw_path)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(f"patch 包含不安全路径：{raw_path}")
        resolved = (workspace_root / candidate).resolve()
        if workspace_root not in (resolved, *resolved.parents):
            raise ValueError(f"patch 路径超出工作区：{raw_path}")
        if raw_path not in paths:
            paths.append(raw_path)
    if not paths:
        raise ValueError("patch 中没有可识别的工作区文件路径。")
    return paths


def count_patch_changes(patch_text: str) -> tuple[int, int]:
    additions = 0
    deletions = 0
    for line in patch_text.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            additions += 1
        elif line.startswith("-") and not line.startswith("---"):
            deletions += 1
    return additions, deletions


def count_patch_file_changes(patch_text: str) -> list[dict[str, Any]]:
    """Return per-file line totals for a validated unified patch."""

    changes: dict[str, dict[str, Any]] = {}
    current_path = ""
    for line in patch_text.splitlines():
        if line.startswith("+++ "):
            raw_path = line[4:].strip().split("\t", 1)[0]
            if raw_path == "/dev/null":
                current_path = ""
                continue
            current_path = raw_path[2:] if raw_path.startswith(("a/", "b/")) else raw_path
            changes.setdefault(
                current_path,
                {"file_path": current_path, "additions": 0, "deletions": 0},
            )
            continue
        if not current_path:
            continue
        if line.startswith("+") and not line.startswith("+++"):
            changes[current_path]["additions"] += 1
        elif line.startswith("-") and not line.startswith("---"):
            changes[current_path]["deletions"] += 1
    return list(changes.values())


def clip_text(value: str, limit: int, *, suffix: str = "\n...[truncated]") -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + suffix


def register_file_tools(registry: ToolRegistry, workspace_root: str | Path) -> None:
    files = WorkspaceFiles(workspace_root)
    registry.register(
        Tool(
            name="read_text_file",
            description="Read a UTF-8 text file from the workspace.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "max_chars": {"type": "integer", "default": 12000},
                },
                "required": ["path"],
            },
            handler=files.read_text,
        )
    )
    registry.register(
        Tool(
            name="write_text_file",
            description=(
                "Write a complete UTF-8 text file under the workspace. "
                "For small changes to existing files, prefer edit_text_file or apply_unified_patch so the model does not rewrite the whole file."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
            handler=files.write_text,
        )
    )
    registry.register(
        Tool(
            name="edit_text_file",
            description=(
                "Make a precise exact-text replacement in an existing UTF-8 text file. "
                "Use this for small edits to prompts, skills, configs, Markdown, or source code instead of rewriting the entire file."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string", "description": "Exact text to replace. Read the file first and copy the target block exactly."},
                    "new_text": {"type": "string"},
                    "expected_replacements": {"type": "integer", "default": 1},
                    "replace_all": {"type": "boolean", "default": False},
                },
                "required": ["path", "old_text", "new_text"],
            },
            handler=files.edit_text,
        )
    )
    registry.register(
        Tool(
            name="apply_unified_patch",
            description=(
                "Apply a standard unified diff patch to workspace files. "
                "Use for multi-line or multi-file code edits when exact replacement is awkward. Paths must stay inside the workspace."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "patch": {
                        "type": "string",
                        "description": "Unified diff text with a/... and b/... file paths.",
                    }
                },
                "required": ["patch"],
            },
            handler=files.apply_unified_patch,
        )
    )
    registry.register(
        Tool(
            name="list_workspace_files",
            description=(
                "List files under a specific workspace directory. "
                "Use only when no exact file path is available; prefer reading explicit paths from the user message, attachments, or prior tool output. "
                "Do not list the workspace root '.' unless the user explicitly asks to inspect the whole project."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "default": "meet_files"},
                    "max_files": {"type": "integer", "default": 80},
                },
            },
            handler=files.list_files,
        )
    )
