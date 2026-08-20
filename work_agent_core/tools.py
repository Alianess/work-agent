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


_DEFAULT_FILE_CHANGE_HANDLER: Callable[[Path], None] | None = None


def set_default_file_change_handler(handler: Callable[[Path], None] | None) -> None:
    """所有 WorkspaceFiles 写入的默认通知目标。

    十五处地方各自构造 WorkspaceFiles，靠"记得传 on_file_changed"必然漏——
    实测会议纪要技能就漏了，它写出来的 ASR 转写稿和纪要从来没进过检索索引。
    默认值放在这里，新增技能不必知道索引的存在也不会漏。
    """

    global _DEFAULT_FILE_CHANGE_HANDLER
    _DEFAULT_FILE_CHANGE_HANDLER = handler


class WorkspaceFiles:
    def __init__(
        self,
        workspace_root: str | Path,
        *,
        on_file_changed: Callable[[Path], None] | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.on_file_changed = on_file_changed

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
        if path.is_dir():
            return self._list_directory(path, args)
        if not path.is_file():
            return f"没有这个文件：{args['path']}"
        image_result = self._read_image(path)
        if image_result is not None:
            return image_result
        max_chars = int(args.get("max_chars") or 12000)
        offset = max(0, int(args.get("offset") or 0))
        text = path.read_text(encoding=args.get("encoding") or "utf-8")
        total = len(text)
        window = text[offset : offset + max_chars]
        if offset == 0 and total <= max_chars:
            return window
        # A cut with no way back leaves the model unable to tell what it missed.
        next_offset = offset + len(window)
        remaining = total - next_offset
        note = f"\n\n[chars {offset}-{next_offset} of {total}"
        if remaining > 0:
            note += f"; {remaining} remaining — read again with offset={next_offset}"
        note += "]"
        return window + note

    def _list_directory(self, directory: Path, args: dict[str, Any]) -> str:
        """目录也是 read 的一部分——和 Pi 一致：一个 read 吃文件、图片和目录。"""

        max_files = int(args.get("max_files") or 80)
        files: list[str] = []
        for item in sorted(directory.rglob("*")):
            if item.is_file():
                files.append(self._display_path(item))
            if len(files) >= max_files:
                break
        if not files:
            return f"目录是空的：{args.get('path')}"
        return "\n".join(files)

    def _read_image(self, path: Path) -> str | None:
        """图片走附件通道，不是这里的返回值。

        OpenAI 兼容的 tool 消息只能是字符串，塞不下图片，所以"看见"这件事只能
        由 harness 完成：内容块交给循环，循环把它作为一条用户消息注入，模型在
        下一步才真正看到。对调用方来说仍然只有一个 read_file。
        """

        import mimetypes

        from .images import encode_image_for_model
        from .progress import offer_tool_attachment

        mime_type = mimetypes.guess_type(path.name)[0] or ""
        if not mime_type.startswith("image/"):
            return None
        encoded_mime, encoded = encode_image_for_model(path, mime_type)
        if not encoded:
            return f"读不了这张图：{path.name}"
        delivered = offer_tool_attachment(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{encoded_mime};base64,{encoded}"},
            }
        )
        if not delivered:
            return (
                f"[当前模型或运行环境不支持图片，{path.name} 未进入本次请求。]"
            )
        return f"已载入图片 {path.name}（{encoded_mime}），在下一步就能看到它。"

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
        # 一个 edit 吃两种改法：小改传 old_text/new_text 精确替换；
        # 跨多处或多文件传 patch（unified diff）。和 Pi 的单 edit 一致，
        # patch 只是我们保留的多文件原子改能力。
        patch_arg = str(args.get("patch") or "")
        if patch_arg.strip():
            # 不 strip 原文：结尾换行是 patch 格式的一部分，裁掉 git apply 就报 corrupt。
            return self.apply_unified_patch({"patch": patch_arg})
        if not str(args.get("path") or "").strip():
            raise ValueError("edit_text_file 需要 path + old_text + new_text，或者 patch。")
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
        for relative_path in touched_paths:
            self._notify_file_changed(self.workspace_root / relative_path)
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
        self._notify_file_changed(path)
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

    def _notify_file_changed(self, path: Path) -> None:
        handler = self.on_file_changed or _DEFAULT_FILE_CHANGE_HANDLER
        if handler is None:
            return
        try:
            handler(path)
        except Exception:
            # File writes are authoritative. Index maintenance must never turn
            # a successful user edit into a failed tool call.
            return

    def _display_path(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.workspace_root))
        except ValueError:
            return str(path)


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


def register_file_tools(
    registry: ToolRegistry,
    workspace_root: str | Path,
    *,
    on_file_changed: Callable[[Path], None] | None = None,
) -> None:
    files = WorkspaceFiles(workspace_root, on_file_changed=on_file_changed)
    registry.register(
        Tool(
            name="read_file",
            description=(
                "Read a path from the workspace: text files come back as text, images (jpg, png, "
                "gif, webp, bmp) are attached to the conversation and become visible on the next "
                "step, and directories return a listing of files under them (at most max_files). "
                "For text, returns at most max_chars from offset; when more remains, the result "
                "says how much and which offset to read next. "
                "When the user message, an attachment, earlier conversation, or a previous tool result already "
                "names an exact path, read that path directly instead of scanning the workspace to confirm it; "
                "list a directory only when no exact path is known, and pick the smallest one that can hold the answer. "
                "Reading, joining, tidying or rewriting plain text and Markdown always goes through the workspace "
                "file tools, never through python or a terminal command."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "max_chars": {"type": "integer", "default": 12000},
                    "offset": {
                        "type": "integer",
                        "default": 0,
                        "description": "Character offset to start from; use the value the previous read reported.",
                    },
                    "max_files": {
                        "type": "integer",
                        "default": 80,
                        "description": "Only for directories: maximum number of files to list.",
                    },
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
                "For changes to existing files, prefer edit_text_file (exact replacement or unified patch) so the model does not rewrite the whole file. "
                "Do not resend a long script or long document through repeated whole-file writes; split it into smaller modules or "
                "staged patches. If a tool result reports finish_reason=length or truncated arguments, immediately send less in one "
                "call — that is a size problem, not a path/content nesting problem, so do not retry the same payload unchanged."
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
                "Edit UTF-8 text files. Two modes: pass path + old_text + new_text for a precise exact-text "
                "replacement in one file (small edits to prompts, skills, configs, Markdown, or source code); "
                "or pass patch (a standard unified diff with a/... and b/... paths) for multi-line or multi-file "
                "changes. Never rewrite an existing file wholesale when one of these fits."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string", "description": "Exact text to replace. Read the file first and copy the target block exactly."},
                    "new_text": {"type": "string"},
                    "expected_replacements": {"type": "integer", "default": 1},
                    "replace_all": {"type": "boolean", "default": False},
                    "patch": {
                        "type": "string",
                        "description": "Unified diff text for multi-line or multi-file edits; when present, path/old_text/new_text are ignored.",
                    },
                },
                "required": [],
            },
            handler=files.edit_text,
        )
    )
