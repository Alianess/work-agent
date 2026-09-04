"""Structured artifact collection at the tool boundary.

This is a Harness adapter, not model behavior.  It observes verified tool
inputs/results/progress, resolves paths inside the account workspace and emits
typed artifact facts.  Final assistant prose is deliberately irrelevant.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from .session_runtime import ConversationRuntime


ARTIFACT_PATH_KEYS = frozenset(
    {
        "artifact",
        "artifact_path",
        "artifacts",
        "content_path",
        "docx_path",
        "file",
        "file_path",
        "files",
        "manifest_path",
        "markdown_path",
        "output",
        "output_file",
        "output_path",
        "outputs",
        "path",
        "pdf_path",
        "pptx_path",
        "report_path",
        "transcript_path",
        "work_docx_path",
        "work_markdown_path",
        "work_path",
        "xlsx_path",
    }
)
DIRECT_FILE_TOOLS = frozenset(
    {"write_text_file", "edit_text_file", "apply_unified_patch", "save_work_report"}
)
DELIVERY_EXTENSIONS = frozenset(
    {".csv", ".doc", ".docx", ".md", ".pdf", ".ppt", ".pptx", ".txt", ".xls", ".xlsx"}
)
NON_DELIVERY_ROOTS = frozenset(
    {".git", "config", "debug_traces", "office_workspace", "tests", "tmp", "web_frontend", "work_agent_core"}
)


class ToolArtifactCollector:
    def __init__(self, workspace_root: str | Path | None) -> None:
        self.workspace_root = Path(workspace_root or Path.cwd())
        self._artifacts: dict[str, dict[str, Any]] = {}
        self._recorded_ids: set[str] = set()

    def reset(self) -> None:
        self._artifacts.clear()
        self._recorded_ids.clear()

    def capture_progress(self, event: dict[str, Any], *, tool_name: str, step: int) -> None:
        candidates: list[str] = []
        raw_changes = event.get("file_changes")
        if isinstance(raw_changes, list):
            candidates.extend(
                str(item.get("file_path") or "")
                for item in raw_changes
                if isinstance(item, dict)
            )
        raw_path = event.get("file_path")
        if isinstance(raw_path, str):
            candidates.append(raw_path)
        for candidate in candidates:
            self._capture(candidate, tool_name=tool_name, step=step)

    def record_tool_result(
        self,
        runtime: ConversationRuntime,
        *,
        tool_name: str,
        tool_input: dict[str, Any],
        observation: str,
        step: int,
    ) -> None:
        for candidate in artifact_path_candidates(tool_name, tool_input, observation):
            self._capture(candidate, tool_name=tool_name, step=step)
        for artifact_id, artifact in self._artifacts.items():
            if artifact_id in self._recorded_ids:
                continue
            runtime.record_artifact(**artifact)
            self._recorded_ids.add(artifact_id)

    def _capture(self, raw_path: str, *, tool_name: str, step: int) -> None:
        artifact = normalize_artifact_path(
            raw_path,
            workspace_root=self.workspace_root,
            tool_name=tool_name,
            step=step,
        )
        if artifact is not None:
            self._artifacts.setdefault(str(artifact["artifact_id"]), artifact)


def artifact_path_candidates(
    tool_name: str,
    tool_input: dict[str, Any],
    observation: str,
) -> list[str]:
    """Extract file references at the tool boundary, never from final prose."""

    candidates: list[str] = []
    normalized_tool = str(tool_name or "").strip()
    if normalized_tool in DIRECT_FILE_TOOLS:
        for key in ("path", "output_path", "file_path"):
            value = tool_input.get(key)
            if isinstance(value, str):
                candidates.append(value)

    try:
        payload = json.loads(str(observation or ""))
    except (TypeError, ValueError, json.JSONDecodeError):
        payload = None

    def visit(value: Any, *, key: str = "") -> None:
        if isinstance(value, dict):
            for child_key, child in value.items():
                normalized_key = str(child_key or "").strip().lower()
                if normalized_key in ARTIFACT_PATH_KEYS:
                    visit(child, key=normalized_key)
                elif isinstance(child, (dict, list)):
                    visit(child, key=normalized_key)
            return
        if isinstance(value, list):
            for child in value:
                visit(child, key=key)
            return
        if key in ARTIFACT_PATH_KEYS and isinstance(value, str):
            candidates.append(value)

    if payload is not None:
        visit(payload)
    return list(dict.fromkeys(item.strip() for item in candidates if item.strip()))


def normalize_artifact_path(
    raw_path: str,
    *,
    workspace_root: Path,
    tool_name: str,
    step: int,
) -> dict[str, Any] | None:
    """Resolve and classify a tool-produced path after verifying it exists."""

    text = str(raw_path or "").strip().strip("`'\"")
    if not text or "\n" in text or text.startswith(("http://", "https://", "data:")):
        return None
    if text.startswith("file://"):
        text = text[7:]
    root = workspace_root.resolve()
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve()
        relative = resolved.relative_to(root)
    except (OSError, ValueError):
        return None
    if not resolved.is_file():
        return None
    parts = relative.parts
    if not parts or parts[0] in NON_DELIVERY_ROOTS or any(part.startswith("_") for part in parts):
        return None
    if len(parts) >= 2 and parts[0] == "meet_files" and parts[1] == "execution":
        return None
    relative_text = relative.as_posix()
    extension = resolved.suffix.lower()
    return {
        "path": relative_text,
        "artifact_id": relative_text,
        "kind": extension.lstrip(".") or "file",
        "title": resolved.name,
        "origin": "tool",
        "tool_name": str(tool_name or ""),
        "status": "created",
        "step": int(step),
        "delivery_ready": extension in DELIVERY_EXTENSIONS,
        "size_bytes": int(resolved.stat().st_size),
    }
