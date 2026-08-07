from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time

from .progress import compact_preview_text, emit_tool_progress, next_command_id, run_logged_process
from .office_preview import convert_office_to_pdf
from .runtime_env import find_runtime_executable
from .skill_manifest import (
    DEFAULT_MAX_SKILL_CHARS,
    SKILLS_DIR,
    SkillManifest,
    build_skill_health_report,
    discover_declared_tools,
    humanize_skill_id,
    is_valid_tool_name,
    load_skill_manifests as load_skill_manifests_module,
    load_single_skill_manifest,
    normalize_skill_id,
    office_python,
    parse_skill_markdown,
    probe_skill_environment,
    read_skill_config,
    skill_roots,
    summarize_declared_skill_tools,
    validate_tool_declaration,
    write_skill_health_report,
)
from .tools import Tool, ToolRegistry, WorkspaceFiles


EXTRA_SKILL_DIRS = [Path("meeting_audio_minutes/skills")]


def load_skill_manifests(workspace_root: str | Path) -> list[SkillManifest]:
    return load_skill_manifests_module(workspace_root)


def register_declared_skill_tools(registry: ToolRegistry, runtime: "SkillRuntime") -> None:
    """Mount skill-declared operations as native backends behind ``sys_skill``.

    Tools are discovered from two sources (priority order):
      1. `work_agent.json` -> `tools[]` (centralized declarations)
      2. `scripts/<name>.tool.json` sidecars (drop-in per-script manifests)

    Validation failures are surfaced via a structured health report written
    to disk, plus stderr lines, so a misconfigured skill is visible at startup
    rather than silently ignored.
    """

    registration_log: list[dict[str, Any]] = []
    for manifest in load_skill_manifests(runtime.workspace_root):
        if not runtime.is_skill_enabled(manifest.id):
            continue
        skill_dir = runtime.workspace_root / manifest.path
        declarations = discover_declared_tools(skill_dir, runtime.workspace_root)
        for raw_tool, source, script_path in declarations:
            name = str(raw_tool.get("name") or "").strip()
            ok, reason = validate_tool_declaration(raw_tool, manifest.id)
            if not ok:
                print(f"[SkillRuntime] Skip skill tool {manifest.id}.{name}: {reason}", file=sys.stderr)
                registration_log.append(
                    {
                        "skill_id": manifest.id,
                        "tool_name": name,
                        "registered": False,
                        "reason": reason,
                        "source": source,
                        "script_path": script_path,
                    }
                )
                continue
            full_script = skill_dir / script_path
            script_exists = full_script.is_file()
            if not script_exists:
                print(
                    f"[SkillRuntime] Skip skill tool {manifest.id}.{name}: 脚本不存在 {script_path}",
                    file=sys.stderr,
                )
                registration_log.append(
                    {
                        "skill_id": manifest.id,
                        "tool_name": name,
                        "registered": False,
                        "reason": f"脚本不存在：{script_path}",
                        "source": source,
                        "script_path": script_path,
                    }
                )
                continue
            try:
                registry.register(
                    Tool(
                        name=name,
                        description=str(raw_tool.get("description") or "").strip(),
                        parameters=raw_tool.get("parameters") or {"type": "object", "properties": {}},
                        handler=runtime.make_declared_skill_tool_handler(
                            skill_id=manifest.id,
                            tool_name=name,
                            execution=raw_tool.get("execution") or {},
                        ),
                    )
                )
                registration_log.append(
                    {
                        "skill_id": manifest.id,
                        "tool_name": name,
                        "registered": True,
                        "reason": "",
                        "source": source,
                        "script_path": script_path,
                    }
                )
            except ValueError as error:
                print(
                    f"[SkillRuntime] Skip duplicate skill tool {manifest.id}.{name}: {error}",
                    file=sys.stderr,
                )
                registration_log.append(
                    {
                        "skill_id": manifest.id,
                        "tool_name": name,
                        "registered": False,
                        "reason": f"duplicate: {error}",
                        "source": source,
                        "script_path": script_path,
                    }
                )
    runtime.skill_tool_log = registration_log


def register_skill_runtime_tools(
    registry: ToolRegistry,
    workspace_root: str | Path,
    *,
    enabled_skill_ids: set[str] | None = None,
) -> None:
    runtime = SkillRuntime(workspace_root, enabled_skill_ids=enabled_skill_ids)
    registry.register(
        Tool(
            name="list_available_skills",
            description=(
                "List installed work-agent skills. Use this to answer capability questions "
                "or inspect which skills are installed. Dedicated skill tools are accessed "
                "through the sys_skill gateway."
            ),
            parameters={"type": "object", "properties": {}},
            handler=runtime.list_available_skills,
        )
    )
    registry.register(
        Tool(
            name="read_skill_instructions",
            description=(
                "Read a skill's SKILL.md and list bundled resources. Use this when the user asks "
                "to inspect a skill before using its tools through the sys_skill gateway."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "skill_id": {"type": "string"},
                    "max_chars": {"type": "integer", "default": DEFAULT_MAX_SKILL_CHARS},
                },
                "required": ["skill_id"],
            },
            handler=runtime.read_skill_instructions,
        )
    )
    registry.register(
        Tool(
            name="validate_work_agent_skill",
            description="Validate a work-agent skill folder for required SKILL.md metadata and basic resource shape.",
            parameters={
                "type": "object",
                "properties": {"skill_id": {"type": "string"}},
                "required": ["skill_id"],
            },
            handler=runtime.validate_skill,
        )
    )
    registry.register(
        Tool(
            name="scaffold_work_agent_skill",
            description="Create a new work-agent skill folder with SKILL.md and work_agent.json metadata.",
            parameters={
                "type": "object",
                "properties": {
                    "skill_id": {"type": "string"},
                    "label": {"type": "string"},
                    "description": {"type": "string"},
                    "when_to_use": {"type": "string"},
                    "tool_name": {"type": "string"},
                },
                "required": ["skill_id", "label", "description"],
            },
            handler=runtime.scaffold_skill,
        )
    )
    registry.register(
        Tool(
            name="run_skill_script",
            description=(
                "Low-level fallback: run a script bundled inside an installed skill folder. "
                "Prefer a dedicated skill tool through sys_skill when the skill exposes one; use this only "
                "for skill workflows that do not have a more specific registered tool."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "skill_id": {"type": "string"},
                    "script_path": {"type": "string"},
                    "arguments": {"type": "array", "items": {"type": "string"}, "default": []},
                    "timeout_seconds": {"type": "integer", "default": 120},
                },
                "required": ["skill_id", "script_path"],
            },
            handler=runtime.run_skill_script,
        )
    )
    registry.register(
        Tool(
            name="process_office_document",
            description=(
                "Extract text, tables, slide text, or PDF text from office files and save a Markdown preview. "
                "Supports .docx, .xlsx/.xlsm/.csv/.tsv, .pptx, and .pdf."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "input_path": {
                        "type": "string",
                        "description": "兼容音频/会议技能的输入路径别名；优先使用 path。",
                    },
                    "operation": {
                        "type": "string",
                        "enum": ["extract_text", "convert_to_markdown", "summarize_structure"],
                        "default": "extract_text",
                    },
                    "output_path": {"type": "string"},
                    "output_dir": {
                        "type": "string",
                        "description": "未指定 output_path 时使用的输出目录。",
                    },
                    "max_chars": {"type": "integer", "default": 50000},
                },
                "required": ["path"],
            },
            handler=runtime.process_office_document,
        )
    )
    registry.register(
        Tool(
            name="create_docx_from_markdown",
            description=(
                "Compatibility converter that creates a fixed-layout Chinese .docx from Markdown content or a Markdown file. "
                "Use the complete docx skill for company-format documents, official documents, templates, editing, comments, "
                "tracked changes, and render QA; do not treat this converter as the whole Word capability."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "markdown_path": {"type": "string", "description": "Existing Markdown source path under the workspace."},
                    "markdown_content": {"type": "string", "description": "Markdown text to convert when no source file exists."},
                    "output_path": {"type": "string", "description": "Target .docx path under the workspace."},
                    "title": {"type": "string", "description": "Optional document title."},
                },
                "required": ["output_path"],
            },
            handler=runtime.create_docx_from_markdown,
        )
    )
    registry.register(
        Tool(
            name="docx_soffice",
            description=(
                "Structured DOCX-to-PDF conversion backed by an isolated LibreOffice profile. "
                "Pass paths only; the backend supplies headless flags, output directory, timeout handling, "
                "and verifies that a non-empty PDF was produced. Use only for an explicit PDF conversion "
                "or layout diagnosis, not for routine Web DOCX preview."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "input_path": {
                        "type": "string",
                        "description": "Existing .docx source path under the workspace.",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Optional target .pdf path; defaults beside the DOCX.",
                    },
                },
                "required": ["input_path"],
            },
            handler=runtime.convert_docx_to_pdf,
        )
    )
    registry.register(
        Tool(
            name="create_xlsx_from_markdown",
            description=(
                "Create an Excel .xlsx workbook from Markdown tables or inline sheet data. "
                "Pass markdown_content with one or more Markdown tables (each becomes a sheet, "
                "using the preceding H2/H3 heading as the sheet name, or Sheet1/Sheet2). "
                "Use this when a skill or user requests an Excel deliverable from structured data."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "markdown_path": {"type": "string", "description": "Existing Markdown source path under the workspace."},
                    "markdown_content": {"type": "string", "description": "Markdown text containing tables to convert."},
                    "output_path": {"type": "string", "description": "Target .xlsx path under the workspace."},
                    "sheet_name": {"type": "string", "description": "Override the sheet name for the first table."},
                },
                "required": ["output_path"],
            },
            handler=runtime.create_xlsx_from_markdown,
        )
    )
    registry.register(
        Tool(
            name="manage_timeline_xlsx",
            description=(
                "Inspect or safely edit time-node, milestone, schedule, and project-progress Excel workbooks. "
                "It detects common Chinese or English headers, lists normalized nodes, and batch-adds, updates, "
                "soft-deletes, or physically deletes rows while preserving untouched workbook content and styles. "
                "Writes default to dry-run; actual writes can create a backup and hidden audit log."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["inspect", "list", "apply"],
                        "default": "inspect",
                    },
                    "path": {"type": "string", "description": "Source .xlsx/.xlsm path under the workspace."},
                    "output_path": {
                        "type": "string",
                        "description": "Optional target path. Omit to update the source when dry_run=false.",
                    },
                    "sheet_name": {"type": "string"},
                    "header_row": {"type": "integer"},
                    "field_mapping": {
                        "type": "object",
                        "description": (
                            "Optional canonical-field mapping when automatic detection is insufficient. "
                            "Values may be exact header names or 1-based column numbers."
                        ),
                        "additionalProperties": {},
                    },
                    "limit": {"type": "integer", "default": 200},
                    "changes": {
                        "type": "array",
                        "description": (
                            "For apply: ordered batch changes. action is add/update/delete; "
                            "update/delete match by row, node_id, title, date, or other detected field. "
                            "delete defaults to soft status=已取消; pass delete_mode=row for physical deletion."
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "action": {
                                    "type": "string",
                                    "enum": ["add", "update", "delete"],
                                },
                                "match": {"type": "object", "additionalProperties": {}},
                                "values": {"type": "object", "additionalProperties": {}},
                                "delete_mode": {
                                    "type": "string",
                                    "enum": ["soft", "row"],
                                    "default": "soft",
                                },
                            },
                            "required": ["action"],
                        },
                    },
                    "create_missing_columns": {"type": "boolean", "default": False},
                    "dry_run": {
                        "type": "boolean",
                        "default": True,
                        "description": "Preview changes without saving. Set false only for an intended write.",
                    },
                    "backup": {"type": "boolean", "default": True},
                    "record_history": {"type": "boolean", "default": True},
                    "change_source": {"type": "string", "default": "Friday"},
                },
                "required": ["path"],
            },
            handler=runtime.manage_timeline_xlsx,
        )
    )
    registry.register(
        Tool(
            name="manage_project_timeline",
            description=(
                "Read, create, or update the selected project's timeline Excel without asking the user for a path. "
                "Use this for project milestones, planned dates, status, progress, next actions, owners, materials, "
                "and completion dates. The project Excel remains the single source of truth."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["inspect", "create", "apply"],
                        "default": "inspect",
                    },
                    "project_id": {
                        "type": "string",
                        "description": "Current project id, for example project-a1b2c3d4e5f6.",
                    },
                    "changes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "action": {
                                    "type": "string",
                                    "enum": ["add", "update", "delete"],
                                },
                                "match": {"type": "object", "additionalProperties": {}},
                                "values": {"type": "object", "additionalProperties": {}},
                                "delete_mode": {
                                    "type": "string",
                                    "enum": ["soft", "row"],
                                    "default": "soft",
                                },
                            },
                            "required": ["action"],
                        },
                    },
                    "change_source": {"type": "string", "default": "Friday"},
                },
                "required": ["project_id"],
            },
            handler=runtime.manage_project_timeline,
        )
    )
    registry.register(
        Tool(
            name="create_pptx_from_outline",
            description=(
                "Create a PowerPoint .pptx deck from a Markdown outline. Each H2 becomes a slide; "
                "the heading text is the title and the body (bullets/paragraphs) becomes slide content. "
                "Use this when a skill or user requests a presentation deliverable."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "markdown_path": {"type": "string", "description": "Existing Markdown outline path under the workspace."},
                    "markdown_content": {"type": "string", "description": "Markdown outline text to convert."},
                    "output_path": {"type": "string", "description": "Target .pptx path under the workspace."},
                    "title": {"type": "string", "description": "Optional title-slide title."},
                    "subtitle": {"type": "string", "description": "Optional title-slide subtitle."},
                },
                "required": ["output_path"],
            },
            handler=runtime.create_pptx_from_outline,
        )
    )
    registry.register(
        Tool(
            name="create_pdf_from_markdown",
            description=(
                "Create a PDF document from Markdown content. Uses reportlab under the hood. "
                "Use this when a skill or user requests a PDF deliverable."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "markdown_path": {"type": "string", "description": "Existing Markdown source path under the workspace."},
                    "markdown_content": {"type": "string", "description": "Markdown text to convert."},
                    "output_path": {"type": "string", "description": "Target .pdf path under the workspace."},
                    "title": {"type": "string", "description": "Optional document title shown on the first page."},
                },
                "required": ["output_path"],
            },
            handler=runtime.create_pdf_from_markdown,
        )
    )
    registry.register(
        Tool(
            name="precheck_skill_environment",
            description=(
                "Probe declared dependencies (Python modules, binaries, node modules) for installed skills. "
                "Call this before running an office/skill workflow so you can adaptively pick an execution "
                "path (e.g. skip LibreOffice-dependent recalculation if soffice is unavailable). "
                "Pass skill_id to probe a single skill, or omit to probe all skills."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "skill_id": {"type": "string", "description": "Optional: probe a single skill by id."},
                },
            },
            handler=runtime.precheck_skill_environment,
        )
    )
    registry.register(
        Tool(
            name="inspect_skill_health",
            description=(
                "Report which skill-declared native tools registered successfully and which were skipped "
                "(missing script, invalid schema, duplicate name). Use this to debug why a skill tool is "
                "not available in the tool list."
            ),
            parameters={"type": "object", "properties": {}},
            handler=runtime.inspect_skill_health,
        )
    )
    register_declared_skill_tools(registry, runtime)
    write_skill_health_report(runtime.workspace_root)


class SkillRuntime:
    def __init__(self, workspace_root: str | Path, *, enabled_skill_ids: set[str] | None = None) -> None:
        self.workspace = WorkspaceFiles(workspace_root)
        self.workspace_root = self.workspace.workspace_root
        self.skill_tool_log: list[dict[str, Any]] = []
        self.enabled_skill_ids = set(enabled_skill_ids) if enabled_skill_ids is not None else None

    def is_skill_enabled(self, skill_id: str) -> bool:
        return self.enabled_skill_ids is None or skill_id in self.enabled_skill_ids

    def require_enabled_skill(self, skill_id: str) -> None:
        if not self.is_skill_enabled(skill_id):
            raise PermissionError(f"技能 {skill_id!r} 当前已关闭，请先在技能目录中启用。")

    def list_available_skills(self, _args: dict[str, Any]) -> str:
        manifests = [
            manifest.to_payload()
            for manifest in load_skill_manifests(self.workspace_root)
            if self.is_skill_enabled(manifest.id)
        ]
        return json.dumps({"skills": manifests}, ensure_ascii=False, indent=2)

    def read_skill_instructions(self, args: dict[str, Any]) -> str:
        self.require_enabled_skill(str(args["skill_id"]))
        skill_dir = self.resolve_skill_dir(str(args["skill_id"]))
        skill_md = skill_dir / "SKILL.md"
        max_chars = int(args.get("max_chars") or DEFAULT_MAX_SKILL_CHARS)
        text = skill_md.read_text(encoding="utf-8", errors="replace")
        truncated = len(text) > max_chars
        resources = list_skill_resources(skill_dir, self.workspace_root)
        runtime_conf = read_optional_text(skill_dir / "runtime.conf", max_chars=6000)
        work_agent_config = read_optional_text(skill_dir / "work_agent.json", max_chars=6000)
        config = read_skill_config(skill_dir)
        return json.dumps(
            {
                "skill_id": skill_dir.name,
                "path": str(skill_dir.relative_to(self.workspace_root)),
                "instructions": text[:max_chars],
                "truncated": truncated,
                "resources": resources,
                "native_tools": summarize_declared_skill_tools(config.get("tools")),
                "script_entrypoints": list_skill_scripts(skill_dir, self.workspace_root),
                "runtime_conf": runtime_conf,
                "work_agent_config": work_agent_config,
                "execution_guidance": (
                    "如果 native_tools 列出了专用工具，应通过 sys_skill.show/call 分层使用。"
                    "run_skill_script 只是低层 fallback：当没有专用工具但 SKILL.md/runtime.conf/README 给出技能目录内脚本时，"
                    "才把脚本路径放入 script_path、把子命令和参数拆成 arguments 数组；"
                    "只有命令不属于技能目录脚本、需要复杂系统命令或 run_skill_script 无法表达时，才使用 shell_exec。"
                    "不要为了某个 skill 临时编造项目专用命令，也不要把工具调用标签或 CLI 命令写进用户正文。"
                    "修改文件时，小改优先用 edit_text_file 或 apply_unified_patch，只有创建完整新文件或重写成品时才用 write_text_file。"
                ),
                "run_skill_script_guidance": build_run_skill_script_guidance(
                    skill_dir=skill_dir,
                    workspace_root=self.workspace_root,
                    runtime_conf=runtime_conf,
                ),
            },
            ensure_ascii=False,
            indent=2,
        )

    def validate_skill(self, args: dict[str, Any]) -> str:
        self.require_enabled_skill(str(args["skill_id"]))
        skill_dir = self.resolve_skill_dir(str(args["skill_id"]))
        issues: list[str] = []
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.is_file():
            issues.append("缺少 SKILL.md")
        else:
            frontmatter, body = parse_skill_markdown(skill_md.read_text(encoding="utf-8", errors="replace"))
            if not frontmatter.get("name"):
                issues.append("SKILL.md frontmatter 缺少 name")
            if not frontmatter.get("description"):
                issues.append("SKILL.md frontmatter 缺少 description")
            if not body.strip():
                issues.append("SKILL.md 正文为空")
        config_path = skill_dir / "work_agent.json"
        if config_path.is_file():
            try:
                json.loads(config_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                issues.append(f"work_agent.json 不是合法 JSON：{error}")
        return json.dumps(
            {
                "skill_id": skill_dir.name,
                "ok": not issues,
                "issues": issues,
                "resources": list_skill_resources(skill_dir, self.workspace_root),
            },
            ensure_ascii=False,
            indent=2,
        )

    def scaffold_skill(self, args: dict[str, Any]) -> str:
        skill_id = normalize_skill_id(str(args["skill_id"]))
        if not skill_id:
            raise ValueError("skill_id 必须包含小写字母、数字或连字符。")
        skills_dir = self.workspace_root / SKILLS_DIR
        skill_dir = skills_dir / skill_id
        if skill_dir.exists():
            raise FileExistsError(f"技能已存在：{skill_id}")
        label = str(args["label"]).strip()
        description = str(args["description"]).strip()
        when_to_use = str(args.get("when_to_use") or description).strip()
        tool_name = str(args.get("tool_name") or "").strip()
        skill_dir.mkdir(parents=True, exist_ok=False)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {skill_id}\ndescription: {description}\n---\n\n"
            f"# {label}\n\n"
            "## Workflow\n\n"
            "- Read this skill when the user request matches the description.\n"
            "- Prefer deterministic scripts in `scripts/` when the task is repetitive or file-sensitive.\n"
            "- Validate outputs before returning them to the user.\n",
            encoding="utf-8",
        )
        config: dict[str, Any] = {
            "id": skill_id,
            "label": label,
            "mention": f"@{label}",
            "description": description,
            "when_to_use": when_to_use,
        }
        if tool_name:
            config["tool_name"] = tool_name
        (skill_dir / "work_agent.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return json.dumps(
            {"skill_id": skill_id, "path": str(skill_dir.relative_to(self.workspace_root))},
            ensure_ascii=False,
            indent=2,
        )

    def run_skill_script(self, args: dict[str, Any]) -> str:
        self.require_enabled_skill(str(args["skill_id"]))
        skill_dir = self.resolve_skill_dir(str(args["skill_id"]))
        script_path = (skill_dir / str(args["script_path"])).resolve()
        try:
            script_path.relative_to(skill_dir)
        except ValueError as error:
            raise ValueError("script_path 必须位于对应技能目录内。") from error
        if not script_path.is_file():
            raise FileNotFoundError(f"技能脚本不存在：{script_path}")
        arguments = [str(item) for item in args.get("arguments") or []]
        timeout_seconds = int(args.get("timeout_seconds") or 120)
        command = [str(script_path), *arguments]
        if script_path.suffix == ".py":
            command = [str(office_python()), str(script_path), *arguments]
        elif script_path.suffix == ".js":
            node = find_runtime_executable("node", self.workspace_root)
            if not node:
                raise FileNotFoundError("Node.js 未安装，无法运行该技能的 JavaScript 脚本。")
            command = [node, str(script_path), *arguments]
        elif script_path.suffix == ".sh":
            command = ["bash", str(script_path), *arguments]
        result = run_logged_process(
            command,
            cwd=self.workspace_root,
            timeout_seconds=timeout_seconds,
            label="技能脚本",
            check=False,
        )
        return json.dumps(
            {
                "script": str(script_path.relative_to(self.workspace_root)),
                "returncode": result.returncode,
                "stdout": truncate(result.stdout, 12000),
                "stderr": truncate(result.stderr, 12000),
            },
            ensure_ascii=False,
            indent=2,
        )

    def make_declared_skill_tool_handler(
        self,
        *,
        skill_id: str,
        tool_name: str,
        execution: dict[str, Any],
    ):
        def handler(args: dict[str, Any]) -> str:
            return self.run_declared_skill_tool(
                skill_id=skill_id,
                tool_name=tool_name,
                execution=execution,
                args=args,
            )

        return handler

    def run_declared_skill_tool(
        self,
        *,
        skill_id: str,
        tool_name: str,
        execution: dict[str, Any],
        args: dict[str, Any],
    ) -> str:
        execution_type = str(execution.get("type") or "script").strip().lower()
        if execution_type != "script":
            raise ValueError(f"Unsupported skill tool execution type for {tool_name}: {execution_type}")
        script_path = str(execution.get("script_path") or "").strip()
        if not script_path:
            raise ValueError(f"Skill tool {tool_name} missing execution.script_path")
        rendered_arguments = render_skill_tool_arguments(execution.get("arguments") or [], args)
        timeout_seconds = int(execution.get("timeout_seconds") or args.get("timeout_seconds") or 120)
        result = self.run_skill_script(
            {
                "skill_id": skill_id,
                "script_path": script_path,
                "arguments": rendered_arguments,
                "timeout_seconds": timeout_seconds,
            }
        )
        return json.dumps(
            {
                "skill_id": skill_id,
                "tool": tool_name,
                "execution": "script",
                "arguments": rendered_arguments,
                "result": json.loads(result),
            },
            ensure_ascii=False,
            indent=2,
        )

    def process_office_document(self, args: dict[str, Any]) -> str:
        source_path = self.workspace.resolve(str(args["path"]))
        if not source_path.is_file():
            raise FileNotFoundError(f"文件不存在：{args['path']}")
        output_path = str(args.get("output_path") or "").strip()
        if output_path:
            output_file = self.workspace.resolve(output_path)
        else:
            output_dir = self.workspace.resolve(
                str(args.get("output_dir") or "meet_files/office_extracts")
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_file = output_dir / f"{timestamp}-{sanitize_filename(source_path.stem)}.md"
        script = Path(__file__).with_name("office_processor.py")
        command = [
            str(office_python()),
            str(script),
            str(source_path),
            "--operation",
            str(args.get("operation") or "extract_text"),
            "--output-path",
            str(output_file),
            "--max-chars",
            str(int(args.get("max_chars") or 50000)),
        ]
        result = run_logged_process(
            command,
            cwd=self.workspace_root,
            timeout_seconds=300,
            label="办公文件处理",
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"办公文件处理失败：{truncate(result.stderr or result.stdout, 2000)}")
        return result.stdout.strip()

    def create_docx_from_markdown(self, args: dict[str, Any]) -> str:
        output_path = self.workspace.resolve(str(args["output_path"]))
        if output_path.suffix.lower() != ".docx":
            raise ValueError("output_path 必须以 .docx 结尾。")
        markdown_path = str(args.get("markdown_path") or "").strip()
        markdown_content = str(args.get("markdown_content") or "")
        source_label = "markdown_content 参数"
        source_preview = markdown_content
        temp_source_path: Path | None = None
        if markdown_path:
            source_path = self.workspace.resolve(markdown_path)
            if not source_path.is_file():
                raise FileNotFoundError(f"Markdown 源文件不存在：{markdown_path}")
            source_label = str(source_path.relative_to(self.workspace_root))
            source_preview = source_path.read_text(encoding="utf-8", errors="replace")
        elif not markdown_content.strip():
            raise ValueError("需要提供 markdown_path 或 markdown_content。")
        else:
            temp_dir = self.workspace_root / "meet_files" / ".work_agent_tmp"
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_source_path = temp_dir / f"docx_source_{int(time.time() * 1000)}.md"
            temp_source_path.write_text(markdown_content, encoding="utf-8")
            source_path = temp_source_path
            markdown_path = str(temp_source_path.relative_to(self.workspace_root))
            source_label = str(temp_source_path.relative_to(self.workspace_root))
        try:
            output_label = str(output_path.relative_to(self.workspace_root))
        except ValueError:
            output_label = str(output_path)
        preview_id = next_command_id()
        emit_tool_progress(
            {
                "event": "activity",
                "id": preview_id,
                "phase": "action",
                "title": "生成DOCX",
                "detail": output_label,
                "content": (
                    f"准备根据 {source_label} 生成 {output_label}。\n\n"
                    "```markdown\n"
                    f"{compact_preview_text(source_preview)}\n"
                    "```"
                ),
                "activity_type": "command",
                "command": (
                    f"create_docx_from_markdown --output-path {output_label} "
                    f"--markdown-source {source_label} --chars {len(source_preview)}"
                ),
                "command_status": "running",
                "tool_name": "create_docx_from_markdown",
            }
        )
        script = Path(__file__).with_name("docx_exporter.py")
        command = [
            str(office_python()),
            str(script),
            "--output-path",
            str(output_path),
        ]
        if markdown_path:
            command.extend(["--markdown-path", str(source_path)])
        else:
            command.extend(["--markdown-content", markdown_content])
        title = str(args.get("title") or "").strip()
        if title:
            command.extend(["--title", title])
        result = run_logged_process(
            command,
            cwd=self.workspace_root,
            timeout_seconds=300,
            label="DOCX生成",
            check=False,
        )
        if result.returncode != 0:
            emit_tool_progress(
                {
                    "event": "activity_delta",
                    "id": preview_id,
                    "phase": "error",
                    "title": "生成DOCX",
                    "content": f"\n✗ DOCX 生成失败：{truncate(result.stderr or result.stdout, 1000)}\n",
                    "activity_type": "command",
                    "command_status": "error",
                    "tool_name": "create_docx_from_markdown",
                }
            )
            raise RuntimeError(f"DOCX 生成失败：{truncate(result.stderr or result.stdout, 3000)}")
        emit_tool_progress(
            {
                "event": "activity_delta",
                "id": preview_id,
                "phase": "action",
                "title": "生成DOCX",
                "content": f"\n✓ 已生成 {output_label}。\n",
                "activity_type": "command",
                "command_status": "success",
                "tool_name": "create_docx_from_markdown",
            }
        )
        return result.stdout.strip()

    def create_xlsx_from_markdown(self, args: dict[str, Any]) -> str:
        output_path = self.workspace.resolve(str(args["output_path"]))
        if output_path.suffix.lower() != ".xlsx":
            raise ValueError("output_path 必须以 .xlsx 结尾。")
        markdown_path, markdown_content, source_label = _resolve_markdown_source(
            self, args, tool_name="create_xlsx_from_markdown"
        )
        script = Path(__file__).with_name("xlsx_exporter.py")
        command = [
            str(office_python()),
            str(script),
            "--output-path",
            str(output_path),
        ]
        if markdown_path:
            command.extend(["--markdown-path", str(markdown_path)])
        else:
            command.extend(["--markdown-content", markdown_content])
        sheet_name = str(args.get("sheet_name") or "").strip()
        if sheet_name:
            command.extend(["--sheet-name", sheet_name])
        try:
            output_label = str(output_path.relative_to(self.workspace_root))
        except ValueError:
            output_label = str(output_path)
        preview_id = next_command_id()
        emit_tool_progress(
            {
                "event": "activity",
                "id": preview_id,
                "phase": "action",
                "title": "生成XLSX",
                "detail": f"从 {source_label} 生成 {output_label}",
                "content": "",
                "activity_type": "command",
                "command": f"create_xlsx_from_markdown --output-path {output_label} --source {source_label}",
                "command_status": "running",
                "tool_name": "create_xlsx_from_markdown",
            }
        )
        result = run_logged_process(
            command,
            cwd=self.workspace_root,
            timeout_seconds=300,
            label="XLSX生成",
            check=False,
        )
        if result.returncode != 0:
            emit_tool_progress(
                {
                    "event": "activity_delta",
                    "id": preview_id,
                    "phase": "error",
                    "title": "生成XLSX",
                    "content": f"\n✗ XLSX 生成失败：{truncate(result.stderr or result.stdout, 1000)}\n",
                    "activity_type": "command",
                    "command_status": "error",
                    "tool_name": "create_xlsx_from_markdown",
                }
            )
            raise RuntimeError(f"XLSX 生成失败：{truncate(result.stderr or result.stdout, 3000)}")
        emit_tool_progress(
            {
                "event": "activity_delta",
                "id": preview_id,
                "phase": "action",
                "title": "生成XLSX",
                "content": f"\n✓ 已生成 {output_label}。\n",
                "activity_type": "command",
                "command_status": "success",
                "tool_name": "create_xlsx_from_markdown",
            }
        )
        return result.stdout.strip()

    def manage_timeline_xlsx(self, args: dict[str, Any]) -> str:
        from .timeline_xlsx import manage_timeline_xlsx

        payload = dict(args)
        payload["path"] = str(self.workspace.resolve(str(args["path"])))
        raw_output = str(args.get("output_path") or "").strip()
        if raw_output:
            payload["output_path"] = str(self.workspace.resolve(raw_output))
        result = manage_timeline_xlsx(payload)
        return json.dumps(result, ensure_ascii=False, indent=2, default=str)

    def manage_project_timeline(self, args: dict[str, Any]) -> str:
        from .project_timeline import (
            PROJECT_ID_PATTERN,
            apply_project_timeline_changes,
            create_project_timeline,
            project_timeline_payload,
        )

        project_id = str(args.get("project_id") or "").strip()
        if not PROJECT_ID_PATTERN.fullmatch(project_id):
            raise ValueError("project_id 格式无效。")
        project_directory = self.workspace.resolve(f"meet_files/projects/{project_id}")
        manifest_path = project_directory / "project.json"
        if not manifest_path.is_file():
            raise ValueError("项目不存在或当前工作区无权访问。")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        operation = str(args.get("operation") or "inspect").strip().lower()
        if operation == "create":
            create_project_timeline(
                project_directory,
                project_name=str(manifest.get("name") or "项目"),
            )
        elif operation == "apply":
            changes = args.get("changes")
            if not isinstance(changes, list) or not changes:
                raise ValueError("apply 操作必须提供非空 changes 数组。")
            apply_project_timeline_changes(
                project_directory,
                changes=changes,
                change_source=str(args.get("change_source") or "Friday"),
            )
        elif operation != "inspect":
            raise ValueError("operation 仅支持 inspect、create 或 apply。")
        result = project_timeline_payload(project_directory)
        if result.get("path"):
            result["path"] = str(Path(result["path"]).relative_to(self.workspace_root))
        return json.dumps(result, ensure_ascii=False, indent=2, default=str)

    def create_pptx_from_outline(self, args: dict[str, Any]) -> str:
        output_path = self.workspace.resolve(str(args["output_path"]))
        if output_path.suffix.lower() != ".pptx":
            raise ValueError("output_path 必须以 .pptx 结尾。")
        markdown_path, markdown_content, source_label = _resolve_markdown_source(
            self, args, tool_name="create_pptx_from_outline"
        )
        script = Path(__file__).with_name("pptx_exporter.py")
        command = [
            str(office_python()),
            str(script),
            "--output-path",
            str(output_path),
        ]
        if markdown_path:
            command.extend(["--markdown-path", str(markdown_path)])
        else:
            command.extend(["--markdown-content", markdown_content])
        title = str(args.get("title") or "").strip()
        subtitle = str(args.get("subtitle") or "").strip()
        if title:
            command.extend(["--title", title])
        if subtitle:
            command.extend(["--subtitle", subtitle])
        try:
            output_label = str(output_path.relative_to(self.workspace_root))
        except ValueError:
            output_label = str(output_path)
        preview_id = next_command_id()
        emit_tool_progress(
            {
                "event": "activity",
                "id": preview_id,
                "phase": "action",
                "title": "生成PPTX",
                "detail": f"从 {source_label} 生成 {output_label}",
                "content": "",
                "activity_type": "command",
                "command": f"create_pptx_from_outline --output-path {output_label} --source {source_label}",
                "command_status": "running",
                "tool_name": "create_pptx_from_outline",
            }
        )
        result = run_logged_process(
            command,
            cwd=self.workspace_root,
            timeout_seconds=300,
            label="PPTX生成",
            check=False,
        )
        if result.returncode != 0:
            emit_tool_progress(
                {
                    "event": "activity_delta",
                    "id": preview_id,
                    "phase": "error",
                    "title": "生成PPTX",
                    "content": f"\n✗ PPTX 生成失败：{truncate(result.stderr or result.stdout, 1000)}\n",
                    "activity_type": "command",
                    "command_status": "error",
                    "tool_name": "create_pptx_from_outline",
                }
            )
            raise RuntimeError(f"PPTX 生成失败：{truncate(result.stderr or result.stdout, 3000)}")
        emit_tool_progress(
            {
                "event": "activity_delta",
                "id": preview_id,
                "phase": "action",
                "title": "生成PPTX",
                "content": f"\n✓ 已生成 {output_label}。\n",
                "activity_type": "command",
                "command_status": "success",
                "tool_name": "create_pptx_from_outline",
            }
        )
        return result.stdout.strip()

    def create_pdf_from_markdown(self, args: dict[str, Any]) -> str:
        output_path = self.workspace.resolve(str(args["output_path"]))
        if output_path.suffix.lower() != ".pdf":
            raise ValueError("output_path 必须以 .pdf 结尾。")
        markdown_path, markdown_content, source_label = _resolve_markdown_source(
            self, args, tool_name="create_pdf_from_markdown"
        )
        script = Path(__file__).with_name("pdf_exporter.py")
        command = [
            str(office_python()),
            str(script),
            "--output-path",
            str(output_path),
        ]
        if markdown_path:
            command.extend(["--markdown-path", str(markdown_path)])
        else:
            command.extend(["--markdown-content", markdown_content])
        title = str(args.get("title") or "").strip()
        if title:
            command.extend(["--title", title])
        try:
            output_label = str(output_path.relative_to(self.workspace_root))
        except ValueError:
            output_label = str(output_path)
        preview_id = next_command_id()
        emit_tool_progress(
            {
                "event": "activity",
                "id": preview_id,
                "phase": "action",
                "title": "生成PDF",
                "detail": f"从 {source_label} 生成 {output_label}",
                "content": "",
                "activity_type": "command",
                "command": f"create_pdf_from_markdown --output-path {output_label} --source {source_label}",
                "command_status": "running",
                "tool_name": "create_pdf_from_markdown",
            }
        )
        result = run_logged_process(
            command,
            cwd=self.workspace_root,
            timeout_seconds=300,
            label="PDF生成",
            check=False,
        )
        if result.returncode != 0:
            emit_tool_progress(
                {
                    "event": "activity_delta",
                    "id": preview_id,
                    "phase": "error",
                    "title": "生成PDF",
                    "content": f"\n✗ PDF 生成失败：{truncate(result.stderr or result.stdout, 1000)}\n",
                    "activity_type": "command",
                    "command_status": "error",
                    "tool_name": "create_pdf_from_markdown",
                }
            )
            raise RuntimeError(f"PDF 生成失败：{truncate(result.stderr or result.stdout, 3000)}")
        emit_tool_progress(
            {
                "event": "activity_delta",
                "id": preview_id,
                "phase": "action",
                "title": "生成PDF",
                "content": f"\n✓ 已生成 {output_label}。\n",
                "activity_type": "command",
                "command_status": "success",
                "tool_name": "create_pdf_from_markdown",
            }
        )
        return result.stdout.strip()

    def precheck_skill_environment(self, args: dict[str, Any]) -> str:
        skill_id = str(args.get("skill_id") or "").strip() or None
        return json.dumps(
            probe_skill_environment(self.workspace_root, skill_id=skill_id),
            ensure_ascii=False,
            indent=2,
        )

    def convert_docx_to_pdf(self, args: dict[str, Any]) -> str:
        source_path = self.workspace.resolve(str(args.get("input_path") or ""))
        if source_path.suffix.lower() != ".docx":
            raise ValueError("input_path 必须是 .docx 文件。")
        if not source_path.is_file():
            raise FileNotFoundError(f"DOCX 文件不存在：{args.get('input_path')}")
        raw_output = str(args.get("output_path") or "").strip()
        output_path = (
            self.workspace.resolve(raw_output)
            if raw_output
            else source_path.with_suffix(".pdf")
        )
        if output_path.suffix.lower() != ".pdf":
            raise ValueError("output_path 必须以 .pdf 结尾。")
        generated = convert_office_to_pdf(
            source_path,
            output_dir=output_path.parent,
            workspace_root=self.workspace_root,
        )
        if generated.resolve() != output_path.resolve():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            generated.replace(output_path)
        if not output_path.is_file() or output_path.stat().st_size <= 0:
            raise RuntimeError("LibreOffice 返回成功，但没有生成有效 PDF。")
        try:
            output_label = str(output_path.relative_to(self.workspace_root))
            source_label = str(source_path.relative_to(self.workspace_root))
        except ValueError:
            output_label = str(output_path)
            source_label = str(source_path)
        return json.dumps(
            {
                "ok": True,
                "source_path": source_label,
                "output_path": output_label,
                "verified": True,
                "size_bytes": output_path.stat().st_size,
            },
            ensure_ascii=False,
            indent=2,
        )

    def inspect_skill_health(self, _args: dict[str, Any]) -> str:
        return json.dumps(
            {
                "registration_log": self.skill_tool_log,
                "health_report": build_skill_health_report(self.workspace_root),
            },
            ensure_ascii=False,
            indent=2,
        )

    def resolve_skill_dir(self, skill_id: str) -> Path:
        normalized = normalize_skill_id(skill_id)
        if not normalized:
            raise ValueError("缺少 skill_id")
        for skills_root in skill_roots(self.workspace_root):
            skill_dir = (skills_root / normalized).resolve()
            try:
                skill_dir.relative_to(skills_root.resolve())
            except ValueError:
                continue
            if skill_dir.is_dir():
                return skill_dir
        available = ", ".join(manifest.id for manifest in load_skill_manifests(self.workspace_root))
        raise FileNotFoundError(f"未知技能：{skill_id}。可用技能：{available}")


def list_skill_resources(skill_dir: Path, workspace_root: Path) -> list[str]:
    resources: list[str] = []
    for item in sorted(skill_dir.rglob("*")):
        if item.is_file() and item.name != "SKILL.md":
            resources.append(str(item.relative_to(workspace_root)))
    return resources[:200]


def list_skill_scripts(skill_dir: Path, workspace_root: Path) -> list[str]:
    scripts_dir = skill_dir / "scripts"
    if not scripts_dir.is_dir():
        return []
    entrypoints: list[str] = []
    for item in sorted(scripts_dir.rglob("*")):
        if item.is_file() and item.suffix.lower() in {".py", ".js", ".sh", ".ps1"}:
            entrypoints.append(str(item.relative_to(workspace_root)))
    return entrypoints[:80]


def list_skill_script_relative_paths(skill_dir: Path) -> list[str]:
    scripts_dir = skill_dir / "scripts"
    if not scripts_dir.is_dir():
        return []
    entrypoints: list[str] = []
    for item in sorted(scripts_dir.rglob("*")):
        if item.is_file() and item.suffix.lower() in {".py", ".js", ".sh", ".ps1"}:
            entrypoints.append(str(item.relative_to(skill_dir)))
    return entrypoints[:80]


def read_optional_text(path: Path, *, max_chars: int) -> str | None:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) > max_chars:
        return text[:max_chars].rstrip() + "\n...[truncated]"
    return text


def build_run_skill_script_guidance(
    *,
    skill_dir: Path,
    workspace_root: Path,
    runtime_conf: str | None,
) -> dict[str, Any]:
    skill_id = skill_dir.name
    script_paths = list_skill_script_relative_paths(skill_dir)
    guidance: dict[str, Any] = {
        "preferred_tool": "run_skill_script",
        "skill_id": skill_id,
        "rule": (
            "当 SKILL.md 或 runtime.conf 中的命令调用技能目录内脚本时，"
            "把脚本路径放入 script_path，把子命令和参数拆成 arguments 数组；"
            "不要把整条命令塞给 shell_exec。"
        ),
        "available_script_paths": script_paths,
    }
    command = parse_runtime_command(runtime_conf or "")
    if command:
        script_path, prefix_args = match_skill_script_from_command(command, skill_dir, workspace_root)
        if script_path:
            guidance["runtime_command_script_path"] = script_path
            guidance["runtime_command_prefix_arguments"] = prefix_args
            guidance["example"] = {
                "tool": "run_skill_script",
                "input": {
                    "skill_id": skill_id,
                    "script_path": script_path,
                    "arguments": [*prefix_args, "<subcommand>", "<arg1>", "<arg2>"],
                    "timeout_seconds": 120,
                },
            }
    return guidance


def parse_runtime_command(runtime_conf: str) -> list[str]:
    for line in str(runtime_conf or "").splitlines():
        if not line.lower().strip().startswith("command:"):
            continue
        raw = line.split(":", 1)[1].strip()
        if not raw:
            return []
        try:
            return shlex.split(raw)
        except ValueError:
            return raw.split()
    return []


def match_skill_script_from_command(
    command: list[str],
    skill_dir: Path,
    workspace_root: Path,
) -> tuple[str | None, list[str]]:
    for index, token in enumerate(command):
        candidate = Path(token)
        if not candidate.is_absolute():
            candidate = workspace_root / candidate
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        try:
            relative = resolved.relative_to(skill_dir)
        except ValueError:
            continue
        if resolved.is_file():
            return str(relative), command[index + 1 :]
    return None, []


def render_skill_tool_arguments(argument_specs: Any, tool_args: dict[str, Any]) -> list[str]:
    if not isinstance(argument_specs, list):
        raise ValueError("execution.arguments must be a list")
    rendered: list[str] = []
    for spec in argument_specs:
        rendered.extend(render_skill_tool_argument(spec, tool_args))
    return rendered


def render_skill_tool_argument(spec: Any, tool_args: dict[str, Any]) -> list[str]:
    if isinstance(spec, str):
        return [interpolate_argument_template(spec, tool_args)]
    if not isinstance(spec, dict):
        raise ValueError(f"Unsupported argument spec: {spec!r}")

    condition = spec.get("if")
    if condition is not None:
        if is_blank_argument_value(get_argument_value(tool_args, str(condition))):
            return []
        then_specs = spec.get("then") or []
        if not isinstance(then_specs, list):
            then_specs = [then_specs]
        return render_skill_tool_arguments(then_specs, tool_args)

    if "literal" in spec:
        return [str(spec.get("literal") or "")]

    param = str(spec.get("param") or spec.get("value") or "").strip()
    if not param:
        raise ValueError(f"Argument spec missing param: {spec!r}")

    value = get_argument_value(tool_args, param)
    if is_blank_argument_value(value) and "default" in spec:
        value = spec.get("default")

    if is_blank_argument_value(value):
        if bool(spec.get("optional")):
            return []
        raise ValueError(f"Missing required tool argument: {param}")

    parts: list[str] = []
    flag = str(spec.get("flag") or "").strip()
    if isinstance(value, bool) and flag:
        return [flag] if value else []
    if flag:
        parts.append(flag)
    parts.append(serialize_argument_value(value, spec))
    return parts


def get_argument_value(args: dict[str, Any], name: str) -> Any:
    current: Any = args
    for part in name.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def is_blank_argument_value(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def serialize_argument_value(value: Any, spec: dict[str, Any]) -> str:
    if isinstance(value, list) and spec.get("join"):
        return str(spec["join"]).join(str(item) for item in value)
    if spec.get("json") or isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def interpolate_argument_template(template: str, args: dict[str, Any]) -> str:
    whole_match = re.fullmatch(r"\{\{\s*([A-Za-z0-9_.-]+)\s*\}\}", template)
    if whole_match:
        value = get_argument_value(args, whole_match.group(1))
        if is_blank_argument_value(value):
            return ""
        return serialize_argument_value(value, {"json": isinstance(value, (dict, list))})

    def replace(match: re.Match[str]) -> str:
        value = get_argument_value(args, match.group(1))
        if is_blank_argument_value(value):
            return ""
        return serialize_argument_value(value, {"json": False})

    return re.sub(r"\{\{\s*([A-Za-z0-9_.-]+)\s*\}\}", replace, template)



def _resolve_markdown_source(
    runtime: "SkillRuntime",
    args: dict[str, Any],
    *,
    tool_name: str,
) -> tuple[Path | None, str, str]:
    """Resolve markdown_path / markdown_content into (source_path, content, label).

    If markdown_path is provided and exists, returns (path, "", relative_label).
    If markdown_content is provided, writes it to a temp file and returns
    (temp_path, content, temp_label).
    Raises ValueError if neither is provided.
    """
    markdown_path = str(args.get("markdown_path") or "").strip()
    markdown_content = str(args.get("markdown_content") or "")
    if markdown_path:
        source_path = runtime.workspace.resolve(markdown_path)
        if not source_path.is_file():
            raise FileNotFoundError(f"Markdown 源文件不存在：{markdown_path}")
        try:
            label = str(source_path.relative_to(runtime.workspace_root))
        except ValueError:
            label = str(source_path)
        return source_path, "", label
    if not markdown_content.strip():
        raise ValueError(f"{tool_name}: 需要提供 markdown_path 或 markdown_content。")
    temp_dir = runtime.workspace_root / "meet_files" / ".work_agent_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_source_path = temp_dir / f"{tool_name}_source_{int(time.time() * 1000)}.md"
    temp_source_path.write_text(markdown_content, encoding="utf-8")
    label = str(temp_source_path.relative_to(runtime.workspace_root))
    return temp_source_path, markdown_content, label


def sanitize_filename(value: str) -> str:
    cleaned = re.sub(r"[^\w\u4e00-\u9fff.-]+", "_", str(value).strip())
    return cleaned.strip("._") or "document"


def truncate(value: str, max_chars: int) -> str:
    text = str(value or "")
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...[truncated]"
