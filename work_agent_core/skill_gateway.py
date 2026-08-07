from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import json

from .skill_runtime import SkillRuntime, load_skill_manifests
from .tool_bus import ToolProvider
from .tools import Tool


SKILL_TOOL_ALIASES: dict[str, set[str]] = {
    "meeting-minutes": {
        "check_meeting_asr_progress",
        "transcribe_meeting_audio",
    },
    "docx": {"process_office_document", "create_docx_from_markdown", "docx_soffice"},
    "pdf": {"process_office_document", "create_pdf_from_markdown"},
    "pptx": {"process_office_document", "create_pptx_from_outline"},
    "xlsx": {
        "process_office_document",
        "create_xlsx_from_markdown",
        "manage_timeline_xlsx",
        "manage_project_timeline",
    },
    "skill-creator": {
        "validate_work_agent_skill",
        "scaffold_work_agent_skill",
        "inspect_skill_health",
    },
    "work-reports": {
        "collect_work_report_evidence",
        "save_work_report",
        "read_saved_work_report",
        "delete_work_report",
        "check_work_report_status",
        "update_workday_calendar",
    },
    "apple-schedule": {"list_apple_schedule", "create_apple_reminder"},
    "edge-browser": {
        "browser_click", "browser_close", "browser_fill_form", "browser_find",
        "browser_hover", "browser_navigate", "browser_navigate_back", "browser_press_key",
        "browser_select_option", "browser_snapshot", "browser_tabs", "browser_type", "browser_wait_for",
    },
    "weixin-search": {
        "weixin_search", "weixin_search_all", "resolve_weixin_article_url",
        "get_weixin_article_content",
    },
}
COMMON_SKILL_TOOLS = {"run_skill_script", "precheck_skill_environment"}


class SkillGateway:
    """Friday-style progressive skill gateway.

    The model always sees one compact ``sys_skill`` tool. Skill manuals and
    concrete native tool schemas are loaded only after the model chooses a
    skill. Concrete tools stay mounted for execution but are not placed in the
    model's top-level tools array.
    """

    def __init__(
        self,
        workspace_root: str | Path,
        skill_providers: Iterable[ToolProvider],
        *,
        enabled_skill_ids: set[str] | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.runtime = SkillRuntime(self.workspace_root, enabled_skill_ids=enabled_skill_ids)
        self.skill_providers = list(skill_providers)

    def as_tool(self) -> Tool:
        return Tool(
            name="sys_skill",
            description=(
                "技能分层入口。领域任务先 open 对应技能读取说明，再用 show 查看某个技能工具的参数，"
                "最后用 call 调用该技能工具。list 只返回技能名称和简介。具体技能工具不会常驻顶层 tools。"
                "参数规则：open 必须带 skill_id；show 必须带 skill_id 和 tool_name；call 必须带 skill_id、tool_name 和 arguments。"
                "通过 call 调用 run_skill_script 时，arguments 只传 script_path、arguments、timeout_seconds，skill_id 会由网关自动注入。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "op": {
                        "type": "string",
                        "enum": ["list", "open", "show", "call"],
                    },
                    "skill_id": {"type": "string", "description": "技能 id，例如 meeting-minutes、docx、anysearch。"},
                    "tool_name": {"type": "string", "description": "show/call 时指定该技能下的工具名。"},
                    "arguments": {"type": "object", "description": "call 时传给技能工具的参数。"},
                    "max_chars": {"type": "integer", "default": 30000},
                },
                "required": ["op"],
            },
            handler=self.handle,
            metadata={"layer": "skill_gateway"},
        )

    def handle(self, args: dict[str, Any]) -> str:
        op = str(args.get("op") or "").strip().lower()
        if op == "list":
            return self._list_skills()

        skill_id = self._normalize_skill_id(args.get("skill_id"))
        if op == "open":
            return self._open_skill(skill_id, args)
        if op == "show":
            tool = self._skill_tool(skill_id, args.get("tool_name"))
            return json.dumps(self._tool_payload(skill_id, tool), ensure_ascii=False, indent=2)
        if op == "call":
            tool = self._skill_tool(skill_id, args.get("tool_name"))
            arguments = args.get("arguments")
            if arguments is None:
                arguments = {}
            if not isinstance(arguments, dict):
                raise ValueError("sys_skill.call 的 arguments 必须是对象。")
            if tool.name == "run_skill_script":
                arguments = dict(arguments)
                arguments.setdefault("skill_id", skill_id)
            missing = [
                str(name)
                for name in (tool.parameters.get("required") or [])
                if name not in arguments or arguments.get(name) in (None, "")
            ]
            if missing:
                raise ValueError(
                    f"技能工具 {skill_id}.{tool.name} 缺少必填参数：{', '.join(missing)}。"
                    f"请先调用 sys_skill(op='show', skill_id='{skill_id}', tool_name='{tool.name}') 查看参数。"
                )
            return tool.handler(normalize_skill_tool_arguments(tool, arguments))
        raise ValueError(f"不支持的 sys_skill 操作：{op or '（空）'}")

    def _list_skills(self) -> str:
        items = []
        for manifest in self._enabled_manifests():
            items.append(
                {
                    "id": manifest.id,
                    "label": manifest.label,
                    "description": manifest.description,
                    "when_to_use": manifest.when_to_use,
                    "default_enabled": manifest.default_enabled,
                }
            )
        return json.dumps({"skills": items}, ensure_ascii=False, indent=2)

    def _open_skill(self, skill_id: str, args: dict[str, Any]) -> str:
        max_chars = max(1000, min(int(args.get("max_chars") or 30000), 60000))
        payload = json.loads(
            self.runtime.read_skill_instructions(
                {"skill_id": skill_id, "max_chars": max_chars}
            )
        )
        payload["available_tools"] = [
            {"name": tool.name, "description": tool.description}
            for tool in self._tools_for_skill(skill_id)
        ]
        payload["execution_guidance"] = (
            "需要技能工具时，先用 sys_skill(op='show') 查看参数，再用 sys_skill(op='call') 执行。"
            "read_text_file、write_text_file、edit_text_file、apply_unified_patch、list_workspace_files、shell_exec "
            "属于常驻 core 工具，可直接调用。"
        )
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _skill_tool(self, skill_id: str, raw_tool_name: Any) -> Tool:
        tool_name = str(raw_tool_name or "").strip()
        if not tool_name:
            raise ValueError("sys_skill 的 show/call 操作必须提供 tool_name。")
        tools = {tool.name: tool for tool in self._tools_for_skill(skill_id)}
        tool = tools.get(tool_name)
        if tool is None:
            available = "、".join(sorted(tools)) or "无"
            raise KeyError(f"技能 {skill_id!r} 没有工具 {tool_name!r}。可用工具：{available}")
        return tool

    def _tools_for_skill(self, skill_id: str) -> list[Tool]:
        allowed_names = self._skill_tool_names().get(skill_id)
        if allowed_names is None:
            raise KeyError(f"未找到技能：{skill_id}")
        tools: dict[str, Tool] = {}
        for provider in self.skill_providers:
            for tool in provider.list_tools():
                if tool.name not in allowed_names or tool.name in tools:
                    continue
                tools[tool.name] = Tool(
                    name=tool.name,
                    description=tool.description,
                    parameters=tool.parameters,
                    handler=lambda arguments, provider=provider, name=tool.name: provider.call_tool(name, arguments),
                    provider_id=provider.provider_id,
                    provider_kind=provider.provider_kind,
                    metadata=tool.metadata,
                )
        return [tools[name] for name in sorted(tools)]

    def _skill_tool_names(self) -> dict[str, set[str]]:
        manifests = {manifest.id: manifest for manifest in self._enabled_manifests()}
        mapping: dict[str, set[str]] = {}

        def names_for(skill_id: str, visiting: set[str] | None = None) -> set[str]:
            """Resolve a skill's own tools plus explicitly declared dependencies.

            Most skills remain isolated. Composite skills can declare
            ``dependencies.skills`` when their workflow intentionally
            orchestrates another skill's tools. The recursion guard keeps a
            malformed dependency cycle harmless.
            """

            if skill_id in mapping:
                return set(mapping[skill_id])
            manifest = manifests.get(skill_id)
            if manifest is None:
                return set()
            visiting = set(visiting or ())
            if skill_id in visiting:
                return set()
            visiting.add(skill_id)

            names = {
                str(item.get("name") or "").strip()
                for item in manifest.native_tools
                if isinstance(item, dict) and str(item.get("name") or "").strip()
            }
            if manifest.tool_name:
                names.add(manifest.tool_name)
            names.update(SKILL_TOOL_ALIASES.get(manifest.id, set()))
            names.update(COMMON_SKILL_TOOLS)
            for dependency_id in manifest.skill_dependencies:
                names.update(names_for(dependency_id, visiting))
            mapping[skill_id] = names
            return set(names)

        for skill_id in manifests:
            names_for(skill_id)
        return mapping

    def _normalize_skill_id(self, raw_skill_id: Any) -> str:
        value = str(raw_skill_id or "").strip()
        aliases = {
            "meeting": "meeting-minutes",
            "meeting_minutes": "meeting-minutes",
            "会议纪要": "meeting-minutes",
            "@会议纪要": "meeting-minutes",
            "official_document": "official-document",
            "公文": "official-document",
            "@公文": "official-document",
        }
        value = aliases.get(value, value)
        if not value:
            raise ValueError("sys_skill 的 open/show/call 操作必须提供 skill_id。")
        known = {manifest.id for manifest in load_skill_manifests(self.workspace_root)}
        if value not in known:
            raise KeyError(f"未找到技能：{value}")
        if not self.runtime.is_skill_enabled(value):
            raise PermissionError(f"技能 {value!r} 当前已关闭，请先在网页“技能”页启用后再开始新对话。")
        return value

    def _enabled_manifests(self):
        return [
            manifest
            for manifest in load_skill_manifests(self.workspace_root)
            if self.runtime.is_skill_enabled(manifest.id)
        ]

    @staticmethod
    def _tool_payload(skill_id: str, tool: Tool) -> dict[str, Any]:
        return {
            "skill_id": skill_id,
            "tool": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
            "call": {
                "op": "call",
                "skill_id": skill_id,
                "tool_name": tool.name,
                "arguments": {},
            },
        }


def normalize_skill_tool_arguments(tool: Tool, arguments: dict[str, Any]) -> dict[str, Any]:
    """Accept stable cross-skill path aliases at the gateway boundary.

    Audio skills historically call their source argument ``input_path`` while
    office/document tools use ``path``.  The model can see the latter schema
    after ``show``, but a recovered or long-running turn may still emit the
    former name.  Normalizing here avoids a brittle ``KeyError`` without
    changing the concrete tool contract or accepting paths outside the
    workspace.
    """

    normalized = dict(arguments)
    if tool.name == "process_office_document" and not str(normalized.get("path") or "").strip():
        for alias in ("input_path", "audio_path", "transcript_path"):
            value = str(normalized.get(alias) or "").strip()
            if value:
                normalized["path"] = value
                break
    return normalized
