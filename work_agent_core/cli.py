from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Callable
import json
import os
import sys

from .config import DEFAULT_CONFIG_PATH, ModelProfile, ModelRegistry
from .llm import OpenAICompatibleClient
from .mcp_provider import build_mcp_tool_provider
from .mcp_gateway import MCPGateway
from .react import DEFAULT_MAX_STEPS, ReActAgent
from .shell_tools import register_shell_tools
from .skills.meeting_minutes import MeetingMinutesSkill, register_meeting_minutes_skill
from .skill_runtime import register_skill_runtime_tools
from .skill_gateway import SkillGateway
from .history_recall import register_history_recall_tool
from .host_services.apple_pim import ApplePimService, register_apple_pim_tools
from .session_store import SessionStore
from .tool_bus import LocalToolProvider, ToolBus
from .tools import Tool, register_file_tools
from .work_reports import register_work_report_tools


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return dispatch(args)
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


def dispatch(args: Namespace) -> int:
    registry = ModelRegistry.load(args.config)
    client = OpenAICompatibleClient()
    profile = registry.get(getattr(args, "profile", None))

    if args.command == "models":
        if args.models_command == "list":
            print(registry.as_table())
            return 0
        if args.models_command == "current":
            print(registry.default_profile)
            return 0
        if args.models_command == "use":
            set_default_profile(args.config, args.name)
            if os.getenv("WORK_AGENT_MODEL_PROFILE"):
                print(
                    "Updated config default, but WORK_AGENT_MODEL_PROFILE is set and will override it."
                )
            else:
                print(f"Default model profile set to {args.name}")
            return 0
        if args.models_command == "add":
            add_model_profile(
                args.config,
                {
                    "name": args.name,
                    "provider": args.provider,
                    "base_url": args.base_url,
                    "model": args.model,
                    "api_key_env": args.api_key_env,
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens,
                    "timeout_seconds": args.timeout_seconds,
                },
                set_default=args.set_default,
            )
            print(f"Added model profile {args.name}")
            return 0
        raise ValueError("Unknown models command.")

    if args.command == "chat":
        response = client.chat([{"role": "user", "content": args.prompt}], profile=profile)
        print(response.content)
        return 0

    if args.command == "run":
        tools = build_default_tools(args.workspace, client, profile)
        agent = ReActAgent(
            client=client,
            profile=profile,
            tools=tools,
            max_steps=args.max_steps,
        )
        result = agent.run(args.goal)
        print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))
        return 0

    if args.command == "skill" and args.skill_command == "meeting-minutes":
        skill = MeetingMinutesSkill(workspace_root=args.workspace, client=client, profile=profile)
        print(
            skill.run(
                {
                    "transcript_path": args.transcript,
                    "output_dir": args.output_dir,
                    "meeting_name": args.meeting_name,
                    "confirmed_info": args.confirmed_info or "",
                    "supplemental_paths": args.supplemental or [],
                }
            )
        )
        return 0

    raise ValueError("Unsupported command.")


def build_default_tools(
    workspace: str | Path,
    client: OpenAICompatibleClient,
    profile,
    *,
    data_workspace: str | Path | None = None,
    report_data_root: str | Path | None = None,
    include_shared_tools: bool = True,
    session_store: SessionStore | None = None,
    conversation_id: str | None = None,
    project_id: str | None = None,
    execution_account_id: str = "local",
    execution_turn_id: str = "",
    enabled_skill_ids: set[str] | None = None,
    friday_notification_handler: Callable[[dict[str, Any]], str] | None = None,
) -> ToolBus:
    bus = ToolBus()
    private_workspace = data_workspace or workspace

    core_tools = LocalToolProvider("core", provider_kind="local")
    register_file_tools(core_tools.registry, private_workspace)
    register_shell_tools(
        core_tools.registry,
        private_workspace,
        runtime_workspace_root=workspace,
        account_id=execution_account_id,
        turn_id=execution_turn_id,
        conversation_id=str(conversation_id or ""),
        project_id=str(project_id or ""),
    )
    if session_store is not None and conversation_id:
        register_history_recall_tool(
            core_tools.registry,
            session_store,
            conversation_id,
            project_id=str(project_id or ""),
        )
    if friday_notification_handler is not None:
        core_tools.registry.register(
            Tool(
                name="notify_user",
                description=(
                    "Friday-only delivery interface. Use kind=reminder for a one-way bell notification "
                    "that needs no reply; use kind=conversation for an important proactive message that "
                    "should appear in Friday's ongoing conversation and invite a response."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "kind": {
                            "type": "string",
                            "enum": ["reminder", "conversation"],
                        },
                        "title": {"type": "string"},
                        "body": {"type": "string"},
                        "deliver_at": {
                            "type": "string",
                            "description": (
                                "Optional Unix seconds or ISO 8601 timestamp with timezone. "
                                "Omit for immediate delivery."
                            ),
                        },
                    },
                    "required": ["kind", "body"],
                },
                handler=friday_notification_handler,
            )
        )
    bus.add_provider(core_tools)

    if not include_shared_tools:
        return bus

    # EventKit runs on the trusted host. The progressive skill gateway exposes
    # reads plus explicitly user-requested Reminder creation; Calendar events
    # and browser-side writes stay unavailable to the model.
    apple_pim_tools = LocalToolProvider("apple-schedule", provider_kind="skill")
    register_apple_pim_tools(apple_pim_tools.registry, ApplePimService(workspace))
    bus.add_provider(apple_pim_tools)

    work_report_tools = LocalToolProvider("work-reports", provider_kind="skill")
    register_work_report_tools(
        work_report_tools.registry,
        report_data_root or private_workspace,
    )
    bus.add_provider(work_report_tools)

    meeting_tools = LocalToolProvider("meeting", provider_kind="skill")
    register_meeting_minutes_skill(
        meeting_tools.registry,
        workspace_root=private_workspace,
        runtime_workspace_root=workspace,
        client=client,
        profile=profile,
    )
    bus.add_provider(meeting_tools)

    skill_tools = LocalToolProvider("skills", provider_kind="skill")
    register_skill_runtime_tools(skill_tools.registry, workspace, enabled_skill_ids=enabled_skill_ids)
    bus.add_provider(skill_tools)

    # Browser-like MCP servers hold cookies and tabs. Scope their process cache
    # to this account workspace + conversation so a logged-in browser session
    # can never be reused by another Work Agent account or conversation.
    mcp_scope_workspace = Path(data_workspace or workspace).resolve()
    mcp_scope = f"{mcp_scope_workspace}:{conversation_id or 'cli'}"
    mcp_tools = build_mcp_tool_provider(
        workspace,
        scope_id=mcp_scope,
        enabled_skill_ids=enabled_skill_ids,
    )
    bus.add_provider(mcp_tools)

    skill_gateway = LocalToolProvider("skill-gateway", provider_kind="local")
    skill_gateway.register(
        SkillGateway(
            workspace,
            [core_tools, apple_pim_tools, work_report_tools, meeting_tools, skill_tools, mcp_tools],
            enabled_skill_ids=enabled_skill_ids,
        ).as_tool()
    )
    bus.add_provider(skill_gateway)

    mcp_gateway = LocalToolProvider("mcp-gateway", provider_kind="local")
    mcp_gateway.register(MCPGateway(mcp_tools).as_tool())
    bus.add_provider(mcp_gateway)

    return bus


def set_default_profile(config_path: str | Path, name: str) -> None:
    path = Path(config_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    names = {item["name"] for item in data.get("profiles", [])}
    if name not in names:
        available = ", ".join(sorted(names))
        raise ValueError(f"Unknown model profile {name!r}. Available: {available}")
    data["default_profile"] = name
    write_model_config(path, data)


def add_model_profile(config_path: str | Path, profile_data: dict, *, set_default: bool) -> None:
    path = Path(config_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    profile = ModelProfile.from_dict(profile_data)
    profiles = data.setdefault("profiles", [])
    if any(item["name"] == profile.name for item in profiles):
        raise ValueError(f"Model profile already exists: {profile.name}")
    profiles.append(profile_data)
    if set_default:
        data["default_profile"] = profile.name
    write_model_config(path, data)


def update_model_profile(config_path: str | Path, name: str, profile_data: dict) -> None:
    path = Path(config_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    profile = ModelProfile.from_dict({**profile_data, "name": name})
    for index, existing in enumerate(data.get("profiles", [])):
        if existing.get("name") != name:
            continue
        data["profiles"][index] = {
            "name": name,
            "provider": profile.provider,
            "base_url": profile.base_url,
            "model": profile.model,
            "api_key_env": profile.api_key_env,
            "temperature": profile.temperature,
            "max_tokens": profile.max_tokens,
            "timeout_seconds": profile.timeout_seconds,
            "supports_vision": profile.supports_vision,
        }
        write_model_config(path, data)
        return
    raise ValueError(f"Unknown model profile {name!r}")


def delete_model_profile(config_path: str | Path, name: str) -> dict:
    path = Path(config_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    profiles = list(data.get("profiles", []))
    if data.get("default_profile") == name:
        raise ValueError("当前正在使用的模型不能删除，请先切换到其他模型。")
    if len(profiles) <= 1:
        raise ValueError("至少需要保留一个模型配置。")
    removed = next((item for item in profiles if item.get("name") == name), None)
    if removed is None:
        raise ValueError(f"Unknown model profile {name!r}")
    data["profiles"] = [item for item in profiles if item.get("name") != name]
    write_model_config(path, data)
    return removed


def update_model_profile_api_key_env(config_path: str | Path, name: str, api_key_env: str) -> None:
    path = Path(config_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    for profile in data.get("profiles", []):
        if profile.get("name") == name:
            profile["api_key_env"] = api_key_env
            write_model_config(path, data)
            return
    raise ValueError(f"Unknown model profile {name!r}")


def write_model_config(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def parse_args(argv: list[str] | None) -> Namespace:
    parser = ArgumentParser(prog="work-agent", description="Local work agent framework.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Model profiles JSON path.")
    parser.add_argument("--profile", help="Model profile name. Defaults to config/env.")
    parser.add_argument("--workspace", default=".", help="Workspace root for file tools.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    models = subparsers.add_parser("models", help="Inspect model profiles.")
    models_sub = models.add_subparsers(dest="models_command", required=True)
    models_sub.add_parser("list", help="List configured model profiles.")
    models_sub.add_parser("current", help="Show current default profile.")
    models_use = models_sub.add_parser("use", help="Persistently set the default model profile.")
    models_use.add_argument("name")
    models_add = models_sub.add_parser("add", help="Add an OpenAI-compatible model profile.")
    models_add.add_argument("name")
    models_add.add_argument("--provider", default="openai-compatible")
    models_add.add_argument("--base-url", required=True)
    models_add.add_argument("--model", required=True)
    models_add.add_argument("--api-key-env", required=True)
    models_add.add_argument("--temperature", type=float, default=0.6)
    models_add.add_argument("--max-tokens", type=int, default=8192)
    models_add.add_argument("--timeout-seconds", type=int, default=120)
    models_add.add_argument("--set-default", action="store_true")

    chat = subparsers.add_parser("chat", help="Send one prompt to the selected model.")
    chat.add_argument("prompt")

    run = subparsers.add_parser("run", help="Run the ReAct agent.")
    run.add_argument("goal")
    run.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)

    skill = subparsers.add_parser("skill", help="Run a skill directly.")
    skill_sub = skill.add_subparsers(dest="skill_command", required=True)
    meeting = skill_sub.add_parser("meeting-minutes", help="Generate meeting minutes from ASR text.")
    meeting.add_argument("--transcript", required=True)
    meeting.add_argument("--output-dir", default="meet_files")
    meeting.add_argument("--meeting-name", required=True)
    meeting.add_argument("--confirmed-info")
    meeting.add_argument("--supplemental", action="append", default=[])

    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
