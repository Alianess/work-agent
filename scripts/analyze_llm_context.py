#!/usr/bin/env python3
"""构建并分析 Work Agent 新聊天发送给模型的完整请求。

默认只模拟发送前装配，不访问模型：
    .venv/bin/python scripts/analyze_llm_context.py

指定内容、模型和推理强度：
    .venv/bin/python scripts/analyze_llm_context.py \
      --prompt 你好 --profile lmstudio-qwen3.8-27b --reasoning-effort very_high

需要同时真实发送并取得供应商 usage 时显式加 --send。
输出 JSON 不包含 Authorization 或 API key。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from work_agent_core.cli import build_default_tools  # noqa: E402
from work_agent_core.config import ModelRegistry, load_env_file  # noqa: E402
from work_agent_core.llm import (  # noqa: E402
    OpenAICompatibleClient,
    build_chat_tools_payload,
    normalize_reasoning_effort,
)
from work_agent_core.react import ReActAgent  # noqa: E402
from work_agent_core.session_runtime import ConversationRuntime  # noqa: E402
from work_agent_core.web_server import (  # noqa: E402
    WORKSPACE_ROOT,
    agent_extra_read_roots,
    agent_stable_system_context,
    build_append_only_turn_runtime_context,
    enabled_skill_ids,
    get_session_store,
    user_data_dir,
    user_file_reference_index_path,
)

DEFAULT_OUTPUT = ROOT / "meet_files/debug_traces/llm_context_analysis.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="你好", help="模拟的新聊天用户消息")
    parser.add_argument("--profile", default="", help="模型 profile；默认使用当前默认模型")
    parser.add_argument(
        "--reasoning-effort",
        default="very_high",
        choices=("light", "medium", "high", "very_high"),
        help="推理强度；默认 very_high（前端‘极高’）",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="完整请求 JSON 输出路径")
    parser.add_argument("--show", action="store_true", help="同时把完整 JSON 打印到终端")
    parser.add_argument("--send", action="store_true", help="真实发送给所选模型并打印 usage")
    return parser.parse_args(argv)


def build_frontend_new_chat_payload(
    *,
    prompt: str,
    profile_name: str,
    reasoning_effort: str,
) -> tuple[dict[str, Any], OpenAICompatibleClient, Any]:
    """走与前端新聊天相同的系统上下文、工具和 payload 构建器。"""

    registry = ModelRegistry.load(ROOT / "config/model_profiles.json")
    profile = registry.get(profile_name or registry.default_profile)
    client = OpenAICompatibleClient()
    store = get_session_store()
    conversation_id = f"context-analysis-{int(time.time() * 1000)}"
    user_message = {"role": "user", "content": prompt}

    tools = build_default_tools(
        WORKSPACE_ROOT,
        client,
        profile,
        data_workspace=WORKSPACE_ROOT,
        report_data_root=user_data_dir(),
        include_shared_tools=True,
        session_store=store,
        conversation_id=conversation_id,
        project_id="",
        execution_account_id="context-analysis",
        execution_turn_id="context-analysis",
        enabled_skill_ids=enabled_skill_ids(),
        friday_notification_handler=lambda _payload: "",
        agent_reminder_source=lambda: [],
        sandbox_auto_allow=True,
        recall_data_root=user_data_dir(),
        extra_read_roots=agent_extra_read_roots(),
    )
    turn_runtime_context = build_append_only_turn_runtime_context(
        session_messages=[],
        latest_user_content=prompt,
        profile=profile,
        conversation_id=conversation_id,
        project_context="",
        project_id="",
        summary_message_count=0,
        skill_hint=None,
        context_file_paths=[],
        workspace_root=WORKSPACE_ROOT,
        index_path=user_file_reference_index_path(),
        session_metadata={},
    )
    agent = ReActAgent(
        client=client,
        profile=profile,
        tools=tools,
        workspace_root=WORKSPACE_ROOT,
        extra_system_context=agent_stable_system_context(mode="task"),
        reasoning_effort=reasoning_effort,
        late_task_plan_context=False,
    )
    runtime = ConversationRuntime.from_messages(
        [
            {"role": "system", "content": turn_runtime_context},
            user_message,
        ],
        session_id=conversation_id,
    )
    request_messages = agent._request_messages(runtime)
    tool_schemas = agent._tool_schemas()
    payload = build_chat_tools_payload(
        request_messages,
        profile=profile,
        tools=tool_schemas,
        tool_choice="auto",
        reasoning_effort=reasoning_effort,
        request_usage=True,
    )
    return payload, client, profile


def payload_summary(payload: dict[str, Any], output: Path) -> dict[str, Any]:
    messages = payload.get("messages") or []
    tools = payload.get("tools") or []
    return {
        "output": str(output.resolve()),
        "bytes": output.stat().st_size,
        "messages": len(messages),
        "message_roles": [str(item.get("role") or "") for item in messages],
        "message_chars": [len(str(item.get("content") or "")) for item in messages],
        "tools": len(tools),
        "tool_names": [str(item.get("function", {}).get("name") or "") for item in tools],
        "model": payload.get("model"),
        "reasoning_effort": payload.get("reasoning_effort"),
        "max_tokens": payload.get("max_tokens"),
        "stream": payload.get("stream"),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_env_file(ROOT / ".env")
    effort = normalize_reasoning_effort(args.reasoning_effort)
    payload, client, profile = build_frontend_new_chat_payload(
        prompt=args.prompt,
        profile_name=args.profile,
        reasoning_effort=effort,
    )
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload_summary(payload, output), ensure_ascii=False, indent=2))
    if args.show:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.send:
        started = time.monotonic()
        response = client.chat_tools_stream(
            payload["messages"],
            profile=profile,
            tools=payload.get("tools"),
            tool_choice=payload.get("tool_choice"),
            reasoning_effort=effort,
        )
        print(
            json.dumps(
                {
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "finish_reason": response.raw.get("choices", [{}])[0].get("finish_reason"),
                    "usage": response.raw.get("usage") or {},
                    "content": response.content,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
