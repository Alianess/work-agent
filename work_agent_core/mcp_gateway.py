from __future__ import annotations

from typing import Any
import json
import re

from .mcp_provider import MCPToolProvider
from .tools import Tool


class MCPGateway:
    """Compact Friday-style gateway for MCP tools hidden from top-level schemas."""

    def __init__(self, provider: MCPToolProvider) -> None:
        self.provider = provider

    def as_tool(self) -> Tool:
        return Tool(
            name="mcporter",
            description=(
                "外部 MCP 能力分层入口。用 list 查看已连接服务和工具名，show 按需读取单个工具参数，"
                "call 调用该工具。外部 MCP 工具不会直接铺在顶层 tools。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": ["list", "show", "call"]},
                    "tool_name": {"type": "string"},
                    "arguments": {"type": "object"},
                },
                "required": ["op"],
            },
            handler=self.handle,
            metadata={"layer": "mcp_gateway"},
        )

    def handle(self, args: dict[str, Any]) -> str:
        op = str(args.get("op") or "").strip().lower()
        if op == "list":
            return json.dumps(
                {
                    "provider": self.provider.status(),
                    "tools": [
                        {"name": tool.name, "description": tool.description}
                        for tool in self.provider.list_tools()
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        tool_name = str(args.get("tool_name") or "").strip()
        if not tool_name:
            raise ValueError("mcporter 的 show/call 操作必须提供 tool_name。")
        tools = {tool.name: tool for tool in self.provider.list_tools()}
        tool = tools.get(tool_name)
        if tool is None:
            available = "、".join(sorted(tools)) or "无"
            raise KeyError(f"未找到 MCP 工具 {tool_name!r}。可用工具：{available}")
        if op == "show":
            return json.dumps(
                {
                    "tool": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                        "provider": tool.metadata,
                    }
                },
                ensure_ascii=False,
                indent=2,
            )
        if op == "call":
            arguments = args.get("arguments")
            if arguments is None:
                arguments = {}
            if not isinstance(arguments, dict):
                raise ValueError("mcporter.call 的 arguments 必须是对象。")
            arguments = normalize_playwright_target_references(tool_name, arguments)
            result = self.provider.call_tool(tool_name, arguments)
            if playwright_click_lost_snapshot_target(tool_name, arguments, result):
                # Playwright MCP 0.0.78 can turn a snapshot ref for an unnamed
                # icon button into a page-wide empty-text locator.  That command
                # may click a different control while still returning success.
                # Surface it as an error so the agent does not claim success or
                # keep retrying the same unsafe locator.
                return (
                    "MCP_TOOL_ERROR: browser_click 丢失了快照元素引用，实际执行的是页面范围的"
                    "空文本 button 定位，不能确认点击了 target。不要重试该点击；如是在聊天输入框发送消息，"
                    "对刚刚确认的 textbox 使用 browser_type，并传 submit: true。"
                )
            return result
        raise ValueError(f"不支持的 mcporter 操作：{op or '（空）'}")


_PLAYWRIGHT_SNAPSHOT_REF = re.compile(r"^\[ref=([A-Za-z0-9_-]+)\]$")
_PLAYWRIGHT_SNAPSHOT_TARGET = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
_EMPTY_BUTTON_FALLBACK = re.compile(
    r"getByRole\(['\"]button['\"]\)\.filter\(\{\s*hasText:\s*/\^\$/\s*\}\)",
)


def normalize_playwright_target_references(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Convert a snapshot's display form ``[ref=e12]`` to Playwright's ``e12``.

    Playwright MCP renders accessible snapshots with brackets, but interaction
    tools accept the bare reference.  This adapter removes a needless retry and
    prevents the model from falling back to a broad CSS selector that can match
    a hidden duplicate control.
    """
    if not tool_name.startswith("browser_"):
        return arguments

    def normalize(value: Any, *, key: str = "") -> Any:
        if isinstance(value, dict):
            return {str(item_key): normalize(item_value, key=str(item_key)) for item_key, item_value in value.items()}
        if isinstance(value, list):
            return [normalize(item) for item in value]
        if key == "target" and isinstance(value, str):
            match = _PLAYWRIGHT_SNAPSHOT_REF.fullmatch(value.strip())
            if match:
                return match.group(1)
        return value

    return normalize(arguments)


def playwright_click_lost_snapshot_target(tool_name: str, arguments: dict[str, Any], result: str) -> bool:
    """Detect Playwright MCP's unsafe fallback for unnamed icon buttons.

    A snapshot reference must remain an exact element target.  The older MCP
    server used here sometimes emitted a page-wide ``hasText: /^$/`` selector
    instead, which is ambiguous whenever a page has multiple icon buttons.
    """
    if tool_name != "browser_click" or not isinstance(result, str):
        return False
    target = arguments.get("target")
    if not isinstance(target, str) or not _PLAYWRIGHT_SNAPSHOT_TARGET.fullmatch(target.strip()):
        return False
    return bool(_EMPTY_BUTTON_FALLBACK.search(result))
