from __future__ import annotations

from dataclasses import replace
from typing import Any, Protocol, runtime_checkable

from .tools import Tool, ToolRegistry


@runtime_checkable
class ToolProvider(Protocol):
    """A source of callable tools.

    Providers hide where tools come from: in-process Python handlers, skill
    manifests, MCP servers, browser runtimes, or future remote services. The
    agent loop should depend on this interface instead of concrete registries.
    """

    provider_id: str
    provider_kind: str

    def list_tools(self) -> list[Tool]:
        ...

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        ...


class LocalToolProvider:
    """Provider backed by the existing in-process ToolRegistry."""

    def __init__(self, provider_id: str, *, provider_kind: str = "local") -> None:
        self.provider_id = provider_id
        self.provider_kind = provider_kind
        self.registry = ToolRegistry()

    def register(self, tool: Tool) -> None:
        self.registry.register(tool)

    def list_tools(self) -> list[Tool]:
        return self.registry.list()

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        tool = self.registry.get(name)
        return tool.handler(arguments)


class ToolBus:
    """Merged tool surface for the agent runtime.

    This is the architectural seam Friday has and this project was missing:
    ReAct sees a single bus, while capabilities are mounted by providers. Name
    collisions are resolved by provider order, with the first provider owning a
    tool name. This keeps the model-facing schema deterministic.
    """

    def __init__(self) -> None:
        self._providers: list[ToolProvider] = []

    def add_provider(self, provider: ToolProvider) -> None:
        provider_id = str(getattr(provider, "provider_id", "") or "").strip()
        if not provider_id:
            raise ValueError("Tool provider must have provider_id")
        if any(existing.provider_id == provider_id for existing in self._providers):
            raise ValueError(f"Tool provider already mounted: {provider_id}")
        self._providers.append(provider)

    def providers(self) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        seen_tool_names: set[str] = set()
        for provider in self._providers:
            tools = []
            for tool in provider.list_tools():
                owned = tool.name not in seen_tool_names
                tools.append({"name": tool.name, "owned": owned})
                seen_tool_names.add(tool.name)
            provider_payload: dict[str, Any] = {
                "id": provider.provider_id,
                "kind": provider.provider_kind,
                "tools": tools,
            }
            status = getattr(provider, "status", None)
            if callable(status):
                try:
                    provider_payload.update(status())
                except Exception as error:
                    provider_payload.update(
                        {
                            "status": "error",
                            "detail": f"{type(error).__name__}: {error}",
                        }
                    )
            payload.append(provider_payload)
        return payload

    def list(self) -> list[Tool]:
        """Return every mounted tool, including skill backends hidden from the model."""
        return self._list_tools(include_hidden=True)

    def list_model_tools(self) -> list[Tool]:
        """Return the stable model surface: core/meta tools plus configured MCP tools."""
        return self._list_tools(include_hidden=False)

    def _list_tools(self, *, include_hidden: bool) -> list[Tool]:
        tools: list[Tool] = []
        seen: set[str] = set()
        for provider in self._providers:
            if not include_hidden and provider.provider_kind in {"skill", "mcp"}:
                continue
            for tool in provider.list_tools():
                if tool.name in seen:
                    continue
                seen.add(tool.name)
                tools.append(self._wrap_tool(provider, tool))
        return sorted(tools, key=lambda tool: tool.name)

    def get(self, name: str) -> Tool:
        """Resolve any mounted tool for trusted internal gateway calls."""
        return self._get(name, include_hidden=True)

    def get_model_tool(self, name: str) -> Tool:
        """Resolve only tools exposed in the model's native tools schema."""
        return self._get(name, include_hidden=False)

    def _get(self, name: str, *, include_hidden: bool) -> Tool:
        for provider in self._providers:
            if not include_hidden and provider.provider_kind in {"skill", "mcp"}:
                continue
            for tool in provider.list_tools():
                if tool.name == name:
                    return self._wrap_tool(provider, tool)
        available_tools = self.list() if include_hidden else self.list_model_tools()
        available = ", ".join(tool.name for tool in available_tools)
        raise KeyError(f"Unknown tool {name!r}. Available tools: {available}")

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        return self.get(name).handler(arguments)

    def prompt_block(self) -> str:
        return "\n".join(tool.render_for_prompt() for tool in self.list())

    def _wrap_tool(self, provider: ToolProvider, tool: Tool) -> Tool:
        return replace(
            tool,
            handler=lambda arguments, provider=provider, tool_name=tool.name: provider.call_tool(
                tool_name,
                arguments,
            ),
            provider_id=provider.provider_id,
            provider_kind=provider.provider_kind,
        )
