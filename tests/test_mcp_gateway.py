from __future__ import annotations

import unittest

from work_agent_core.mcp_gateway import (
    MCPGateway,
    normalize_playwright_target_references,
    playwright_click_lost_snapshot_target,
)
from work_agent_core.tools import Tool


class _FakeProvider:
    def __init__(self, result: str) -> None:
        self.result = result

    def list_tools(self):
        return [
            Tool(
                name="browser_click",
                description="test browser click",
                parameters={"type": "object"},
                handler=lambda _args: "",
            )
        ]

    def status(self):
        return {}

    def call_tool(self, _tool_name, _arguments):
        return self.result


class MCPGatewayTests(unittest.TestCase):
    def test_playwright_snapshot_references_are_normalized_in_nested_form_fields(self) -> None:
        normalized = normalize_playwright_target_references(
            "browser_fill_form",
            {"fields": [{"target": "[ref=e36]", "value": "hello"}]},
        )
        self.assertEqual(normalized["fields"][0]["target"], "e36")

    def test_only_browser_tools_normalize_snapshot_references(self) -> None:
        normalized = normalize_playwright_target_references(
            "browser_click", {"target": "[ref=f1e6]"}
        )
        self.assertEqual(normalized["target"], "f1e6")
        unchanged = normalize_playwright_target_references("unrelated_tool", {"target": "[ref=f1e6]"})
        self.assertEqual(unchanged["target"], "[ref=f1e6]")

    def test_detects_empty_button_fallback_for_snapshot_ref(self) -> None:
        result = """### Ran Playwright code
```js
await page.locator('#global').getByRole('button').filter({ hasText: /^$/ }).click();
```"""
        self.assertTrue(playwright_click_lost_snapshot_target("browser_click", {"target": "e1281"}, result))
        self.assertFalse(playwright_click_lost_snapshot_target("browser_click", {"target": "button.send"}, result))

    def test_gateway_returns_error_instead_of_false_success_for_empty_button_fallback(self) -> None:
        result = """### Ran Playwright code
```js
await page.getByRole('button').filter({ hasText: /^$/ }).click();
```"""
        gateway = MCPGateway(_FakeProvider(result))
        observed = gateway.handle(
            {"op": "call", "tool_name": "browser_click", "arguments": {"target": "e1281"}}
        )
        self.assertTrue(observed.startswith("MCP_TOOL_ERROR:"), observed)


if __name__ == "__main__":
    unittest.main()
