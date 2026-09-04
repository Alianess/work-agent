from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from work_agent_core.cli import build_default_tools
from work_agent_core.config import ModelProfile
from work_agent_core.llm import OpenAICompatibleClient
from work_agent_core.memory import estimate_context_tokens
from work_agent_core.react import ReActAgent
from work_agent_core.session_store import SessionStore


WORKSPACE = Path(__file__).resolve().parents[1]
PROFILE = ModelProfile(
    name="tool-schema-budget-test",
    provider="openai-compatible",
    base_url="https://example.invalid/v1",
    model="test-model",
    api_key_env="UNUSED",
)


class ToolSchemaBudgetTests(unittest.TestCase):
    def test_default_chat_tool_surface_stays_progressive_and_compact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)
            store = SessionStore(WORKSPACE, session_dir=data_root / "sessions")
            bus = build_default_tools(
                WORKSPACE,
                OpenAICompatibleClient(),
                PROFILE,
                session_store=store,
                conversation_id="tool-schema-budget",
                friday_notification_handler=lambda _payload: "",
                recall_data_root=data_root,
            )
            agent = ReActAgent(
                client=object(),  # type: ignore[arg-type]
                profile=PROFILE,
                tools=bus,
            )
            schemas = agent._tool_schemas()

        names = {item["function"]["name"] for item in schemas}
        self.assertEqual(
            names,
            {
                "edit_text_file",
                "forget",
                "mcporter",
                "notify_user",
                "read_file",
                "recall",
                "recall_expand",
                "remember",
                "shell_exec",
                "sys_skill",
                "update_plan",
                "write_text_file",
            },
        )
        estimated_tokens = sum(
            estimate_context_tokens(
                json.dumps(item, ensure_ascii=False, separators=(",", ":"))
            )
            for item in schemas
        )
        self.assertLessEqual(estimated_tokens, 1700)

        descriptions = {
            item["function"]["name"]: item["function"]["description"]
            for item in schemas
        }
        self.assertNotIn("先 open", descriptions["sys_skill"])
        self.assertNotIn("Use it when", descriptions["recall"])
        self.assertNotIn("recall_expand only when", descriptions["recall"])


if __name__ == "__main__":
    unittest.main()
