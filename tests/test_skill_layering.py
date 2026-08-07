from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from work_agent_core.cli import build_default_tools
from work_agent_core.config import ModelProfile
from work_agent_core.llm import OpenAICompatibleClient
from work_agent_core.session_store import SessionStore
from work_agent_core.skill_gateway import SkillGateway, normalize_skill_tool_arguments
from work_agent_core.tool_bus import LocalToolProvider
from work_agent_core.tools import Tool


WORKSPACE = Path(__file__).resolve().parents[1]
PROFILE = ModelProfile(
    name="layering-test",
    provider="openai-compatible",
    base_url="https://example.invalid/v1",
    model="test-model",
    api_key_env="UNUSED",
)
CORE_MODEL_TOOLS = {
    "apply_unified_patch",
    "edit_text_file",
    "list_workspace_files",
    "mcporter",
    "read_text_file",
    "shell_exec",
    "sys_skill",
    "write_text_file",
}


class SkillLayeringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bus = build_default_tools(WORKSPACE, OpenAICompatibleClient(), PROFILE)

    def test_core_model_surface_is_stable_and_skill_tools_are_hidden(self) -> None:
        model_names = {tool.name for tool in self.bus.list_model_tools()}
        all_names = {tool.name for tool in self.bus.list()}

        self.assertEqual(model_names, CORE_MODEL_TOOLS)
        self.assertIn("transcribe_meeting_audio", all_names)
        self.assertIn("check_meeting_asr_progress", all_names)
        self.assertIn("get_apple_schedule_status", all_names)
        self.assertIn("list_apple_schedule", all_names)
        self.assertIn("create_apple_reminder", all_names)
        self.assertIn("anysearch_search", all_names)
        self.assertNotIn("transcribe_meeting_audio", model_names)
        self.assertNotIn("anysearch_search", model_names)
        self.assertNotIn("get_apple_schedule_status", model_names)
        with self.assertRaises(KeyError):
            self.bus.get_model_tool("transcribe_meeting_audio")

    def test_history_recall_is_a_request_scoped_core_tool(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(WORKSPACE, session_dir=Path(directory) / "sessions")
            bus = build_default_tools(
                WORKSPACE,
                OpenAICompatibleClient(),
                PROFILE,
                session_store=store,
                conversation_id="conversation-layering-test",
            )
            self.assertIn("recall_chat_history", {tool.name for tool in bus.list_model_tools()})
            opened = json.loads(
                bus.get_model_tool("sys_skill").handler(
                    {"op": "open", "skill_id": "recall-chat-history", "max_chars": 5000}
                )
            )
            self.assertIn("recall_chat_history", {item["name"] for item in opened["available_tools"]})

    def test_work_report_store_can_be_separate_from_file_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            file_workspace = root / "workspace"
            report_account = root / "account"
            bus = build_default_tools(
                WORKSPACE,
                OpenAICompatibleClient(),
                PROFILE,
                data_workspace=file_workspace,
                report_data_root=report_account,
            )

            result = json.loads(bus.get_model_tool("sys_skill").handler({
                "op": "call",
                "skill_id": "work-reports",
                "tool_name": "save_work_report",
                "arguments": {
                    "report_type": "daily",
                    "target_date": "2026-07-09",
                    "content": "# test",
                },
            }))

            self.assertTrue(result["ok"])
            self.assertTrue((report_account / "work_reports/daily/2026-07-09.md").is_file())
            self.assertFalse((file_workspace / "work_reports/daily/2026-07-09.md").exists())

    def test_sys_skill_lists_opens_and_reveals_one_schema_on_demand(self) -> None:
        gateway = self.bus.get_model_tool("sys_skill")

        index = json.loads(gateway.handler({"op": "list"}))
        skills_by_id = {item["id"]: item for item in index["skills"]}
        self.assertIn("meeting-minutes", skills_by_id)
        self.assertIn("apple-schedule", skills_by_id)
        self.assertIn("official-document", skills_by_id)
        self.assertTrue(skills_by_id["official-document"]["default_enabled"])

        opened = json.loads(
            gateway.handler({"op": "open", "skill_id": "meeting-minutes", "max_chars": 5000})
        )
        self.assertEqual(opened["skill_id"], "meeting-minutes")
        self.assertIn("transcribe_meeting_audio", {item["name"] for item in opened["available_tools"]})
        self.assertIn("check_meeting_asr_progress", {item["name"] for item in opened["available_tools"]})
        self.assertIn("sys_skill", opened["execution_guidance"])
        self.assertNotIn("create_docx_from_markdown", {item["name"] for item in opened["available_tools"]})
        self.assertNotIn("generate_meeting_minutes", {item["name"] for item in opened["available_tools"]})

        apple_schedule = json.loads(
            gateway.handler({"op": "open", "skill_id": "apple-schedule", "max_chars": 5000})
        )
        self.assertIn("get_apple_schedule_status", {item["name"] for item in apple_schedule["available_tools"]})
        self.assertIn("list_apple_schedule", {item["name"] for item in apple_schedule["available_tools"]})
        self.assertIn("create_apple_reminder", {item["name"] for item in apple_schedule["available_tools"]})

        official = json.loads(
            gateway.handler({"op": "open", "skill_id": "official-document", "max_chars": 8000})
        )
        self.assertIn("docx", official["instructions"])
        self.assertIn("GB/T 9704", official["instructions"])

        reports = json.loads(
            gateway.handler({"op": "open", "skill_id": "work-reports", "max_chars": 8000})
        )
        report_tools = {item["name"] for item in reports["available_tools"]}
        self.assertIn("collect_work_report_evidence", report_tools)
        self.assertIn("save_work_report", report_tools)
        self.assertIn("read_saved_work_report", report_tools)
        self.assertIn("delete_work_report", report_tools)
        self.assertIn("check_work_report_status", report_tools)
        self.assertIn("update_workday_calendar", report_tools)

        progress = json.loads(
            gateway.handler(
                {
                    "op": "call",
                    "skill_id": "meeting-minutes",
                    "tool_name": "check_meeting_asr_progress",
                    "arguments": {
                        "input_path": "meet_files/attachments/20260715-121324-新录音 6.m4a"
                    },
                }
            )
        )
        self.assertTrue(progress["ok"])
        self.assertIn("新录音 6.m4a", progress["input_path"])

        shown = json.loads(
            gateway.handler(
                {"op": "show", "skill_id": "anysearch", "tool_name": "anysearch_search"}
            )
        )
        self.assertEqual(shown["tool"]["name"], "anysearch_search")
        self.assertIn("query", shown["tool"]["parameters"]["properties"])

    def test_sys_skill_rejects_cross_skill_tool_calls(self) -> None:
        gateway = self.bus.get_model_tool("sys_skill")
        with self.assertRaises(KeyError):
            gateway.handler(
                {
                    "op": "call",
                    "skill_id": "docx",
                    "tool_name": "anysearch_search",
                    "arguments": {"query": "test"},
                }
            )
        with self.assertRaises(KeyError):
            gateway.handler(
                {
                    "op": "call",
                    "skill_id": "meeting-minutes",
                    "tool_name": "create_docx_from_markdown",
                    "arguments": {
                        "markdown_content": "# 测试",
                        "output_path": "meet_files/test.docx",
                    },
                }
            )

    def test_sys_skill_reports_conditional_arguments_and_injects_scope(self) -> None:
        gateway = self.bus.get_model_tool("sys_skill")
        with self.assertRaisesRegex(ValueError, "skill_id"):
            gateway.handler({"op": "show", "tool_name": "anysearch_search"})
        with self.assertRaisesRegex(ValueError, "tool_name"):
            gateway.handler({"op": "show", "skill_id": "anysearch"})

        captured: list[dict] = []
        provider = LocalToolProvider("script-test")
        provider.register(
            Tool(
                name="run_skill_script",
                description="test script runner",
                parameters={
                    "type": "object",
                    "properties": {
                        "skill_id": {"type": "string"},
                        "script_path": {"type": "string"},
                    },
                    "required": ["skill_id", "script_path"],
                },
                handler=lambda args: captured.append(dict(args)) or "ok",
            )
        )
        scoped_gateway = SkillGateway(
            WORKSPACE,
            [provider],
            enabled_skill_ids={"anysearch"},
        ).as_tool()
        self.assertEqual(
            scoped_gateway.handler(
                {
                    "op": "call",
                    "skill_id": "anysearch",
                    "tool_name": "run_skill_script",
                    "arguments": {"script_path": "scripts/probe.py"},
                }
            ),
            "ok",
        )
        self.assertEqual(captured[0]["skill_id"], "anysearch")

    def test_composite_skill_can_call_declared_skill_dependencies(self) -> None:
        provider = LocalToolProvider("dependency-test", provider_kind="mcp")
        for name in ("anysearch_search", "browser_navigate"):
            provider.register(
                Tool(
                    name=name,
                    description=name,
                    parameters={"type": "object", "properties": {}},
                    handler=lambda _args, name=name: name,
                )
            )

        gateway = SkillGateway(
            WORKSPACE,
            [provider],
            enabled_skill_ids={"company-verification", "anysearch", "edge-browser"},
        ).as_tool()
        opened = json.loads(
            gateway.handler({"op": "open", "skill_id": "company-verification", "max_chars": 5000})
        )
        available = {item["name"] for item in opened["available_tools"]}
        self.assertIn("anysearch_search", available)
        self.assertIn("browser_navigate", available)
        self.assertNotIn("company_verification_query", available)

        result = gateway.handler(
            {
                "op": "call",
                "skill_id": "company-verification",
                "tool_name": "anysearch_search",
                "arguments": {},
            }
        )
        self.assertEqual(result, "anysearch_search")

    def test_document_skill_accepts_audio_style_input_path_alias(self) -> None:
        tool = Tool(
            name="process_office_document",
            description="test",
            parameters={"type": "object", "properties": {"path": {"type": "string"}}},
            handler=lambda _args: "",
        )
        normalized = normalize_skill_tool_arguments(
            tool,
            {
                "input_path": "meet_files/course.pdf",
                "output_dir": "meet_files/course_materials",
            },
        )
        self.assertEqual(normalized["path"], "meet_files/course.pdf")
        self.assertEqual(normalized["output_dir"], "meet_files/course_materials")


if __name__ == "__main__":
    unittest.main()
