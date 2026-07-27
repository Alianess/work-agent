from __future__ import annotations

import json
import os
import subprocess
import unittest
from pathlib import Path

from work_agent_core.cli import build_default_tools
from work_agent_core.config import ModelProfile
from work_agent_core.llm import OpenAICompatibleClient
from work_agent_core.skill_manifest import load_single_skill_manifest
from work_agent_core import web_server


WORKSPACE = Path(__file__).resolve().parents[1]
PROFILE = ModelProfile("test", "openai-compatible", "https://example.invalid/v1", "test", "UNUSED")


class SearchSkillSwitchTests(unittest.TestCase):
    def test_search_and_browser_are_separate_skills(self) -> None:
        expected = {
            "baidu-web-search": {"baidu_web_search"},
            "tavily-search": {"tavily_search", "tavily_extract", "tavily_usage"},
            "edge-browser": set(),
        }
        for skill_id, tool_names in expected.items():
            skill_dir = WORKSPACE / "work_agent_skills" / skill_id
            manifest = load_single_skill_manifest(WORKSPACE / "work_agent_skills", skill_dir)
            self.assertIsNotNone(manifest)
            config = json.loads((skill_dir / "work_agent.json").read_text(encoding="utf-8"))
            self.assertEqual({item["name"] for item in config.get("tools", [])}, tool_names)

    def test_only_anysearch_mode_hides_costly_search_skills_and_tools(self) -> None:
        bus = build_default_tools(
            WORKSPACE,
            OpenAICompatibleClient(),
            PROFILE,
            enabled_skill_ids={"anysearch"},
        )
        all_tools = {tool.name for tool in bus.list()}
        self.assertIn("anysearch_search", all_tools)
        self.assertNotIn("baidu_web_search", all_tools)
        self.assertNotIn("tavily_search", all_tools)
        gateway = bus.get_model_tool("sys_skill")
        skills = json.loads(gateway.handler({"op": "list"}))["skills"]
        self.assertEqual({item["id"] for item in skills}, {"anysearch"})
        with self.assertRaises(PermissionError):
            gateway.handler({"op": "open", "skill_id": "baidu-web-search"})

    def test_closed_skill_reports_that_it_is_closed(self) -> None:
        bus = build_default_tools(
            WORKSPACE,
            OpenAICompatibleClient(),
            PROFILE,
            enabled_skill_ids={"anysearch"},
        )
        gateway = bus.get_model_tool("sys_skill")
        with self.assertRaisesRegex(PermissionError, "当前已关闭"):
            gateway.handler({"op": "open", "skill_id": "baidu-web-search"})

    def test_search_clients_do_not_echo_missing_keys(self) -> None:
        env = dict(os.environ)
        env.pop("BAIDU_API_KEY", None)
        baidu = subprocess.run(
            ["node", str(WORKSPACE / "work_agent_skills" / "baidu-web-search" / "scripts" / "baidu_web_search.js"), "--query", "具身智能"],
            capture_output=True, text=True, env=env, check=False,
        )
        self.assertEqual(baidu.returncode, 2)
        self.assertIn("BAIDU_API_KEY", json.loads(baidu.stdout)["error"])
        env.pop("TAVILY_API_KEY", None)
        tavily = subprocess.run(
            [str(WORKSPACE / ".venv" / "bin" / "python"), str(WORKSPACE / "work_agent_skills" / "tavily-search" / "scripts" / "tavily_search.py"), "usage"],
            capture_output=True, text=True, env=env, check=False,
        )
        self.assertEqual(tavily.returncode, 2)
        self.assertIn("TAVILY_API_KEY", json.loads(tavily.stdout)["error"])

    def test_skill_catalog_can_return_a_save_message(self) -> None:
        payload = web_server.skill_catalog_payload(message="edge-browser 已启用")
        self.assertEqual(payload["message"], "edge-browser 已启用")
        self.assertTrue(payload["skills"])

    def test_skill_instruction_routes_only_match_the_explicit_endpoint(self) -> None:
        self.assertEqual(
            web_server.parse_skill_instruction_route("/api/skills/meeting-minutes/instructions"),
            "meeting-minutes",
        )
        self.assertIsNone(web_server.parse_skill_instruction_route("/api/skills/meeting-minutes"))
        self.assertIsNone(web_server.parse_skill_instruction_route("/api/skills/settings"))

    def test_meeting_minutes_skill_instruction_path_is_inside_the_skill_root(self) -> None:
        path = web_server.skill_instruction_path("meeting-minutes")
        self.assertEqual(path.name, "SKILL.md")
        self.assertTrue(path.is_relative_to(WORKSPACE / "meeting_audio_minutes" / "skills"))


if __name__ == "__main__":
    unittest.main()
