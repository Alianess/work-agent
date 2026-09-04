from __future__ import annotations

import unittest
from pathlib import Path

from work_agent_core.config import ModelProfile
from work_agent_core.react import ReActAgent
from work_agent_core.tool_bus import ToolBus


class PromptTerminationContractTests(unittest.TestCase):
    @staticmethod
    def _prompt() -> str:
        profile = ModelProfile(
            name="prompt-contract-test",
            provider="openai-compatible",
            base_url="https://example.invalid/v1",
            model="test-model",
            api_key_env="UNUSED",
        )
        return ReActAgent(
            client=object(),  # type: ignore[arg-type]
            profile=profile,
            tools=ToolBus(),
        ).system_prompt

    def test_termination_is_carried_by_the_loop_not_by_the_prompt(self) -> None:
        """终止语义归循环管，不再靠提示词反复叮嘱。

        原来这里有 500 多字在恐吓模型"不许只输出 content 就收尾"。那是在替
        缺失的 steering / follow-up 队列付租金：模型提前收尾时无法接续，只能
        用文字预防。队列补上之后（见 tests/test_turn_steering.py），人补一句
        就能继续，这些字就该消失。
        """

        prompt = self._prompt()

        self.assertNotIn("必须牢记 ReAct 的终止语义", prompt)
        self.assertNotIn("就不得只输出 content", prompt)
        self.assertNotIn("必须在同一条 assistant 消息中同时发起", prompt)
        self.assertNotIn("就绝对不得输出 content-only 最终答复", prompt)
        self.assertNotIn("本轮尚未成功执行写入/生成类工具并完成相应核验", prompt)

    def test_prompt_keeps_only_what_has_nowhere_closer_to_live(self) -> None:
        """留在提示词里的，是没有更近的家可回的规则。"""

        prompt = self._prompt()

        self.assertIn("未来时计划冒充交付", prompt)
        self.assertIn("不要编造工具结果", prompt)
        self.assertIn("原生 tool calling", prompt)
        # 这条不是文风偏好：UI 的"实施路径"面板全靠它写出来。
        self.assertIn("先写一小段自然语言工作说明", prompt)

    def test_terminal_tool_description_only_contains_calling_guidance(self) -> None:
        """Runtime 强制的终端规则不再作为每轮自然语言提示词。"""

        from work_agent_core.shell_tools import register_shell_tools
        from work_agent_core.tools import ToolRegistry

        registry = ToolRegistry()
        register_shell_tools(registry, Path.cwd())
        description = registry.get("shell_exec").description

        self.assertLess(len(description), 200)
        self.assertIn("dedicated core or skill tool", description)
        self.assertNotIn("确认", self._prompt())
        for runtime_rule in ("approval_required", "Seatbelt", "risk_category", "stdout", "redirection"):
            self.assertNotIn(runtime_rule, description)

    def test_plan_rules_moved_onto_the_plan_tool(self) -> None:
        profile = ModelProfile(
            name="prompt-contract-test",
            provider="openai-compatible",
            base_url="https://example.invalid/v1",
            model="test-model",
            api_key_env="UNUSED",
        )
        agent = ReActAgent(
            client=object(),  # type: ignore[arg-type]
            profile=profile,
            tools=ToolBus(),
        )
        schema = next(
            item
            for item in agent._tool_schemas()
            if item["function"]["name"] == "update_plan"
        )
        description = schema["function"]["description"]
        parameters = schema["function"]["parameters"]

        self.assertIn("2-7 outcome-shaped steps", description)
        self.assertIn("at most one step", description)
        self.assertLess(len(description), 220)
        self.assertEqual(parameters["properties"]["plan"]["minItems"], 2)
        self.assertEqual(parameters["properties"]["plan"]["maxItems"], 7)
        self.assertNotIn("计划执行规则", agent.system_prompt)

    def test_file_tool_schemas_only_describe_their_own_operation(self) -> None:
        from work_agent_core.tools import ToolRegistry, register_file_tools

        registry = ToolRegistry()
        register_file_tools(registry, Path.cwd())

        read_description = registry.get("read_file").description
        write_description = registry.get("write_text_file").description
        edit_description = registry.get("edit_text_file").description
        self.assertLess(len(read_description), 240)
        self.assertLess(len(write_description), 100)
        self.assertLess(len(edit_description), 160)
        self.assertIn("offset/max_chars", read_description)
        self.assertIn("fully replace", write_description)
        self.assertIn("exact replacement", edit_description)
        self.assertNotIn("edit_text_file", write_description)
        self.assertNotIn("write_text_file", edit_description)
        self.assertNotIn("文件使用规则", self._prompt())

    def test_workspace_rules_moved_into_the_workspace_file(self) -> None:
        """环境约定跟着目录走，换个工作区就不该带着上一个项目的 Python 布局。"""

        from work_agent_core.react import read_workspace_context

        prompt_without_workspace = self._prompt()
        self.assertNotIn(".venv_deepfilter", prompt_without_workspace)
        self.assertNotIn("runtime_env.sh", prompt_without_workspace)

        context = read_workspace_context(Path.cwd())
        self.assertIn(".venv", context)
        self.assertIn("runtime_env.sh", context)
        self.assertIn("--user", context)

    def test_todo_semantics_live_in_the_skill_not_the_prompt(self) -> None:
        """常驻目录只有技能简介，不再附加更长的 when_to_use。"""

        from work_agent_core import web_server

        self.assertNotIn("个人待办语义规则", self._prompt())
        catalog = web_server.render_chat_skill_catalog()
        self.assertIn("- apple-schedule：", catalog)
        self.assertNotIn("任务明确匹配某技能时", catalog)
        self.assertNotIn("无法判断对应技能时", catalog)
        self.assertIn("sys_skill.list", self._prompt())
        self.assertNotIn("询问“我有什么待办”", catalog)

    def test_recall_guidance_is_lean_and_zero_compaction_state_is_omitted(self) -> None:
        from work_agent_core import web_server

        guidance = web_server.recall_guidance_system_context()
        self.assertIn("当前上下文不足时，先使用 recall", guidance)
        self.assertIn("项目会话优先 scope=project", guidance)
        self.assertLess(len(guidance), 180)
        for implementation_detail in ("BM25", "RRF", "memory_results", "runtime messages"):
            self.assertNotIn(implementation_detail, guidance)
        self.assertEqual(web_server.recall_runtime_state_context(0), "")
        self.assertIn("7 条较早消息", web_server.recall_runtime_state_context(7))

    def test_skills_are_in_the_stable_prefix_and_time_is_not(self) -> None:
        from unittest.mock import patch
        from work_agent_core import web_server

        with (
            patch.object(web_server, "agent_system_context", return_value="固定资料"),
            patch.object(web_server, "render_chat_skill_catalog", return_value="Skills固定"),
        ):
            stable = web_server.agent_stable_system_context()

        self.assertEqual(
            stable,
            "固定资料\n\n" + web_server.recall_guidance_system_context() + "\n\nSkills固定",
        )
        self.assertNotIn("系统当前时间", stable)
        self.assertTrue(
            web_server.agent_turn_runtime_context().endswith(web_server.agent_turn_time_context())
        )

    def test_prompt_stays_within_its_budget(self) -> None:
        """提示词是每一次请求都要付的税，给它一个会响的上限。"""

        self.assertLess(len(self._prompt()), 3200)

    def test_apple_schedule_skill_distinguishes_reminders_from_work_tasks(self) -> None:
        skill_text = (
            Path(__file__).resolve().parents[1]
            / "work_agent_skills"
            / "apple-schedule"
            / "SKILL.md"
        ).read_text(encoding="utf-8")

        self.assertIn("待办事项只指 Apple「提醒事项」", skill_text)
        self.assertIn("include_events=false", skill_text)
        self.assertIn("create_apple_reminder", skill_text)
        self.assertIn("不能新增日历事件", skill_text)

    def test_meeting_minutes_skill_has_artifact_completion_gate(self) -> None:
        skill_text = (
            Path(__file__).resolve().parents[1]
            / "meeting_audio_minutes"
            / "skills"
            / "meeting-minutes"
            / "SKILL.md"
        ).read_text(encoding="utf-8")

        self.assertIn("## Completion gate", skill_text)
        self.assertIn("They are never a completed result", skill_text)
        self.assertIn("canonical_outputs", skill_text)
        self.assertIn("the Web file viewer generates the preview automatically", skill_text)
        self.assertIn("Do not wait for or simulate human page-layout acceptance", skill_text)
        self.assertNotIn("rendered-page visual QA", skill_text)
        self.assertIn("existing path as `canonical_outputs.asr`", skill_text)
        self.assertIn("Never manually replay a large ASR", skill_text)
        self.assertIn("The ASR path may point to the existing completed transcript", skill_text)
        self.assertNotIn("This completes the meeting skill's content responsibility", skill_text)
        self.assertNotIn("copy its complete content into this canonical archive file", skill_text)


if __name__ == "__main__":
    unittest.main()
