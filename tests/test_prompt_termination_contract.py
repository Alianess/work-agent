from __future__ import annotations

import unittest
from pathlib import Path

from work_agent_core.config import ModelProfile
from work_agent_core.react import ReActAgent
from work_agent_core.tool_bus import ToolBus


class PromptTerminationContractTests(unittest.TestCase):
    def test_system_prompt_explains_content_only_ends_react(self) -> None:
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

        prompt = agent.system_prompt
        self.assertIn("只输出 assistant content 而不输出 tool_calls", prompt)
        self.assertIn("立即视为最终答复并结束整个 ReAct", prompt)
        self.assertIn("必须在同一条 assistant 消息中同时发起", prompt)
        self.assertIn("未来时计划冒充交付", prompt)
        self.assertIn("不得因为预计某个后续动作可能需要权限", prompt)
        self.assertIn("审批由系统审批卡处理", prompt)
        self.assertIn("不得让用户手工输入许可", prompt)
        self.assertIn("本轮尚未成功执行写入/生成类工具并完成相应核验", prompt)
        self.assertIn("查看工具参数、环境预检", prompt)
        self.assertIn("已经存在且已核验的产物路径", prompt)
        self.assertIn("个人待办语义规则", prompt)
        self.assertIn("待办事项只指 Apple「提醒事项」", prompt)
        self.assertIn("不得从项目、日报、会议纪要、历史对话或工作上下文推测任务", prompt)

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
