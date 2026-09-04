from __future__ import annotations

from types import SimpleNamespace
import json
import unittest
from unittest.mock import patch

from work_agent_core.config import ModelProfile
from work_agent_core.llm import LLMResponse
from work_agent_core.memory import ContextCompactionError
from work_agent_core.session_runtime import ConversationRuntime
from work_agent_core.react import ReActAgent
from work_agent_core.tool_bus import ToolBus


class _PlanThenFinalClient:
    def __init__(self) -> None:
        self.calls = 0

    def chat_tools_stream(self, *_args, **_kwargs) -> LLMResponse:
        self.calls += 1
        if self.calls <= 2:
            completed = self.calls == 2
            plan = [
                {"step": "核对现状", "status": "completed" if completed else "in_progress"},
                {"step": "完成修改并验证", "status": "completed" if completed else "pending"},
            ]
            return LLMResponse(
                content="",
                raw={
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [{
                                "id": f"plan-{self.calls}",
                                "type": "function",
                                "function": {
                                    "name": "update_plan",
                                    "arguments": json.dumps({"plan": plan}, ensure_ascii=False),
                                },
                            }],
                        }
                    }]
                },
            )
        return LLMResponse(
            content="完成",
            raw={"choices": [{"message": {"role": "assistant", "content": "完成"}}]},
        )


class _CheckpointClient:
    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages, *, profile, max_tokens, reasoning_effort=None):
        self.calls += 1
        return SimpleNamespace(
            content=(
                "## 当前目标与完成条件\n- 完成复杂改造\n"
                "## 当前计划与进度\n- 已核对代码\n"
                "## 已走过的实施路径（按顺序）\n- 读取后修改\n"
                "## 已修改内容与关键证据\n- /tmp/demo.py\n"
                "## 错误、风险与待确认事项\n- 无\n"
                "## 下一步准确动作\n- 运行测试"
            )
        )


class _EmptyCheckpointClient:
    def chat(self, _messages, *, profile, max_tokens, reasoning_effort=None):
        return SimpleNamespace(
            content="",
            raw={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": "",
                            "reasoning_content": "只返回了推理通道",
                        },
                    }
                ]
            },
        )


class PlanExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = ModelProfile(
            name="test",
            provider="openai-compatible",
            base_url="http://example.invalid",
            model="test-model",
            api_key_env="TEST_API_KEY",
        )

    def test_model_can_create_and_complete_living_plan_inside_react(self) -> None:
        saved: list[list[dict[str, str]]] = []
        agent = ReActAgent(
            client=_PlanThenFinalClient(),
            profile=self.profile,
            tools=ToolBus(),
            plan_update_callback=lambda plan, _explanation: saved.append(plan),
        )
        messages = [{"role": "user", "content": "完成复杂改造"}]

        events = list(agent.iter_message_events(messages))

        plan_events = [event for event in events if event.get("activity_type") == "plan"]
        self.assertEqual(len(plan_events), 2)
        self.assertEqual(plan_events[-1]["plan_completed"], 2)
        self.assertEqual(saved[-1][1]["status"], "completed")
        self.assertEqual(events[-1]["event"], "final")

    def test_running_react_path_is_model_compacted_to_checkpoint(self) -> None:
        client = _CheckpointClient()
        agent = ReActAgent(client=client, profile=self.profile, tools=ToolBus())
        runtime = ConversationRuntime.from_messages([
            {"role": "user", "content": "完成复杂改造"},
            {"role": "assistant", "content": "我先核对代码", "tool_calls": [{
                "id": "call-1",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path":"/tmp/demo.py"}'},
            }]},
            {"role": "tool", "tool_call_id": "call-1", "name": "read_file", "content": "x = 1"},
        ])
        messages = agent._request_messages(runtime)

        with patch("work_agent_core.react.profile_context_trigger_tokens", return_value=1):
            event = agent._maybe_compact_active_runtime(runtime, messages, step=2)

        self.assertIsNotNone(event)
        self.assertEqual(event["activity_type"], "runtime_summary")
        self.assertEqual(client.calls, 1)

        # The model view folds to the prompt plus one checkpoint...
        compacted = agent._request_messages(runtime)
        self.assertEqual([item["role"] for item in compacted], ["system", "user", "assistant"])
        self.assertIn("高保真检查点", compacted[-1]["content"])
        self.assertIn("运行测试", compacted[-1]["content"])

        # ...while the raw implementation path stays readable for a human.
        transcript = runtime.log.derive_transcript()
        self.assertEqual([item["role"] for item in transcript], ["user", "assistant", "tool"])
        self.assertEqual(transcript[-1]["content"], "x = 1")

    def test_running_react_compacts_when_tool_results_cross_the_workset_budget(self) -> None:
        client = _CheckpointClient()
        agent = ReActAgent(client=client, profile=self.profile, tools=ToolBus())
        runtime = ConversationRuntime.from_messages([
            {"role": "user", "content": "完成复杂改造"},
            {"role": "assistant", "content": "我先核对代码", "tool_calls": [{
                "id": "call-1",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path":"/tmp/demo.py"}'},
            }]},
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "read_file",
                "content": "x = 1\n" * 40,
            },
        ])
        messages = agent._request_messages(runtime)

        with (
            patch("work_agent_core.react.profile_context_trigger_tokens", return_value=1_000_000),
            patch("work_agent_core.react.ACTIVE_REACT_CHECKPOINT_SERIALIZED_BYTES", 1_000_000),
            patch("work_agent_core.react.ACTIVE_REACT_CHECKPOINT_TOOL_RESULT_CHARS", 20),
        ):
            event = agent._maybe_compact_active_runtime(runtime, messages, step=2)

        self.assertIsNotNone(event)
        self.assertEqual(event["activity_type"], "runtime_summary")
        self.assertEqual(client.calls, 1)

    def test_running_react_stops_when_checkpoint_model_returns_no_content(self) -> None:
        agent = ReActAgent(client=_EmptyCheckpointClient(), profile=self.profile, tools=ToolBus())
        runtime = ConversationRuntime.from_messages([
            {"role": "user", "content": "完成复杂改造"},
            {"role": "assistant", "content": "已经读取材料"},
            {"role": "tool", "tool_call_id": "call-1", "name": "read_file", "content": "证据"},
        ])
        original_transcript = runtime.log.derive_transcript()

        with patch("work_agent_core.react.profile_context_trigger_tokens", return_value=1):
            with self.assertRaises(ContextCompactionError) as captured:
                list(agent.iter_message_events(runtime))

        self.assertIn("reasoning_chars", str(captured.exception))
        self.assertEqual(runtime.log.derive_transcript(), original_transcript)
        self.assertEqual(list(runtime.log.iter_type("compaction/replacement")), [])

if __name__ == "__main__":
    unittest.main()
