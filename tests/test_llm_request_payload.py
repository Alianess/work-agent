from __future__ import annotations

import unittest

from work_agent_core.config import ModelProfile
from work_agent_core.llm import build_chat_tools_payload


class LLMRequestPayloadTests(unittest.TestCase):
    def test_qwen_payload_builder_is_shared_by_analysis_and_send(self) -> None:
        profile = ModelProfile(
            name="lmstudio-qwen3.8-27b",
            provider="lm-studio",
            base_url="http://100.86.69.1:1234/v1",
            model="qwen3.8-27b",
            api_key_env="TEST_KEY",
            temperature=1.0,
            max_tokens=32768,
        )
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "read",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        payload = build_chat_tools_payload(
            [{"role": "user", "content": "你好"}],
            profile=profile,
            tools=tools,
            tool_choice="auto",
            reasoning_effort="very_high",
        )

        self.assertEqual(payload["messages"], [{"role": "user", "content": "你好"}])
        self.assertEqual(payload["tools"], tools)
        self.assertEqual(payload["reasoning_effort"], "xhigh")
        self.assertEqual(payload["max_tokens"], 32768)
        self.assertEqual(payload["stream_options"], {"include_usage": True})

    def test_qwen_preserves_complete_reasoning_history(self) -> None:
        profile = ModelProfile(
            name="lmstudio-qwen3.8-27b",
            provider="lm-studio",
            base_url="http://100.86.69.1:1234/v1",
            model="qwen3.8-27b",
            api_key_env="TEST_KEY",
        )
        payload = build_chat_tools_payload(
            [
                {"role": "user", "content": "你好"},
                {
                    "role": "assistant",
                    "content": "你好。",
                    "reasoning_content": "respond in Chinese",
                },
                {"role": "user", "content": "你是谁"},
            ],
            profile=profile,
            reasoning_effort="medium",
        )

        self.assertTrue(payload["chat_template_kwargs"]["preserve_thinking"])
        self.assertEqual(payload["messages"][1]["reasoning_content"], "respond in Chinese")
        self.assertEqual(payload["messages"][1]["reasoning"], "respond in Chinese")

    def test_qwen_disables_preservation_for_legacy_incomplete_history(self) -> None:
        profile = ModelProfile(
            name="lmstudio-qwen3.8-27b",
            provider="lm-studio",
            base_url="http://100.86.69.1:1234/v1",
            model="qwen3.8-27b",
            api_key_env="TEST_KEY",
        )
        payload = build_chat_tools_payload(
            [
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "你好。"},
                {"role": "user", "content": "你是谁"},
            ],
            profile=profile,
            reasoning_effort="medium",
        )

        self.assertFalse(payload["chat_template_kwargs"]["preserve_thinking"])

    def test_qwen_legacy_guard_does_not_depend_on_reasoning_selector(self) -> None:
        profile = ModelProfile(
            name="lmstudio-qwen3.8-27b",
            provider="lm-studio",
            base_url="http://100.86.69.1:1234/v1",
            model="qwen3.8-27b",
            api_key_env="TEST_KEY",
        )
        payload = build_chat_tools_payload(
            [
                {"role": "assistant", "content": "legacy answer"},
                {"role": "user", "content": "continue"},
            ],
            profile=profile,
            reasoning_effort=None,
        )

        self.assertEqual(
            payload["chat_template_kwargs"],
            {"enable_thinking": True, "preserve_thinking": False},
        )


if __name__ == "__main__":
    unittest.main()
