from __future__ import annotations

import unittest

from work_agent_core.config import ModelProfile
from work_agent_core.llm import apply_reasoning_controls, extract_reasoning_delta


def profile(*, provider: str, model: str, base_url: str) -> ModelProfile:
    return ModelProfile(
        name=model,
        provider=provider,
        base_url=base_url,
        model=model,
        api_key_env="TEST_API_KEY",
        temperature=0.6,
    )


class ReasoningControlTests(unittest.TestCase):
    def test_lmstudio_qwen38_levels_control_thinking_and_sampling(self) -> None:
        current = profile(
            provider="lm-studio",
            model="qwen3.8-27b",
            base_url="http://100.86.69.1:1234/v1",
        )

        expected = {
            "light": (True, "low", 1.0, 0.95, 0.0),
            "medium": (True, "low", 1.0, 0.95, 0.0),
            "high": (True, "medium", 1.0, 0.95, 0.0),
            "very_high": (True, "xhigh", 1.0, 0.95, 0.0),
        }
        for ui_level, (
            thinking,
            api_effort,
            temperature,
            top_p,
            presence_penalty,
        ) in expected.items():
            with self.subTest(ui_level=ui_level):
                payload = {"temperature": 0.6}
                apply_reasoning_controls(payload, profile=current, reasoning_effort=ui_level)
                self.assertEqual(
                    payload["chat_template_kwargs"],
                    {"enable_thinking": thinking, "preserve_thinking": True},
                )
                self.assertEqual(payload.get("reasoning_effort"), api_effort)
                self.assertEqual(payload["temperature"], temperature)
                self.assertEqual(payload["top_p"], top_p)
                self.assertEqual(payload["top_k"], 20)
                self.assertEqual(payload["min_p"], 0.0)
                self.assertEqual(payload["presence_penalty"], presence_penalty)
                self.assertEqual(payload["repeat_penalty"], 1.0)

    def test_openai_levels_map_to_reasoning_effort(self) -> None:
        current = profile(
            provider="openai-compatible",
            model="gpt-5.6-luna",
            base_url="https://example.com/v1",
        )
        for ui_level, api_level in {
            "light": "low",
            "medium": "medium",
            "high": "high",
            "very_high": "max",
        }.items():
            payload = {"temperature": 0.6}
            apply_reasoning_controls(payload, profile=current, reasoning_effort=ui_level)
            self.assertEqual(payload["reasoning_effort"], api_level)
            self.assertEqual(payload["temperature"], 0.6)

    def test_glm5_light_keeps_thinking_enabled_for_thinking_only_provider(self) -> None:
        current = profile(
            provider="opencode-go",
            model="glm-5.2",
            base_url="https://opencode.ai/zen/go/v1",
        )

        light_payload = {"temperature": 0.6}
        apply_reasoning_controls(
            light_payload,
            profile=current,
            reasoning_effort="light",
        )
        self.assertEqual(light_payload["thinking"], {"type": "enabled"})

        high_payload = {"temperature": 0.6}
        apply_reasoning_controls(
            high_payload,
            profile=current,
            reasoning_effort="high",
        )
        self.assertEqual(high_payload["thinking"], {"type": "enabled"})

    def test_glm4_light_can_still_disable_thinking(self) -> None:
        current = profile(
            provider="openai-compatible",
            model="glm-4.7",
            base_url="https://example.com/v1",
        )
        payload = {"temperature": 0.6}
        apply_reasoning_controls(payload, profile=current, reasoning_effort="light")
        self.assertEqual(payload["thinking"], {"type": "disabled"})

    def test_deepseek_light_disables_thinking_and_keeps_temperature(self) -> None:
        current = profile(
            provider="deepseek",
            model="deepseek-v4-pro",
            base_url="https://api.deepseek.com",
        )
        payload = {"temperature": 0.6}
        apply_reasoning_controls(payload, profile=current, reasoning_effort="light")
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertNotIn("reasoning_effort", payload)
        self.assertEqual(payload["temperature"], 0.6)

    def test_deepseek_thinking_uses_high_or_max_without_temperature(self) -> None:
        current = profile(
            provider="deepseek",
            model="deepseek-v4-pro",
            base_url="https://api.deepseek.com",
        )
        for ui_level, api_level in {
            "medium": "high",
            "high": "high",
            "very_high": "max",
        }.items():
            payload = {"temperature": 0.6}
            apply_reasoning_controls(payload, profile=current, reasoning_effort=ui_level)
            self.assertEqual(payload["thinking"], {"type": "enabled"})
            self.assertEqual(payload["reasoning_effort"], api_level)
            self.assertNotIn("temperature", payload)

    def test_minimax_m3_uses_split_adaptive_thinking(self) -> None:
        current = profile(
            provider="openai-compatible",
            model="MiniMaxAI/MiniMax-M3",
            base_url="https://api.gmi-serving.com/v1",
        )
        payload = {"temperature": 0.6}
        apply_reasoning_controls(payload, profile=current, reasoning_effort="high")
        self.assertTrue(payload["reasoning_split"])
        self.assertEqual(payload["thinking"], {"type": "adaptive"})
        self.assertNotIn("reasoning_effort", payload)

        light_payload = {"temperature": 0.6}
        apply_reasoning_controls(light_payload, profile=current, reasoning_effort="light")
        self.assertEqual(light_payload["thinking"], {"type": "disabled"})

    def test_minimax_reasoning_details_are_extracted(self) -> None:
        delta = {
            "reasoning_details": [
                {"type": "reasoning.text", "text": "先分析"},
                {"type": "reasoning.text", "text": "再回答"},
            ]
        }
        self.assertEqual(extract_reasoning_delta(delta, {}), "先分析再回答")


if __name__ == "__main__":
    unittest.main()
