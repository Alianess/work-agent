from __future__ import annotations

import unittest

from work_agent_core.config import ModelProfile
from work_agent_core.llm import apply_reasoning_controls


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


if __name__ == "__main__":
    unittest.main()
