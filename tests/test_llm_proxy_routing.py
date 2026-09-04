from __future__ import annotations

import unittest
import urllib.request
from unittest.mock import Mock, patch

from work_agent_core.config import ModelProfile
from work_agent_core.llm import (
    OpenAICompatibleClient,
    is_local_or_private_endpoint,
    recovery_request_timeout_seconds,
    should_prefer_direct_connection,
    stream_start_timeout_seconds,
)


def profile(*, name: str, base_url: str, model: str) -> ModelProfile:
    return ModelProfile(
        name=name,
        provider="deepseek" if "deepseek" in model else "openai-compatible",
        base_url=base_url,
        model=model,
        api_key_env="TEST_API_KEY",
    )


class LLMProxyRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.request = urllib.request.Request("https://api.deepseek.com/chat/completions")

    def test_official_deepseek_and_private_endpoints_prefer_direct_connection(self) -> None:
        self.assertTrue(
            should_prefer_direct_connection(
                profile(
                    name="deepseek-v4-pro",
                    base_url="https://api.deepseek.com",
                    model="deepseek-v4-pro",
                )
            )
        )
        self.assertFalse(
            should_prefer_direct_connection(
                profile(
                    name="deepseek-relay",
                    base_url="https://relay.example.com/v1",
                    model="deepseek-v4-pro",
                )
            )
        )
        for base_url in (
            "http://100.86.69.1:1234/v1",
            "http://192.168.1.20:1234/v1",
            "http://127.0.0.1:1234/v1",
            "http://desktop.tailnet.ts.net:1234/v1",
        ):
            with self.subTest(base_url=base_url):
                self.assertTrue(
                    should_prefer_direct_connection(
                        profile(name="local-qwen", base_url=base_url, model="qwen3.8-27b")
                    )
                )

    def test_local_model_recovery_keeps_full_prefill_budget(self) -> None:
        current = ModelProfile(
            name="lmstudio-qwen3.8-27b",
            provider="lm-studio",
            base_url="http://100.86.69.1:1234/v1",
            model="qwen3.8-27b",
            api_key_env="TEST_API_KEY",
            timeout_seconds=300,
            stream_idle_timeout_seconds=30,
        )

        self.assertTrue(is_local_or_private_endpoint(current))
        self.assertEqual(recovery_request_timeout_seconds(current), 300)

    def test_cloud_model_recovery_remains_bounded(self) -> None:
        current = ModelProfile(
            name="cloud-model",
            provider="openai-compatible",
            base_url="https://api.example.com/v1",
            model="cloud-model",
            api_key_env="TEST_API_KEY",
            timeout_seconds=300,
        )

        self.assertFalse(is_local_or_private_endpoint(current))
        self.assertEqual(recovery_request_timeout_seconds(current), 60)

    def test_stream_idle_override_never_shortens_prefill_budget(self) -> None:
        current = ModelProfile(
            name="local-model",
            provider="lm-studio",
            base_url="http://127.0.0.1:1234/v1",
            model="local-model",
            api_key_env="TEST_API_KEY",
            timeout_seconds=300,
            stream_idle_timeout_seconds=75,
        )

        self.assertEqual(stream_start_timeout_seconds(current), 300)

    def test_official_deepseek_ignores_configured_system_proxy(self) -> None:
        client = OpenAICompatibleClient()
        direct_response = object()
        client._direct_opener = Mock()
        client._direct_opener.open.return_value = direct_response
        current = profile(
            name="deepseek-v4-pro",
            base_url="https://api.deepseek.com",
            model="deepseek-v4-pro",
        )

        with patch("work_agent_core.llm.urllib.request.urlopen") as urlopen:
            response = client._open_request(self.request, profile=current, timeout=12)

        self.assertIs(response, direct_response)
        client._direct_opener.open.assert_called_once_with(self.request, timeout=12)
        urlopen.assert_not_called()

    def test_tailscale_endpoint_ignores_configured_system_proxy(self) -> None:
        client = OpenAICompatibleClient()
        direct_response = object()
        client._direct_opener = Mock()
        client._direct_opener.open.return_value = direct_response
        current = profile(
            name="lmstudio-qwen3.8-27b",
            base_url="http://100.86.69.1:1234/v1",
            model="qwen3.8-27b",
        )
        request = urllib.request.Request("http://100.86.69.1:1234/v1/chat/completions")

        with patch("work_agent_core.llm.urllib.request.urlopen") as urlopen:
            response = client._open_request(request, profile=current, timeout=12)

        self.assertIs(response, direct_response)
        client._direct_opener.open.assert_called_once_with(request, timeout=12)
        urlopen.assert_not_called()

    def test_official_deepseek_never_falls_back_to_system_proxy(self) -> None:
        client = OpenAICompatibleClient()
        client._direct_opener = Mock()
        client._direct_opener.open.side_effect = OSError("direct route unavailable")
        current = profile(
            name="deepseek-v4-flash",
            base_url="https://api.deepseek.com/v1",
            model="deepseek-v4-flash",
        )

        with patch("work_agent_core.llm.urllib.request.urlopen") as urlopen:
            with self.assertRaisesRegex(OSError, "direct route unavailable"):
                client._open_request(self.request, profile=current, timeout=12)

        client._direct_opener.open.assert_called_once_with(self.request, timeout=12)
        urlopen.assert_not_called()

    def test_other_provider_keeps_system_proxy_route(self) -> None:
        client = OpenAICompatibleClient()
        client._direct_opener = Mock()
        proxy_response = object()
        current = profile(
            name="luna",
            base_url="https://api.example.com/v1",
            model="gpt-5.6-luna",
        )

        with patch("work_agent_core.llm.urllib.request.urlopen", return_value=proxy_response) as urlopen:
            response = client._open_request(self.request, profile=current, timeout=12)

        self.assertIs(response, proxy_response)
        client._direct_opener.open.assert_not_called()
        urlopen.assert_called_once_with(self.request, timeout=12)


if __name__ == "__main__":
    unittest.main()
