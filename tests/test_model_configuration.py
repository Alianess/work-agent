from __future__ import annotations

import os
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from work_agent_core import web_server
from work_agent_core.cli import (
    delete_model_profile,
    update_model_profile,
    update_model_profile_api_key_env,
)
from work_agent_core.config import ModelProfile, api_key_env_for_profile, delete_env_value, save_env_value
from work_agent_core.llm import chat_completions_endpoint


class FakeResponse:
    def __init__(self, payload: dict, *, status: int = 200) -> None:
        self.payload = payload
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class ModelConfigurationTests(unittest.TestCase):
    def test_model_profile_vision_capability_is_explicit_and_legacy_safe(self) -> None:
        deepseek = ModelProfile.from_dict({
            "name": "deepseek-v4-flash",
            "provider": "deepseek",
            "base_url": "https://api.deepseek.com",
            "model": "deepseek-v4-flash",
            "api_key_env": "DEEPSEEK_API_KEY",
        })
        vision = ModelProfile.from_dict({
            "name": "vision-proxy",
            "provider": "openai-compatible",
            "base_url": "https://api.example.com/v1",
            "model": "gpt-5.6-luna",
            "api_key_env": "VISION_KEY",
            "supports_vision": True,
        })

        self.assertFalse(deepseek.supports_vision)
        self.assertTrue(vision.supports_vision)

    def test_profile_key_names_are_stable_and_isolated(self) -> None:
        first = api_key_env_for_profile("gpt-5.6-luna")
        second = api_key_env_for_profile("gpt_5.6_luna")

        self.assertEqual(first, api_key_env_for_profile("gpt-5.6-luna"))
        self.assertNotEqual(first, second)
        self.assertTrue(first.startswith("WORK_AGENT_MODEL_GPT_5_6_LUNA_"))

    def test_save_env_value_preserves_other_entries_and_updates_process_env(self) -> None:
        key = "WORK_AGENT_MODEL_TEST_API_KEY"
        previous = os.environ.get(key)
        try:
            with tempfile.TemporaryDirectory() as directory:
                env_path = Path(directory) / ".env"
                env_path.write_text("EXISTING=value\n\n# comment\n", encoding="utf-8")

                save_env_value(env_path, key, "secret-value")
                save_env_value(env_path, key, "replacement-value")

                content = env_path.read_text(encoding="utf-8")
                self.assertIn("EXISTING=value", content)
                self.assertIn("# comment", content)
                self.assertEqual(content.count(f"{key}="), 1)
                self.assertIn(f"{key}=replacement-value", content)
                self.assertEqual(os.environ[key], "replacement-value")
                self.assertEqual(env_path.stat().st_mode & 0o777, 0o600)
        finally:
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous

    def test_chat_endpoint_accepts_base_or_full_url(self) -> None:
        self.assertEqual(
            chat_completions_endpoint("https://api.example.com/v1"),
            "https://api.example.com/v1/chat/completions",
        )

    def test_existing_profile_can_migrate_to_an_isolated_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "model_profiles.json"
            config_path.write_text(
                json.dumps(
                    {
                        "default_profile": "existing",
                        "profiles": [
                            {
                                "name": "existing",
                                "provider": "openai-compatible",
                                "base_url": "https://api.example.com/v1",
                                "model": "example-model",
                                "api_key_env": "OPENAI_API_KEY",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            update_model_profile_api_key_env(
                config_path,
                "existing",
                "WORK_AGENT_MODEL_EXISTING_API_KEY",
            )

            payload = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["profiles"][0]["api_key_env"],
                "WORK_AGENT_MODEL_EXISTING_API_KEY",
            )
        self.assertEqual(
            chat_completions_endpoint("https://api.example.com/v1/chat/completions/"),
            "https://api.example.com/v1/chat/completions",
        )

    def test_profile_can_be_updated_and_non_default_profile_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "model_profiles.json"
            config_path.write_text(
                json.dumps(
                    {
                        "default_profile": "primary",
                        "profiles": [
                            {
                                "name": "primary",
                                "provider": "openai-compatible",
                                "base_url": "https://api.example.com/v1",
                                "model": "model-a",
                                "api_key_env": "PRIMARY_KEY",
                            },
                            {
                                "name": "secondary",
                                "provider": "openai-compatible",
                                "base_url": "https://old.example.com/v1",
                                "model": "old-model",
                                "api_key_env": "SECONDARY_KEY",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            update_model_profile(
                config_path,
                "secondary",
                {
                    "provider": "openai-compatible",
                    "base_url": "https://new.example.com/v1",
                    "model": "new-model",
                    "api_key_env": "SECONDARY_KEY",
                    "temperature": 0.2,
                    "max_tokens": 4096,
                    "timeout_seconds": 60,
                },
            )
            updated = json.loads(config_path.read_text(encoding="utf-8"))
            secondary = next(item for item in updated["profiles"] if item["name"] == "secondary")
            self.assertEqual(secondary["base_url"], "https://new.example.com/v1")
            self.assertEqual(secondary["model"], "new-model")

            removed = delete_model_profile(config_path, "secondary")
            self.assertEqual(removed["name"], "secondary")
            remaining = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual([item["name"] for item in remaining["profiles"]], ["primary"])
            with self.assertRaisesRegex(ValueError, "当前正在使用"):
                delete_model_profile(config_path, "primary")

    def test_delete_env_value_removes_only_selected_secret(self) -> None:
        key = "WORK_AGENT_MODEL_DELETE_ME_API_KEY"
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text(f"KEEP=value\n{key}=secret\n", encoding="utf-8")
            os.environ[key] = "secret"

            delete_env_value(env_path, key)

            self.assertEqual(env_path.read_text(encoding="utf-8"), "KEEP=value\n")
            self.assertNotIn(key, os.environ)

    def test_model_connection_and_discovery_use_compatible_endpoints(self) -> None:
        connection_response = FakeResponse(
            {"choices": [{"message": {"role": "assistant", "content": "OK"}}]}
        )
        with patch.object(web_server.urllib.request, "urlopen", return_value=connection_response) as urlopen:
            result = web_server.test_model_connection_payload(
                {
                    "base_url": "https://api.example.com/v1/chat/completions",
                    "model": "demo-model",
                    "api_key": "secret",
                    "timeout_seconds": 30,
                }
            )
        self.assertTrue(result["ok"])
        self.assertEqual(result["endpoint"], "https://api.example.com/v1/chat/completions")
        self.assertEqual(urlopen.call_args.args[0].full_url, result["endpoint"])

        models_response = FakeResponse({"data": [{"id": "model-b"}, {"id": "model-a"}]})
        with patch.object(web_server.urllib.request, "urlopen", return_value=models_response):
            discovered = web_server.discover_models_payload(
                {
                    "base_url": "https://api.example.com/v1/chat/completions",
                    "model": "demo-model",
                    "api_key": "secret",
                }
            )
        self.assertEqual(discovered["endpoint"], "https://api.example.com/v1/models")
        self.assertEqual(discovered["models"], ["model-a", "model-b"])

    def test_official_deepseek_connection_test_uses_direct_opener(self) -> None:
        connection_response = FakeResponse(
            {"choices": [{"message": {"role": "assistant", "content": "OK"}}]}
        )
        with (
            patch.object(
                web_server.OpenAICompatibleClient,
                "_open_request",
                return_value=connection_response,
            ) as open_request,
            patch.object(web_server.urllib.request, "urlopen") as urlopen,
        ):
            result = web_server.test_model_connection_payload(
                {
                    "base_url": "https://api.deepseek.com",
                    "model": "deepseek-v4-pro",
                    "api_key": "test-key",
                    "timeout_seconds": 12,
                }
            )

        self.assertTrue(result["ok"])
        self.assertEqual(open_request.call_args.kwargs["profile"].provider, "deepseek")
        self.assertEqual(open_request.call_args.kwargs["timeout"], 12)
        urlopen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
