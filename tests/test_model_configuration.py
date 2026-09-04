from __future__ import annotations

import os
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from work_agent_core import web_server
from work_agent_core.cli import (
    delete_model_endpoint,
    delete_model_profile,
    update_model_endpoint,
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
    def test_llm_route_not_found_has_actionable_message(self) -> None:
        message = web_server.friendly_error_message(
            RuntimeError("LLM stream failed with HTTP 404: edge route not found")
        )

        self.assertIn("模型接口没有找到可用路由", message)
        self.assertIn("连接测试", message)
        self.assertIn("不会自动改用其他模型", message)

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

    def test_deepseek_vision_variant_is_inferred_as_vision_capable(self) -> None:
        profile = ModelProfile.from_dict({
            "name": "opencode-go-deepseek-v4-flash-vision-exp",
            "provider": "opencode-go",
            "base_url": "https://opencode.ai/zen/go/v1",
            "model": "deepseek-v4-flash-vision-exp",
            "api_key_env": "OPENCODE_GO_API_KEY",
        })

        self.assertTrue(profile.supports_vision)

    def test_model_profile_keeps_the_serving_context_length(self) -> None:
        profile = ModelProfile.from_dict({
            "name": "lmstudio-qwen3.8-27b",
            "provider": "lm-studio",
            "base_url": "http://100.86.69.1:1234/v1",
            "model": "qwen3.8-27b",
            "api_key_env": "LM_STUDIO_API_KEY",
            "context_length": 108032,
        })

        self.assertEqual(profile.context_length, 108032)

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

    def test_endpoint_update_applies_to_all_members_and_delete_switches_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "model_profiles.json"
            config_path.write_text(
                json.dumps(
                    {
                        "default_profile": "shared-a",
                        "profiles": [
                            {
                                "name": "shared-a",
                                "provider": "openai-compatible",
                                "base_url": "https://old.example.com/v1",
                                "model": "model-a",
                                "api_key_env": "SHARED_KEY",
                                "endpoint_id": "shared",
                            },
                            {
                                "name": "shared-b",
                                "provider": "openai-compatible",
                                "base_url": "https://old.example.com/v1",
                                "model": "model-b",
                                "api_key_env": "SHARED_KEY",
                                "endpoint_id": "shared",
                            },
                            {
                                "name": "standalone",
                                "provider": "openai-compatible",
                                "base_url": "https://standalone.example.com/v1",
                                "model": "model-c",
                                "api_key_env": "STANDALONE_KEY",
                                "endpoint_id": "standalone",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            updated = update_model_endpoint(
                config_path,
                "shared",
                {
                    "endpoint_label": "Shared Endpoint",
                    "base_url": "https://new.example.com/v1",
                    "provider": "custom-provider",
                },
            )
            self.assertEqual(updated, ["shared-a", "shared-b"])

            data = json.loads(config_path.read_text(encoding="utf-8"))
            shared_profiles = [
                item for item in data["profiles"] if item["name"] in {"shared-a", "shared-b"}
            ]
            self.assertTrue(
                all(item["base_url"] == "https://new.example.com/v1" for item in shared_profiles)
            )
            self.assertTrue(
                all(item["endpoint_label"] == "Shared Endpoint" for item in shared_profiles)
            )

            removed = delete_model_endpoint(config_path, "shared")
            self.assertEqual([item["name"] for item in removed], ["shared-a", "shared-b"])
            remaining = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual([item["name"] for item in remaining["profiles"]], ["standalone"])
            self.assertEqual(remaining["default_profile"], "standalone")

    def test_models_payload_groups_profiles_by_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            config_dir = workspace / "config"
            config_dir.mkdir()
            (config_dir / "model_profiles.json").write_text(
                json.dumps(
                    {
                        "default_profile": "sensenova-glm-5.2",
                        "profiles": [
                            {
                                "name": "sensenova-glm-5.2",
                                "provider": "openai-compatible",
                                "base_url": "https://token.sensenova.cn/v1",
                                "model": "glm-5.2",
                                "api_key_env": "SENSENOVA_API_KEY",
                                "endpoint_id": "sensenova",
                                "endpoint_label": "商汤日日新",
                            },
                            {
                                "name": "sensenova-flash-lite",
                                "provider": "openai-compatible",
                                "base_url": "https://token.sensenova.cn/v1",
                                "model": "sensenova-6.8-flash-lite",
                                "api_key_env": "SENSENOVA_API_KEY",
                                "endpoint_id": "sensenova",
                                "endpoint_label": "商汤日日新",
                            },
                            {
                                "name": "opencode-go-qwen3.8-max",
                                "provider": "opencode-go",
                                "base_url": "https://opencode.ai/zen/go/v1",
                                "model": "qwen3.8-max",
                                "api_key_env": "OPENCODE_GO_API_KEY",
                                "endpoint_id": "opencode-go",
                                "endpoint_label": "Opencode Go",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(web_server, "WORKSPACE_ROOT", workspace):
                payload = web_server.models_payload()

            self.assertEqual(payload["total_profiles"], 3)
            self.assertEqual(payload["total_endpoints"], 2)
            endpoints = payload["endpoints"]
            default_endpoint = next(item for item in endpoints if item["default"])
            self.assertEqual(default_endpoint["endpoint_id"], "sensenova")
            self.assertEqual(default_endpoint["profile_count"], 2)
            opencode = next(item for item in endpoints if item["endpoint_id"] == "opencode-go")
            self.assertEqual(opencode["label"], "Opencode Go")
            self.assertEqual(
                [profile["name"] for profile in opencode["models"]],
                ["opencode-go-qwen3.8-max"],
            )

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

    def test_import_endpoint_models_adds_missing_and_skips_existing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "config").mkdir()
            config_path = workspace / "config" / "model_profiles.json"
            config_path.write_text(
                json.dumps(
                    {
                        "default_profile": "shared-model-a",
                        "profiles": [
                            {
                                "name": "shared-model-a",
                                "provider": "openai-compatible",
                                "base_url": "https://api.example.com/v1",
                                "model": "model-a",
                                "api_key_env": "SHARED_KEY",
                                "endpoint_id": "shared",
                                "endpoint_label": "Shared",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            os.environ["SHARED_KEY"] = "secret"
            self.addCleanup(os.environ.pop, "SHARED_KEY", None)

            models_response = FakeResponse(
                {"data": [{"id": "model-a"}, {"id": "model-b"}, {"id": "model-c"}]}
            )
            with (
                patch.object(web_server, "WORKSPACE_ROOT", workspace),
                patch.object(web_server, "CONFIG_PATH", config_path.relative_to(workspace)),
                patch.object(
                    web_server.OpenAICompatibleClient,
                    "_open_request",
                    return_value=models_response,
                ),
            ):
                result = web_server.import_endpoint_models_payload({"endpoint_id": "shared"})

            self.assertEqual(result["imported"], ["model-b", "model-c"])
            self.assertEqual(result["skipped"], ["model-a"])
            data = json.loads(config_path.read_text(encoding="utf-8"))
            names = [item["name"] for item in data["profiles"]]
            self.assertIn("shared-model-b", names)
            self.assertIn("shared-model-c", names)
            self.assertEqual(len(names), 3)
            new_profile = next(item for item in data["profiles"] if item["name"] == "shared-model-b")
            self.assertEqual(new_profile["api_key_env"], "SHARED_KEY")
            self.assertEqual(new_profile["endpoint_id"], "shared")


class ProviderAuthTests(unittest.TestCase):
    """认证头各家写法不同，所以它属于 profile，不属于客户端。"""

    @staticmethod
    def _profile(**overrides):
        from work_agent_core.config import ModelProfile

        base = {
            "name": "p",
            "provider": "openai-compatible",
            "base_url": "https://example.invalid/v1",
            "model": "m",
            "api_key_env": "TEST_PROVIDER_KEY",
        }
        base.update(overrides)
        return ModelProfile(**base)

    def setUp(self) -> None:
        os.environ["TEST_PROVIDER_KEY"] = "k-123"
        self.addCleanup(os.environ.pop, "TEST_PROVIDER_KEY", None)

    def test_the_default_is_a_bearer_authorization_header(self) -> None:
        headers = self._profile().auth_headers()
        self.assertEqual(headers["Authorization"], "Bearer k-123")

    def test_a_provider_can_use_a_bare_key_in_its_own_header(self) -> None:
        headers = self._profile(auth_header="api-key", auth_scheme="").auth_headers()

        self.assertEqual(headers["api-key"], "k-123")
        self.assertNotIn("Authorization", headers)

    def test_dots_maps_the_thinking_switch_instead_of_reasoning_effort(self) -> None:
        from work_agent_core.llm import apply_reasoning_controls

        profile = self._profile(
            name="dots3-note",
            base_url="https://note3-prev-api.askdiandian.com/v1",
            model="dots3-note-prev",
        )

        on = apply_reasoning_controls({}, profile=profile, reasoning_effort="high")
        off = apply_reasoning_controls({}, profile=profile, reasoning_effort="light")

        self.assertEqual(on["chat_template_kwargs"], {"enable_thinking": True})
        self.assertEqual(off["chat_template_kwargs"], {"enable_thinking": False})
        # 该模型只有开/关两档，没有 reasoning_effort
        self.assertNotIn("reasoning_effort", on)


if __name__ == "__main__":
    unittest.main()
