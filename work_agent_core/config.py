from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import hashlib
import json
import os
import re


DEFAULT_CONFIG_PATH = Path("config/model_profiles.json")


def load_env_file(path: str | Path, *, override: bool = False) -> None:
    env_path = Path(path)
    if not env_path.exists() or not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or (key in os.environ and not override):
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


def api_key_env_for_profile(profile_name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", str(profile_name).strip()).strip("_").upper()
    normalized = normalized[:48] or "MODEL"
    suffix = hashlib.sha256(str(profile_name).encode("utf-8")).hexdigest()[:8].upper()
    return f"WORK_AGENT_MODEL_{normalized}_{suffix}_API_KEY"


def save_env_value(path: str | Path, key: str, value: str) -> None:
    env_path = Path(path)
    env_key = str(key).strip()
    env_value = str(value).strip()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", env_key):
        raise ValueError("API 密钥环境变量名格式无效。")
    if not env_value:
        raise ValueError("API 密钥不能为空。")
    if "\n" in env_value or "\r" in env_value or "\x00" in env_value:
        raise ValueError("API 密钥不能包含换行或空字符。")

    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.is_file() else []
    replacement = f"{env_key}={env_value}"
    key_pattern = re.compile(rf"^\s*{re.escape(env_key)}\s*=")
    updated = False
    next_lines: list[str] = []
    for line in lines:
        if key_pattern.match(line):
            if not updated:
                next_lines.append(replacement)
                updated = True
            continue
        next_lines.append(line)
    if not updated:
        if next_lines and next_lines[-1].strip():
            next_lines.append("")
        next_lines.append(replacement)

    env_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = env_path.with_suffix(f"{env_path.suffix}.tmp")
    temporary_path.write_text("\n".join(next_lines) + "\n", encoding="utf-8")
    temporary_path.chmod(0o600)
    temporary_path.replace(env_path)
    env_path.chmod(0o600)
    os.environ[env_key] = env_value


def delete_env_value(path: str | Path, key: str) -> None:
    env_path = Path(path)
    env_key = str(key).strip()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", env_key):
        raise ValueError("API 密钥环境变量名格式无效。")
    if env_path.is_file():
        key_pattern = re.compile(rf"^\s*{re.escape(env_key)}\s*=")
        lines = [
            line
            for line in env_path.read_text(encoding="utf-8").splitlines()
            if not key_pattern.match(line)
        ]
        temporary_path = env_path.with_suffix(f"{env_path.suffix}.tmp")
        temporary_path.write_text(
            "\n".join(lines).rstrip() + ("\n" if lines else ""),
            encoding="utf-8",
        )
        temporary_path.chmod(0o600)
        temporary_path.replace(env_path)
        env_path.chmod(0o600)
    os.environ.pop(env_key, None)


def infer_model_vision_support(data: dict[str, Any]) -> bool:
    """Conservative fallback for legacy profiles without an explicit flag."""

    identity = " ".join(
        str(data.get(key) or "")
        for key in ("name", "provider", "base_url", "model")
    ).lower()
    if "deepseek" in identity:
        return False
    return any(
        marker in identity
        for marker in (
            "gpt-4o", "gpt-4.1", "gpt-5", "gemini", "claude-3", "claude-4",
            "qwen-vl", "qwen2.5-vl", "qvq", "glm-4v", "pixtral", "llava", "vision",
        )
    )


@dataclass(frozen=True)
class ModelProfile:
    name: str
    provider: str
    base_url: str
    model: str
    api_key_env: str
    temperature: float = 0.6
    max_tokens: int = 8192
    timeout_seconds: int = 120
    supports_vision: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelProfile":
        return cls(
            name=str(data["name"]),
            provider=str(data.get("provider") or "openai-compatible"),
            base_url=str(data["base_url"]).rstrip("/"),
            model=str(data["model"]),
            api_key_env=str(data.get("api_key_env") or "OPENAI_API_KEY"),
            temperature=float(data.get("temperature", 0.6)),
            max_tokens=int(data.get("max_tokens", 8192)),
            timeout_seconds=int(data.get("timeout_seconds", 120)),
            supports_vision=(
                bool(data["supports_vision"])
                if isinstance(data.get("supports_vision"), bool)
                else infer_model_vision_support(data)
            ),
        )

    def api_key(self) -> str:
        value = os.getenv(self.api_key_env)
        if not value:
            raise RuntimeError(
                f"Missing API key env var {self.api_key_env!r} for model profile {self.name!r}."
            )
        return value


class ModelRegistry:
    def __init__(self, profiles: dict[str, ModelProfile], default_profile: str) -> None:
        if default_profile not in profiles:
            raise ValueError(f"Default profile {default_profile!r} is not defined.")
        self._profiles = profiles
        self.default_profile = default_profile

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONFIG_PATH) -> "ModelRegistry":
        config_path = Path(path)
        if config_path.parent.name == "config":
            load_env_file(config_path.parent.parent / ".env")
        load_env_file(Path(".env"))
        data = json.loads(config_path.read_text(encoding="utf-8"))
        profiles = {
            profile.name: profile
            for profile in (ModelProfile.from_dict(item) for item in data.get("profiles", []))
        }
        default_profile = os.getenv("WORK_AGENT_MODEL_PROFILE") or data["default_profile"]
        return cls(profiles, default_profile)

    def names(self) -> list[str]:
        return sorted(self._profiles)

    def get(self, name: str | None = None) -> ModelProfile:
        profile_name = name or self.default_profile
        try:
            return self._profiles[profile_name]
        except KeyError as error:
            available = ", ".join(self.names())
            raise KeyError(f"Unknown model profile {profile_name!r}. Available: {available}") from error

    def as_table(self) -> str:
        lines = ["name\tprovider\tmodel\tbase_url\tapi_key_env\tdefault"]
        for name in self.names():
            profile = self._profiles[name]
            marker = "*" if name == self.default_profile else ""
            lines.append(
                "\t".join(
                    [
                        profile.name,
                        profile.provider,
                        profile.model,
                        profile.base_url,
                        profile.api_key_env,
                        marker,
                    ]
                )
            )
        return "\n".join(lines)
