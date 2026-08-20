"""Runtime profiles: what an assistant runtime is allowed to be and to do.

An agent runtime is a set of capabilities and a voice, not a conversation id.
The previous shape compared ``conversation_id == "friday-main"`` in ten places
across two modules, so a persona could only exist by editing every call site,
and a capability could only be granted to the one conversation that literal
named. Memory extraction and reminder creation were both switched off
everywhere else as a side effect of that comparison rather than by decision.

A profile declares its capabilities as data. Adding a persona means registering
one, and granting a capability means setting a field.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable
import threading


DEFAULT_PROFILE_ID = "task"
FRIDAY_PROFILE_ID = "friday"
FRIDAY_CONVERSATION_ID = "friday-main"


@dataclass(frozen=True)
class RuntimeProfile:
    """One assistant runtime and the capabilities it carries."""

    id: str
    label: str
    # Conversations this profile claims. A profile with no ids is the fallback.
    conversation_ids: frozenset[str] = frozenset()

    # Capabilities. These were previously implied by the id comparison; making
    # them explicit is what lets an ordinary task chat keep a memory and set a
    # reminder without inheriting the rest of a persona.
    memory_enabled: bool = True
    reminders_enabled: bool = True
    proactive_messages: bool = False
    persistent_identity: bool = False

    # Voice. Rendered ahead of the account's own long-term profile text.
    system_context: str = ""

    priority: int = 0

    def claims(self, conversation_id: str) -> bool:
        return str(conversation_id or "").strip() in self.conversation_ids


TASK_PROFILE = RuntimeProfile(
    id=DEFAULT_PROFILE_ID,
    label="任务聊天",
    memory_enabled=True,
    reminders_enabled=True,
    proactive_messages=False,
    persistent_identity=False,
    system_context=(
        "你正在一个普通任务聊天中工作。它是可独立创建、可结束的任务工作区，"
        "不具备持续主体、跨任务人格或外部消息通道；但你可以记住跨任务稳定信息，"
        "也可以在用户明确要求时创建提醒。"
        "只围绕本聊天和当前项目材料完成用户交办，不要主动介绍其他助理运行时。"
    ),
    priority=0,
)


def friday_profile(assistant_name: str = "Friday") -> RuntimeProfile:
    name = str(assistant_name or "Friday").strip() or "Friday"
    return RuntimeProfile(
        id=FRIDAY_PROFILE_ID,
        label=name,
        conversation_ids=frozenset({FRIDAY_CONVERSATION_ID}),
        memory_enabled=True,
        reminders_enabled=True,
        proactive_messages=True,
        persistent_identity=True,
        system_context=(
            f"你是用户唯一、持续存在的项目经理助理“{name}”。"
            "微信、网页持续会话、项目节点和提醒属于同一个助理运行时；"
            "用户无需使用 /new，你应保持连续主体与跨工作回忆。"
            "日历等尚未提供的接口不得声称已经接入。"
        ),
        priority=100,
    )


class RuntimeProfileRegistry:
    """Holds the registered profiles and resolves a conversation to one."""

    def __init__(self, profiles: Iterable[RuntimeProfile] = ()) -> None:
        self._lock = threading.RLock()
        self._profiles: dict[str, RuntimeProfile] = {}
        for profile in profiles:
            self.register(profile)

    def register(self, profile: RuntimeProfile) -> RuntimeProfile:
        with self._lock:
            self._profiles[profile.id] = profile
        return profile

    def get(self, profile_id: str) -> RuntimeProfile | None:
        with self._lock:
            return self._profiles.get(str(profile_id or "").strip())

    def list(self) -> list[RuntimeProfile]:
        with self._lock:
            return sorted(self._profiles.values(), key=lambda item: (-item.priority, item.id))

    def resolve(self, conversation_id: str) -> RuntimeProfile:
        """Return the highest-priority profile claiming this conversation."""
        for profile in self.list():
            if profile.claims(conversation_id):
                return profile
        return self._profiles.get(DEFAULT_PROFILE_ID) or TASK_PROFILE


_REGISTRY_LOCK = threading.RLock()
_REGISTRY: RuntimeProfileRegistry | None = None


def registry(*, assistant_name_provider: Callable[[], str] | None = None) -> RuntimeProfileRegistry:
    """Return the process registry, building the built-in profiles once."""
    global _REGISTRY
    with _REGISTRY_LOCK:
        if _REGISTRY is None:
            name = assistant_name_provider() if assistant_name_provider else "Friday"
            _REGISTRY = RuntimeProfileRegistry([TASK_PROFILE, friday_profile(name)])
        return _REGISTRY


def reset_registry() -> None:
    """Drop the cached registry. Used by tests and by settings changes."""
    global _REGISTRY
    with _REGISTRY_LOCK:
        _REGISTRY = None


def resolve_profile(
    conversation_id: str,
    *,
    assistant_name_provider: Callable[[], str] | None = None,
) -> RuntimeProfile:
    return registry(assistant_name_provider=assistant_name_provider).resolve(conversation_id)
