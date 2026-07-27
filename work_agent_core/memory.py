from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json

from .config import ModelProfile
from .llm import OpenAICompatibleClient
from .session_store import ConversationSession, repair_runtime_message_sequence, sanitize_runtime_message


CHAT_CONTEXT_TOKEN_BUDGET = 256_000
CHAT_SUMMARY_TRIGGER_TOKENS = int(CHAT_CONTEXT_TOKEN_BUDGET * 0.9)
CHAT_RECENT_CONTEXT_TOKENS = 80_000
CHAT_SUMMARY_MAX_TOKENS = 4_096


@dataclass(frozen=True)
class PreparedSessionMemory:
    messages: list[dict[str, Any]]
    summary: str
    summary_message_count: int
    compacted: bool
    estimated_tokens: int
    system_context: str


def prepare_session_memory(
    client: OpenAICompatibleClient,
    profile: ModelProfile,
    session: ConversationSession,
) -> PreparedSessionMemory:
    session.messages = [
        message for message in (sanitize_runtime_message(item) for item in session.messages) if message
    ]
    session.messages = repair_runtime_message_sequence(session.messages)
    if session.summary_message_count > len(session.messages):
        session.summary_message_count = len(session.messages)
    if not session.summary:
        session.summary_message_count = 0

    covered_count = min(session.summary_message_count, len(session.messages))
    unsummarized_messages = session.messages[covered_count:]
    estimated_tokens = estimate_messages_tokens(unsummarized_messages) + estimate_context_tokens(session.summary)

    if estimated_tokens < CHAT_SUMMARY_TRIGGER_TOKENS:
        recent = unsummarized_messages if session.summary else session.messages
        return PreparedSessionMemory(
            messages=trim_to_valid_context_boundary(recent),
            summary=session.summary,
            summary_message_count=covered_count,
            compacted=False,
            estimated_tokens=estimated_tokens,
            system_context=render_summary_system_context(session.summary),
        )

    recent_messages = select_recent_messages_by_budget(unsummarized_messages, CHAT_RECENT_CONTEXT_TOKENS)
    older_count = max(0, len(unsummarized_messages) - len(recent_messages))
    older_messages = unsummarized_messages[:older_count]
    if not older_messages:
        return PreparedSessionMemory(
            messages=trim_to_valid_context_boundary(recent_messages),
            summary=session.summary,
            summary_message_count=covered_count,
            compacted=False,
            estimated_tokens=estimated_tokens,
            system_context=render_summary_system_context(session.summary),
        )

    summary = summarize_session_messages(client, profile, session.summary, older_messages)
    next_covered_count = covered_count + len(older_messages)
    session.summary = summary
    session.summary_message_count = next_covered_count

    remaining_budget = max(
        16_000,
        CHAT_CONTEXT_TOKEN_BUDGET - estimate_context_tokens(summary) - 12_000,
    )
    recent_messages = select_recent_messages_by_budget(session.messages[next_covered_count:], remaining_budget)
    return PreparedSessionMemory(
        messages=trim_to_valid_context_boundary(recent_messages),
        summary=summary,
        summary_message_count=next_covered_count,
        compacted=True,
        estimated_tokens=estimated_tokens,
        system_context=render_summary_system_context(summary),
    )


def render_summary_system_context(summary: str) -> str:
    text = str(summary or "").strip()
    if not text:
        return ""
    return (
        "当前会话 working memory 已压缩摘要（仅代表本会话较早内容，不是长期记忆）：\n"
        f"{text}\n\n"
        "使用规则：继续本会话时可依赖该摘要恢复上下文；若摘要与最近消息或工具结果冲突，"
        "以最近消息和工具结果为准。"
    )


def summarize_session_messages(
    client: OpenAICompatibleClient,
    profile: ModelProfile,
    existing_summary: str,
    older_messages: list[dict[str, Any]],
) -> str:
    response = client.chat(
        [
            {
                "role": "system",
                "content": (
                    "你是本地智能体的会话 working memory 压缩器。请把已有摘要和较早的"
                    "OpenAI messages 合并为后续同一会话可直接使用的分点 Markdown 摘要。"
                    "只能总结原文事实，不要补充新信息。必须保留：用户明确偏好、当前任务目标、"
                    "已经完成/失败的工具调用结果、关键文件路径、重要纠错、未完成事项和下一步。"
                    "工具调用不用逐字复刻参数，但要保留会影响后续判断的工具名、结果和错误。"
                ),
            },
            {
                "role": "user",
                "content": (
                    "已有摘要：\n"
                    f"{existing_summary or '（无）'}\n\n"
                    "需要并入摘要的较早 messages：\n"
                    f"{serialize_runtime_messages_for_summary(older_messages)}"
                ),
            },
        ],
        profile=profile,
        max_tokens=CHAT_SUMMARY_MAX_TOKENS,
    )
    return response.content.strip() or existing_summary


def serialize_runtime_messages_for_summary(messages: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for message in messages:
        role = message.get("role")
        if role == "user":
            blocks.append(f"用户：\n{clip_text(str(message.get('content') or ''), 4000)}")
        elif role == "assistant":
            content = str(message.get("content") or "").strip()
            tool_calls = message.get("tool_calls") if isinstance(message.get("tool_calls"), list) else []
            parts: list[str] = []
            if content:
                parts.append(clip_text(content, 4000))
            if tool_calls:
                parts.append("工具调用：")
                for call in tool_calls:
                    function = call.get("function") if isinstance(call, dict) else {}
                    name = function.get("name") if isinstance(function, dict) else ""
                    args = function.get("arguments") if isinstance(function, dict) else ""
                    parts.append(f"- {name}: {clip_text(str(args or ''), 1200)}")
            blocks.append("助手：\n" + ("\n".join(parts) if parts else "（空）"))
        elif role == "tool":
            name = str(message.get("name") or "")
            content = clip_text(str(message.get("content") or ""), 4000)
            blocks.append(f"工具结果 {name or message.get('tool_call_id') or ''}：\n{content}")
    return "\n\n".join(blocks)


def select_recent_messages_by_budget(
    messages: list[dict[str, Any]],
    token_budget: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    total = 0
    for message in reversed(messages):
        cost = estimate_message_tokens(message)
        if selected and total + cost > token_budget:
            break
        selected.append(message)
        total += cost
    selected.reverse()
    return trim_to_valid_context_boundary(selected)


def trim_to_valid_context_boundary(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    trimmed = list(messages)
    while trimmed and trimmed[0].get("role") == "tool":
        trimmed = trimmed[1:]
    return trimmed


def estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    return sum(estimate_message_tokens(message) for message in messages)


def estimate_message_tokens(message: dict[str, Any]) -> int:
    return max(8, estimate_context_tokens(serialize_message_for_token_estimate(message)) + 8)


def serialize_message_for_token_estimate(message: dict[str, Any]) -> str:
    try:
        return json.dumps(message, ensure_ascii=False, separators=(",", ":"))
    except TypeError:
        return str(message)


def estimate_context_tokens(text: str) -> int:
    if not text:
        return 0
    # Mixed Chinese/English rough estimate. We only need a stable trigger, not
    # exact tokenizer parity.
    return max(1, len(text) // 2)


def clip_text(text: str, limit: int) -> str:
    value = str(text or "")
    if len(value) <= limit:
        return value
    return value[:limit] + f"\n…[truncated {len(value) - limit} chars]"
