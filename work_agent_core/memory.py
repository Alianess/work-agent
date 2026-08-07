from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
from hashlib import sha256
import json
import math
import re
import threading
import time

from .config import ModelProfile
from .llm import OpenAICompatibleClient
from .recall_archive import (
    RECALL_ARCHIVE_VERSION,
    build_recall_episodes,
    compact_messages_for_archive,
)
from .session_store import ConversationSession, repair_runtime_message_sequence, sanitize_runtime_message


CHAT_CONTEXT_TOKEN_BUDGET = 256_000
CHAT_SUMMARY_TRIGGER_TOKENS = int(CHAT_CONTEXT_TOKEN_BUDGET * 0.9)
CHAT_SUMMARY_MAX_TOKENS = 8_192
CHAT_RECENT_VISIBLE_TURNS = 2
# Leave room for the model's answer plus the system prompt, native tool
# schemas, skill catalog and other request framing that is not stored in
# ``session.messages``.  Without this reserve a 160k-message session can
# silently become a >230k request and repeatedly time out before compaction.
CHAT_RUNTIME_OVERHEAD_RESERVE_TOKENS = 24_000
ACTIVE_REACT_CHECKPOINT_TRIGGER_TOKENS = CHAT_SUMMARY_TRIGGER_TOKENS
ACTIVE_REACT_CHECKPOINT_MAX_TOKENS = 8_192

CHAT_SUMMARY_SECTIONS = (
    "当前目标与用户意图",
    "已确认事实与关键决定",
    "已完成步骤与结果",
    "未完成事项与下一步",
    "当前任务涉及的人物、公司与项目",
    "文件、产出与证据位置",
    "用户对本任务的要求、约束与纠正",
    "工具执行状态、错误与待审批动作",
)


class ContextCompactionError(RuntimeError):
    """The configured model could not produce a trustworthy continuation summary."""


class ContextCompactionCancelled(RuntimeError):
    """The user cancelled while a continuation summary was being generated."""


@dataclass(frozen=True)
class SessionMemoryInspection:
    messages: list[dict[str, Any]]
    covered_count: int
    estimated_tokens: int


@dataclass(frozen=True)
class PreparedSessionMemory:
    messages: list[dict[str, Any]]
    summary: str
    summary_message_count: int
    compacted: bool
    estimated_tokens: int
    system_context: str


def inspect_session_memory(
    session: ConversationSession,
    *,
    reserved_tokens: int = 0,
) -> SessionMemoryInspection:
    """Sanitize and account for a session once, without mutating it."""
    messages = [
        message for message in (sanitize_runtime_message(item) for item in session.messages) if message
    ]
    messages = repair_runtime_message_sequence(messages)
    covered_count = min(max(0, int(session.summary_message_count or 0)), len(messages))
    if not str(session.summary or "").strip():
        covered_count = 0
    estimated_tokens = (
        estimate_messages_tokens(messages[covered_count:])
        + estimate_context_tokens(session.summary)
        + max(0, int(reserved_tokens))
    )
    return SessionMemoryInspection(
        messages=messages,
        covered_count=covered_count,
        estimated_tokens=estimated_tokens,
    )


def estimate_session_memory_tokens(session: ConversationSession, *, reserved_tokens: int = 0) -> int:
    """Return the estimated request size without mutating the session."""

    return inspect_session_memory(session, reserved_tokens=reserved_tokens).estimated_tokens


def prepare_session_memory(
    client: OpenAICompatibleClient,
    profile: ModelProfile,
    session: ConversationSession,
    *,
    reserved_tokens: int = 0,
    force: bool = False,
    inspection: SessionMemoryInspection | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> PreparedSessionMemory:
    inspected = inspection or inspect_session_memory(session, reserved_tokens=reserved_tokens)
    session.messages = list(inspected.messages)
    session.summary_message_count = inspected.covered_count
    covered_count = inspected.covered_count
    archive_is_stale = any(
        int(item.get("archive_version") or 0) != RECALL_ARCHIVE_VERSION
        for item in session.recall_episodes
        if isinstance(item, dict)
    )
    if covered_count and (not session.recall_episodes or archive_is_stale):
        session.recall_episodes = build_recall_episodes(
            session.messages[:covered_count],
        )
        session.messages[:covered_count] = compact_messages_for_archive(
            session.messages[:covered_count]
        )
    unsummarized_messages = session.messages[covered_count:]
    estimated_tokens = inspected.estimated_tokens

    if not force and estimated_tokens < CHAT_SUMMARY_TRIGGER_TOKENS:
        recent = runtime_messages_with_retained_turns(
            session.messages,
            covered_count=covered_count,
            summary=session.summary,
        )
        return PreparedSessionMemory(
            messages=trim_to_valid_context_boundary(recent),
            summary=session.summary,
            summary_message_count=covered_count,
            compacted=False,
            estimated_tokens=estimated_tokens,
            system_context=render_summary_system_context(session.summary),
        )

    completed_end = completed_message_prefix_end(unsummarized_messages)
    completed_messages = unsummarized_messages[:completed_end]
    if not completed_messages:
        return PreparedSessionMemory(
            messages=runtime_messages_with_retained_turns(
                session.messages,
                covered_count=covered_count,
                summary=session.summary,
            ),
            summary=session.summary,
            summary_message_count=covered_count,
            compacted=False,
            estimated_tokens=estimated_tokens,
            system_context=render_summary_system_context(session.summary),
        )

    summary = summarize_session_messages(
        client,
        profile,
        session.summary,
        completed_messages,
        cancel_check=cancel_check,
    )

    next_covered_count = covered_count + len(completed_messages)
    session.recall_episodes = build_recall_episodes(
        completed_messages,
        start_message_index=covered_count,
        existing=session.recall_episodes if covered_count else [],
    )
    session.messages[covered_count:next_covered_count] = compact_messages_for_archive(
        session.messages[covered_count:next_covered_count]
    )
    session.summary = summary
    session.summary_message_count = next_covered_count
    session.compaction_events.append(
        {
            "id": f"compact-{int(time.time())}-{next_covered_count}",
            "from_message_index": covered_count,
            "to_message_index": next_covered_count,
            "summary_sha256": sha256(summary.encode("utf-8")).hexdigest(),
            "episode_count": len(session.recall_episodes),
            "created_at": int(time.time()),
        }
    )
    session.compaction_events = session.compaction_events[-64:]

    recent_visible_turns = extract_recent_visible_turns(
        session.messages[:next_covered_count],
        turn_limit=CHAT_RECENT_VISIBLE_TURNS,
    )
    active_tail = session.messages[next_covered_count:]
    return PreparedSessionMemory(
        messages=trim_to_valid_context_boundary(recent_visible_turns + active_tail),
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
        "当前任务断点续作摘要（只用于在本任务上下文被压缩后继续手头工作；"
        "不是长期记忆、用户画像或项目档案）：\n"
        f"{text}\n\n"
        "使用规则：依靠它恢复当前任务的目标、进度、证据和下一步，不得据此扩展长期事实；"
        "若它与后续原始消息或工具结果冲突，以后者为准。"
    )


def summarize_session_messages(
    client: OpenAICompatibleClient,
    profile: ModelProfile,
    existing_summary: str,
    older_messages: list[dict[str, Any]],
    *,
    cancel_check: Callable[[], bool] | None = None,
) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "你是项目经理助理的高保真“当前任务断点续作”压缩器。你的唯一任务是把"
                "已有工作摘要与本次已完成的 messages 滚动合并，使上下文被截断后仍能"
                "从当前进度继续工作，而不必从头开始。这不是长期记忆、用户画像、人物库、"
                "项目档案或跨会话知识整理；不得为了未来可能有用而扩写。只能记录输入中"
                "与当前任务延续有关的事实，不得推断或补充。\n\n"
                "保真规则：\n"
                "1. 所有会影响后续行动的目标、决定、承诺、纠正、未完成项都必须保留；"
                "不要为了简短合并掉不同事项。\n"
                "2. 人名、公司名、项目名、日期、时间、金额、数量、版本、状态、路径、URL、"
                "错误文本和责任边界应尽量原样保留。\n"
                "3. 已有摘要中的信息，只有在新增 messages 明确否定、纠正或取代它时才能"
                "删除；发生冲突时同时写明旧说法、新说法和当前采用版本。\n"
                "4. 工具调用不必逐字复制参数，但必须保留工具名、关键输入范围、成功结果、"
                "失败原因、部分完成状态、生成文件、待审批动作和仍可复用的中间结果。\n"
                "5. 区分已确认事实、模型建议和待核实信息；不要把建议写成既成事实。\n"
                "6. 每个栏目可以有任意数量条目，以信息完整为先；确实没有内容才写“无”。\n"
                "7. 最近两轮的完整 ReAct 工具链也在输入中；把其中会影响续作的信息并入"
                "摘要，不要因为运行时还会展示最近两轮最终回答而省略工具证据。\n"
                "8. 使用紧凑 Markdown 条目，不写寒暄、修辞、思维链或重复内容。\n\n"
                "必须严格使用以下八个二级标题，保持顺序：\n"
                + "\n".join(f"## {section}" for section in CHAT_SUMMARY_SECTIONS)
            ),
        },
        {
            "role": "user",
            "content": (
                "已有摘要：\n"
                f"{existing_summary or '（无）'}\n\n"
                "本次需要并入工作摘要的完整已完成 messages：\n"
                f"{serialize_runtime_messages_for_summary(older_messages)}"
            ),
        },
    ]
    try:
        if cancel_check is None or not hasattr(client, "chat_tools_stream"):
            response = client.chat(messages, profile=profile, max_tokens=CHAT_SUMMARY_MAX_TOKENS)
        else:
            if cancel_check():
                raise ContextCompactionCancelled("用户停止了上下文压缩。")
            cancel_event = threading.Event()
            watcher_finished = threading.Event()

            def watch_cancellation() -> None:
                while not watcher_finished.wait(0.1):
                    try:
                        if cancel_check():
                            cancel_event.set()
                            return
                    except Exception:
                        continue

            watcher = threading.Thread(
                target=watch_cancellation,
                name="work-agent-compaction-cancel",
                daemon=True,
            )
            watcher.start()
            try:
                response = client.chat_tools_stream(
                    messages,
                    profile=profile,
                    max_tokens=CHAT_SUMMARY_MAX_TOKENS,
                    cancel_event=cancel_event,
                )
            finally:
                watcher_finished.set()
            if cancel_event.is_set() or cancel_check():
                raise ContextCompactionCancelled("用户停止了上下文压缩。")
    except ContextCompactionCancelled:
        raise
    except Exception as error:
        if cancel_check is not None and cancel_check():
            raise ContextCompactionCancelled("用户停止了上下文压缩。") from error
        raise ContextCompactionError(
            f"当前模型压缩会话失败：{type(error).__name__}: {error}。原始会话未改写，也不会自动切换或重试模型。"
        ) from error
    summary = str(response.content or "").strip()
    if not summary:
        raise ContextCompactionError(
            "当前模型没有返回可用的会话摘要。原始会话未改写，也不会自动切换或重试模型。"
        )
    return summary


def summarize_active_react_checkpoint(
    client: OpenAICompatibleClient,
    profile: ModelProfile,
    active_messages: list[dict[str, Any]],
    *,
    task_plan: list[dict[str, Any]] | None = None,
) -> str:
    """Compress a still-running ReAct turn into a continuation checkpoint.

    This is deliberately separate from the rolling conversation summary.  It
    is allowed to replace bulky, already-completed tool exchanges in the next
    model request, while the durable session and turn trace keep the originals.
    """

    plan_text = json.dumps(task_plan or [], ensure_ascii=False, indent=2)
    try:
        response = client.chat(
        [
            {
                "role": "system",
                "content": (
                    "你是单智能体长任务的高保真执行检查点压缩器。输入是一项尚未完成的任务中，"
                    "已经走过的 ReAct 实施路径。请生成可直接交给同一智能体继续执行的检查点，"
                    "而不是总结文章、长期记忆或最终答复。\n\n"
                    "必须保留：用户当前目标与约束；模型已经公开写出的路线判断和关键修正；"
                    "当前计划及每步状态；已经调用的工具及其关键输入范围；每项操作的成功、失败、"
                    "部分完成和验证证据；精确文件路径、URL、命令目的、错误文本、待审批动作；"
                    "已经改变的代码/材料及仍未完成的下一动作。实施路径应按原发生顺序组织，"
                    "公开工作说明尽量保留原文。\n"
                    "可以删除：终端逐行回显、重复进度心跳、大段可再生文件内容、重复参数和不影响"
                    "下一步的机械细节。不得删除整条实施路径，不得把建议写成已完成，不得编造结果。\n\n"
                    "严格使用以下标题：\n"
                    "## 当前目标与完成条件\n"
                    "## 当前计划与进度\n"
                    "## 已走过的实施路径（按顺序）\n"
                    "## 已修改内容与关键证据\n"
                    "## 错误、风险与待确认事项\n"
                    "## 下一步准确动作"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"当前活计划：\n{plan_text}\n\n"
                    "本轮尚未完成的完整 ReAct messages：\n"
                    f"{serialize_runtime_messages_for_summary(active_messages)}"
                ),
            },
        ],
            profile=profile,
            max_tokens=ACTIVE_REACT_CHECKPOINT_MAX_TOKENS,
        )
    except Exception as error:
        raise ContextCompactionError(
            f"当前模型压缩运行检查点失败：{type(error).__name__}: {error}。"
        ) from error
    checkpoint = str(response.content or "").strip()
    if not checkpoint:
        raise ContextCompactionError("当前模型没有返回可用的运行检查点，不能安全继续本轮长任务。")
    return checkpoint


def serialize_runtime_messages_for_summary(messages: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for message in messages:
        role = message.get("role")
        if role == "user":
            blocks.append(f"用户：\n{str(message.get('content') or '')}")
        elif role == "assistant":
            content = str(message.get("content") or "").strip()
            tool_calls = message.get("tool_calls") if isinstance(message.get("tool_calls"), list) else []
            parts: list[str] = []
            if content:
                parts.append(content)
            if tool_calls:
                parts.append("工具调用：")
                for call in tool_calls:
                    function = call.get("function") if isinstance(call, dict) else {}
                    name = function.get("name") if isinstance(function, dict) else ""
                    args = function.get("arguments") if isinstance(function, dict) else ""
                    parts.append(f"- {name}: {str(args or '')}")
            blocks.append("助手：\n" + ("\n".join(parts) if parts else "（空）"))
        elif role == "tool":
            name = str(message.get("name") or "")
            content = str(message.get("content") or "")
            blocks.append(f"工具结果 {name or message.get('tool_call_id') or ''}：\n{content}")
    return "\n\n".join(blocks)


def completed_message_prefix_end(messages: list[dict[str, Any]]) -> int:
    """Return historical messages, leaving only the current turn raw.

    Older interrupted tool turns are still history once a later user turn
    exists.  They are safe to summarize as interrupted work and must not block
    compaction of every turn that follows them.
    """
    user_indexes = [
        index for index, message in enumerate(messages) if message.get("role") == "user"
    ]
    if not user_indexes:
        return 0
    current_turn_start = user_indexes[-1]
    if turn_has_final_answer(messages[current_turn_start:]):
        return len(messages)
    return current_turn_start


def turn_has_final_answer(messages: list[dict[str, Any]]) -> bool:
    for message in reversed(messages):
        role = message.get("role")
        if role == "tool":
            continue
        return bool(
            role == "assistant"
            and str(message.get("content") or "").strip()
            and not message.get("tool_calls")
        )
    return False


def extract_recent_visible_turns(
    messages: list[dict[str, Any]],
    *,
    turn_limit: int,
) -> list[dict[str, Any]]:
    turns: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") == "user":
            if current:
                turns.append(current)
            current = [message]
        elif current:
            current.append(message)
    if current:
        turns.append(current)

    visible: list[dict[str, Any]] = []
    completed = [turn for turn in turns if turn_has_final_answer(turn)]
    for turn in completed[-max(0, turn_limit):]:
        user = turn[0]
        final = next(
            (
                message
                for message in reversed(turn)
                if message.get("role") == "assistant"
                and str(message.get("content") or "").strip()
                and not message.get("tool_calls")
            ),
            None,
        )
        if final is None:
            continue
        visible.append({"role": "user", "content": str(user.get("content") or "")})
        visible.append({"role": "assistant", "content": str(final.get("content") or "")})
    return visible


def runtime_messages_with_retained_turns(
    messages: list[dict[str, Any]],
    *,
    covered_count: int,
    summary: str,
) -> list[dict[str, Any]]:
    if not summary:
        return trim_to_valid_context_boundary(messages)
    retained = extract_recent_visible_turns(
        messages[:covered_count],
        turn_limit=CHAT_RECENT_VISIBLE_TURNS,
    )
    return trim_to_valid_context_boundary(retained + messages[covered_count:])


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
    # CJK text is commonly close to one token per character, while ASCII-heavy
    # JSON and English average closer to four characters per token.  Counting
    # both separately avoids materially underestimating long Chinese meeting
    # transcripts while keeping the estimator deterministic and dependency-free.
    cjk_chars = len(re.findall(r"[\u3400-\u9fff\uf900-\ufaff]", text))
    other_chars = max(0, len(text) - cjk_chars)
    return max(1, cjk_chars + math.ceil(other_chars / 4))


def clip_text(text: str, limit: int) -> str:
    value = str(text or "")
    if len(value) <= limit:
        return value
    return value[:limit] + f"\n…[truncated {len(value) - limit} chars]"
