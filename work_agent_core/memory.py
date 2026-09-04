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
from .session_store import (
    ConversationSession,
    is_turn_runtime_context_message,
    repair_runtime_message_sequence,
    sanitize_runtime_message,
)


CHAT_CONTEXT_TOKEN_BUDGET = 256_000
CONTEXT_COMPACTION_TRIGGER_RATIO = 0.85
CHAT_SUMMARY_TRIGGER_TOKENS = int(
    CHAT_CONTEXT_TOKEN_BUDGET * CONTEXT_COMPACTION_TRIGGER_RATIO
)
# Serialized bytes are a process/transport guard, not a proxy for model tokens.
# CJK text commonly occupies three UTF-8 bytes per token and JSON/tool payloads
# often occupy more, so the previous 256 KB threshold compacted healthy ~140k
# token sessions far before their context limit.
CHAT_SUMMARY_TRIGGER_SERIALIZED_BYTES = 4 * 1024 * 1024
# Tool text is already represented in token pressure. Keep a separate
# multi-megabyte process guard for pathological payloads, but do not invoke a
# model summary merely because an ordinary read/search returned 100k chars.
CHAT_SUMMARY_TRIGGER_TOOL_RESULT_CHARS = 4 * 1024 * 1024
CHAT_SUMMARY_MAX_TOKENS = 131_072
CHAT_RECENT_VISIBLE_TURNS = 2
# Leave room for the model's answer plus the system prompt, native tool
# schemas, skill catalog and other request framing that is not stored in
# ``session.messages``.  Without this reserve a 160k-message session can
# silently become a >230k request and repeatedly time out before compaction.
CHAT_RUNTIME_OVERHEAD_RESERVE_TOKENS = 24_000
PROVIDER_USAGE_DYNAMIC_SAFETY_TOKENS = 4_096
PROVIDER_USAGE_BASELINE_KEY = "provider_token_usage_baseline"
ACTIVE_REACT_CHECKPOINT_TRIGGER_TOKENS = CHAT_SUMMARY_TRIGGER_TOKENS
ACTIVE_REACT_CHECKPOINT_MAX_TOKENS = 131_072

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


def profile_context_trigger_tokens(profile: ModelProfile | None) -> int:
    """Compact before the selected endpoint's real context window is exhausted."""

    if profile is None:
        return CHAT_SUMMARY_TRIGGER_TOKENS
    configured = max(1, int(getattr(profile, "context_length", 0) or CHAT_CONTEXT_TOKEN_BUDGET))
    # The old global min() silently capped every large-context model at the
    # 256k fallback's 85% line.  A 1M model therefore compacted at 217.6k even
    # when the provider accurately reported a much larger valid input.
    return max(1, int(configured * CONTEXT_COMPACTION_TRIGGER_RATIO))


def compaction_output_token_budget(profile: ModelProfile) -> int:
    """Give compaction enough output room without crowding out its input.

    The old fixed 8k limit was consumed entirely by reasoning on GLM.  A
    compaction cap should follow the selected model, while remaining inside
    the space left after the 85% input trigger and a small framing reserve.
    """

    configured_output = max(1, int(profile.max_tokens or 1))
    available_after_trigger = max(
        1_024,
        int(profile.context_length)
        - profile_context_trigger_tokens(profile)
        - PROVIDER_USAGE_DYNAMIC_SAFETY_TOKENS,
    )
    return min(configured_output, available_after_trigger)


def token_count_source_label(source: str) -> str:
    """Render accounting provenance without implying an estimate is exact."""

    return {
        "provider_usage": "供应商最近一次请求的 input/prompt usage",
        "provider_usage_plus_estimated_delta": (
            "供应商最近一次请求的 input/prompt usage + 此后新增消息估算"
        ),
        "estimated_full_request": "发送前完整请求本地估算",
        "estimated_session_plus_reserve": "会话消息本地估算 + 请求预留",
    }.get(str(source or ""), str(source or "未知"))


@dataclass(frozen=True)
class SessionMemoryInspection:
    messages: list[dict[str, Any]]
    covered_count: int
    estimated_tokens: int
    serialized_bytes: int
    tool_result_chars: int
    token_count_source: str = "estimated_session_plus_reserve"


@dataclass(frozen=True)
class PreparedSessionMemory:
    messages: list[dict[str, Any]]
    summary: str
    summary_message_count: int
    compacted: bool
    # ``estimated_tokens`` is deliberately the pressure measured *before*
    # compaction.  Keep the post-compaction working-set estimate separate so
    # callers never compare the smaller result with the trigger and claim the
    # impossible (for example, "134k exceeded 230k").
    estimated_tokens: int
    post_compaction_estimated_tokens: int
    serialized_bytes: int
    tool_result_chars: int
    post_compaction_serialized_bytes: int
    post_compaction_tool_result_chars: int
    system_context: str
    pressure_reasons: tuple[str, ...] = ()
    token_count_source: str = "estimated_session_plus_reserve"


def message_pressure_metrics(
    messages: list[dict[str, Any]],
    *,
    summary: str = "",
) -> tuple[int, int]:
    """Measure replay pressure that token-window accounting does not capture."""

    serialized_bytes = len(
        json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ) + len(str(summary or "").encode("utf-8"))
    tool_result_chars = sum(
        len(str(message.get("content") or ""))
        for message in messages
        if message.get("role") == "tool"
    )
    return serialized_bytes, tool_result_chars


def context_pressure_reasons(
    inspection: SessionMemoryInspection,
    profile: ModelProfile | None,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if inspection.estimated_tokens >= profile_context_trigger_tokens(profile):
        reasons.append("tokens")
    if inspection.serialized_bytes >= CHAT_SUMMARY_TRIGGER_SERIALIZED_BYTES:
        reasons.append("serialized_bytes")
    if inspection.tool_result_chars >= CHAT_SUMMARY_TRIGGER_TOOL_RESULT_CHARS:
        reasons.append("tool_results")
    return tuple(reasons)


def inspect_session_memory(
    session: ConversationSession,
    *,
    reserved_tokens: int = 0,
    profile: ModelProfile | None = None,
) -> SessionMemoryInspection:
    """Sanitize and account for a session once, without mutating it."""
    messages = [
        message for message in (sanitize_runtime_message(item) for item in session.messages) if message
    ]
    messages = repair_runtime_message_sequence(messages)
    covered_count = min(max(0, int(session.summary_message_count or 0)), len(messages))
    if not str(session.summary or "").strip():
        covered_count = 0
    active_messages = messages[covered_count:]
    raw_context_tokens = estimate_messages_tokens(active_messages) + estimate_context_tokens(session.summary)
    serialized_bytes, tool_result_chars = message_pressure_metrics(
        active_messages,
        summary=session.summary,
    )
    estimated_tokens, token_count_source = usage_baseline_context_accounting(
        session,
        messages=messages,
        raw_context_tokens=raw_context_tokens,
        reserved_tokens=reserved_tokens,
        profile=profile,
    )
    return SessionMemoryInspection(
        messages=messages,
        covered_count=covered_count,
        estimated_tokens=estimated_tokens,
        serialized_bytes=serialized_bytes,
        tool_result_chars=tool_result_chars,
        token_count_source=token_count_source,
    )


def estimate_session_memory_tokens(
    session: ConversationSession,
    *,
    reserved_tokens: int = 0,
    profile: ModelProfile | None = None,
) -> int:
    """Return the estimated request size without mutating the session."""

    return inspect_session_memory(
        session,
        reserved_tokens=reserved_tokens,
        profile=profile,
    ).estimated_tokens


def raw_session_context_tokens(session: ConversationSession) -> int:
    """Estimate only the conversation state represented by a persisted session."""
    messages = [
        message for message in (sanitize_runtime_message(item) for item in session.messages) if message
    ]
    messages = repair_runtime_message_sequence(messages)
    covered_count = min(max(0, int(session.summary_message_count or 0)), len(messages))
    if not str(session.summary or "").strip():
        covered_count = 0
    return estimate_messages_tokens(messages[covered_count:]) + estimate_context_tokens(session.summary)


def usage_baseline_context_tokens(
    session: ConversationSession,
    *,
    messages: list[dict[str, Any]],
    raw_context_tokens: int,
    reserved_tokens: int,
    profile: ModelProfile | None,
) -> int:
    """Use the last real provider input count plus only the later message delta."""
    return usage_baseline_context_accounting(
        session,
        messages=messages,
        raw_context_tokens=raw_context_tokens,
        reserved_tokens=reserved_tokens,
        profile=profile,
    )[0]


def usage_baseline_context_accounting(
    session: ConversationSession,
    *,
    messages: list[dict[str, Any]],
    raw_context_tokens: int,
    reserved_tokens: int,
    profile: ModelProfile | None,
) -> tuple[int, str]:
    """Return pressure and provenance without conflating usage categories.

    A provider's most recent single-request input count is authoritative for
    that exact request.  Only content added after its persisted anchor is
    estimated locally.  ``completion_tokens`` and ``total_tokens`` are billing
    metrics and intentionally never participate in context-window pressure.
    """
    fallback = max(0, int(raw_context_tokens)) + max(0, int(reserved_tokens))
    baseline = session.metadata.get(PROVIDER_USAGE_BASELINE_KEY)
    if not isinstance(baseline, dict) or profile is None:
        return fallback, "estimated_session_plus_reserve"
    if (
        str(baseline.get("profile") or "") != profile.name
        or str(baseline.get("model") or "") != profile.model
        or str(baseline.get("base_url") or "").rstrip("/") != profile.base_url.rstrip("/")
        or int(baseline.get("summary_message_count") or 0)
        != min(max(0, int(session.summary_message_count or 0)), len(messages))
        or str(baseline.get("summary_sha256") or "")
        != sha256(str(session.summary or "").encode("utf-8")).hexdigest()
    ):
        return fallback, "estimated_session_plus_reserve"
    prompt_tokens = int(baseline.get("prompt_tokens") or 0)
    anchor_tokens = int(baseline.get("anchor_raw_session_tokens") or 0)
    if prompt_tokens <= 0 or anchor_tokens < 0 or raw_context_tokens < anchor_tokens:
        return fallback, "estimated_session_plus_reserve"
    delta_tokens = raw_context_tokens - anchor_tokens
    # prompt_tokens already includes system text, tool schemas and request framing.
    # Only reserve the next answer and a small allowance for dynamic framing.
    output_reserve = max(
        0,
        int(reserved_tokens) - CHAT_RUNTIME_OVERHEAD_RESERVE_TOKENS,
    )
    pressure = (
        prompt_tokens
        + delta_tokens
        + output_reserve
        + PROVIDER_USAGE_DYNAMIC_SAFETY_TOKENS
    )
    if delta_tokens or output_reserve or PROVIDER_USAGE_DYNAMIC_SAFETY_TOKENS:
        return pressure, "provider_usage_plus_estimated_delta"
    return pressure, "provider_usage"


def provider_usage_baseline_payload(
    profile: ModelProfile,
    usage: dict[str, Any],
    *,
    anchor_raw_session_tokens: int,
    summary_message_count: int,
    summary: str = "",
) -> dict[str, Any] | None:
    prompt_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    if prompt_tokens <= 0:
        return None
    return {
        "profile": profile.name,
        "model": profile.model,
        "base_url": profile.base_url.rstrip("/"),
        "prompt_tokens": prompt_tokens,
        "anchor_raw_session_tokens": max(0, int(anchor_raw_session_tokens)),
        "summary_message_count": max(0, int(summary_message_count)),
        "summary_sha256": sha256(str(summary or "").encode("utf-8")).hexdigest(),
        "captured_at": int(time.time()),
    }


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
    inspected = inspection or inspect_session_memory(
        session,
        reserved_tokens=reserved_tokens,
        profile=profile,
    )
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
    pressure_reasons = context_pressure_reasons(inspected, profile)

    if not force and not pressure_reasons:
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
            post_compaction_estimated_tokens=estimated_tokens,
            serialized_bytes=inspected.serialized_bytes,
            tool_result_chars=inspected.tool_result_chars,
            post_compaction_serialized_bytes=inspected.serialized_bytes,
            post_compaction_tool_result_chars=inspected.tool_result_chars,
            system_context=render_summary_system_context(session.summary),
            pressure_reasons=pressure_reasons,
            token_count_source=inspected.token_count_source,
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
            post_compaction_estimated_tokens=estimated_tokens,
            serialized_bytes=inspected.serialized_bytes,
            tool_result_chars=inspected.tool_result_chars,
            post_compaction_serialized_bytes=inspected.serialized_bytes,
            post_compaction_tool_result_chars=inspected.tool_result_chars,
            system_context=render_summary_system_context(session.summary),
            pressure_reasons=pressure_reasons,
            token_count_source=inspected.token_count_source,
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
    session.metadata.pop(PROVIDER_USAGE_BASELINE_KEY, None)
    recent_visible_turns = extract_recent_visible_turns(
        session.messages[:next_covered_count],
        turn_limit=CHAT_RECENT_VISIBLE_TURNS,
    )
    active_tail = session.messages[next_covered_count:]
    prepared_messages = trim_to_valid_context_boundary(recent_visible_turns + active_tail)
    post_serialized_bytes, post_tool_result_chars = message_pressure_metrics(
        prepared_messages,
        summary=summary,
    )
    # A provider usage baseline describes the request before the summary was
    # replaced and is invalid after compaction.  The post value is therefore a
    # fresh, conservative working-set estimate including the same framing and
    # output reserve used by preflight.
    post_estimated_tokens = (
        estimate_messages_tokens(prepared_messages)
        + estimate_context_tokens(summary)
        + max(0, int(reserved_tokens))
    )
    resolved_pressure_reasons = pressure_reasons or ("forced",)
    session.compaction_events.append(
        {
            "id": f"compact-{int(time.time())}-{next_covered_count}",
            "from_message_index": covered_count,
            "to_message_index": next_covered_count,
            "summary_sha256": sha256(summary.encode("utf-8")).hexdigest(),
            "episode_count": len(session.recall_episodes),
            "trigger_reasons": list(resolved_pressure_reasons),
            "before_estimated_tokens": estimated_tokens,
            "token_count_source": inspected.token_count_source,
            "before_serialized_bytes": inspected.serialized_bytes,
            "before_tool_result_chars": inspected.tool_result_chars,
            "token_trigger": profile_context_trigger_tokens(profile),
            "serialized_bytes_trigger": CHAT_SUMMARY_TRIGGER_SERIALIZED_BYTES,
            "tool_result_chars_trigger": CHAT_SUMMARY_TRIGGER_TOOL_RESULT_CHARS,
            "after_estimated_tokens": post_estimated_tokens,
            "after_serialized_bytes": post_serialized_bytes,
            "after_tool_result_chars": post_tool_result_chars,
            "created_at": int(time.time()),
        }
    )
    session.compaction_events = session.compaction_events[-64:]

    return PreparedSessionMemory(
        messages=prepared_messages,
        summary=summary,
        summary_message_count=next_covered_count,
        compacted=True,
        estimated_tokens=estimated_tokens,
        post_compaction_estimated_tokens=post_estimated_tokens,
        serialized_bytes=inspected.serialized_bytes,
        tool_result_chars=inspected.tool_result_chars,
        post_compaction_serialized_bytes=post_serialized_bytes,
        post_compaction_tool_result_chars=post_tool_result_chars,
        system_context=render_summary_system_context(summary),
        pressure_reasons=resolved_pressure_reasons,
        token_count_source=inspected.token_count_source,
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
                "8. 附件图像的像素会在本次压缩后移出模型上下文。只有当当前任务后续"
                "明确需要重新打开某张图时，才在摘要中保留其精确路径及用途；"
                "与续作无关的图片路径应删除，不得用 OCR 文本冒充已保留原图。\n"
                "9. 使用紧凑 Markdown 条目，不写寒暄、修辞、思维链或重复内容。\n\n"
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
            response = client.chat(
                messages,
                profile=profile,
                max_tokens=min(
                    CHAT_SUMMARY_MAX_TOKENS,
                    compaction_output_token_budget(profile),
                ),
                reasoning_effort="light",
            )
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
                    max_tokens=min(
                        CHAT_SUMMARY_MAX_TOKENS,
                        compaction_output_token_budget(profile),
                    ),
                    reasoning_effort="light",
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
            max_tokens=min(
                ACTIVE_REACT_CHECKPOINT_MAX_TOKENS,
                compaction_output_token_budget(profile),
            ),
            reasoning_effort="light",
        )
    except Exception as error:
        raise ContextCompactionError(
            f"当前模型压缩运行检查点失败：{type(error).__name__}: {error}。"
        ) from error
    checkpoint = str(response.content or "").strip()
    if not checkpoint:
        raise ContextCompactionError(
            "当前模型没有返回可用的运行检查点，不能安全继续本轮长任务。"
            + empty_checkpoint_response_diagnostics(getattr(response, "raw", None))
        )
    return checkpoint


def empty_checkpoint_response_diagnostics(raw: Any) -> str:
    """Describe an empty compaction response without persisting private reasoning."""

    if not isinstance(raw, dict):
        return "响应未包含可诊断的原始结构。"
    choices = raw.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    reasoning_chars = 0
    reasoning_fields: list[str] = []
    for key in (
        "reasoning_content",
        "reasoning",
        "reasoning_summary",
        "reasoning_details",
        "thinking",
        "thought",
    ):
        value = message.get(key)
        if value in (None, "", [], {}):
            continue
        reasoning_fields.append(key)
        try:
            reasoning_chars += len(json.dumps(value, ensure_ascii=False))
        except TypeError:
            reasoning_chars += len(str(value))
    finish_reason = str(choice.get("finish_reason") or "unknown")
    message_keys = ",".join(sorted(str(key) for key in message)) or "none"
    return (
        f"响应诊断：finish_reason={finish_reason}，reasoning_chars={reasoning_chars}，"
        f"reasoning_fields={','.join(reasoning_fields) or 'none'}，message_keys={message_keys}。"
    )


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
    while current_turn_start > 0 and is_turn_runtime_context_message(
        messages[current_turn_start - 1]
    ):
        current_turn_start -= 1
    if turn_has_final_answer(messages[current_turn_start:]):
        return len(messages)
    return current_turn_start


def turn_has_final_answer(messages: list[dict[str, Any]]) -> bool:
    for message in reversed(messages):
        role = message.get("role")
        if role in {"tool", "system"}:
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
    pending_turn_contexts: list[dict[str, Any]] = []
    for message in messages:
        if is_turn_runtime_context_message(message):
            pending_turn_contexts.append(message)
            continue
        if message.get("role") == "user":
            if current:
                turns.append(current)
            current = [*pending_turn_contexts, message]
            pending_turn_contexts = []
        elif current:
            current.append(message)
    if current:
        turns.append(current)

    visible: list[dict[str, Any]] = []
    completed = [turn for turn in turns if turn_has_final_answer(turn)]
    for turn in completed[-max(0, turn_limit):]:
        runtime_contexts = [message for message in turn if is_turn_runtime_context_message(message)]
        user = next(message for message in turn if message.get("role") == "user")
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
        visible.extend(runtime_contexts)
        compacted_user_content = strip_compacted_attachment_block(
            str(user.get("content") or "")
        )
        visible.append({
            "role": "user",
            "content": compacted_user_content or "（该轮图片附件已随上下文压缩移除）",
        })
        visible.append({"role": "assistant", "content": str(final.get("content") or "")})
    return visible


def strip_compacted_attachment_block(text: str) -> str:
    """Remove covered attachment references from the live model window.

    The durable transcript remains untouched. The generated compaction summary
    alone decides whether a path remains relevant; covered image pixels are
    never silently resurrected into a later request.
    """

    return re.sub(r"\n*参考附件：[\s\S]*$", "", str(text or "")).strip()


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
    # JSON and English average closer to four characters per token.  This is
    # only used for the delta added after the last provider-reported usage.
    cjk_chars = len(re.findall(r"[\u3400-\u9fff\uf900-\ufaff]", text))
    other_chars = max(0, len(text) - cjk_chars)
    return max(1, cjk_chars + math.ceil(other_chars / 4))


def clip_text(text: str, limit: int) -> str:
    value = str(text or "")
    if len(value) <= limit:
        return value
    return value[:limit] + f"\n…[truncated {len(value) - limit} chars]"
