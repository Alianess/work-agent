from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator
import html
import json
import queue
import re
import threading
import time
import traceback
from pathlib import Path

from .approval_review import ApprovalReview, ApprovalReviewer
from .artifact_ledger import ToolArtifactCollector
from .config import ModelProfile
from .debug_trace import compact_message_summary
from .llm import (
    Message,
    OpenAICompatibleClient,
    build_chat_tools_payload,
    normalize_reasoning_effort,
    recovery_request_timeout_seconds,
    stream_start_timeout_seconds,
)
from .memory import (
    ContextCompactionError,
    estimate_context_tokens,
    estimate_messages_tokens,
    message_pressure_metrics,
    profile_context_trigger_tokens,
    summarize_active_react_checkpoint,
    token_count_source_label,
)
from .session_log import ASSISTANT_MESSAGE, TURN_END_ABORTED, TURN_END_COMPLETED, TURN_END_FAILED
from .session_runtime import ConversationRuntime
from .progress import (
    compact_preview_text,
    set_tool_attachment_sink,
    set_tool_cancel_check,
    set_tool_progress_sink,
)
from .session_store import repair_runtime_message_sequence
from .shell_tools import issue_internal_approval_grant
from .tool_bus import ToolBus


DEFAULT_MAX_STEPS = 50
MODEL_STREAM_IDLE_TIMEOUT_SECONDS = 30
MODEL_STREAM_MAX_MULTIPLIER = 4
MODEL_STREAM_MAX_EXTENSION_SECONDS = 60
MODEL_RECOVERY_TIMEOUT_GRACE_SECONDS = 5
MODEL_RECOVERY_MAX_MULTIPLIER = 4
MAX_CONSECUTIVE_TOOL_LENGTH_TRUNCATIONS = 3
MAX_CONSECUTIVE_INVALID_TOOL_ARGUMENTS = 3
MODEL_STREAM_REPETITION_WINDOW_CHARS = 1200
ACTIVE_REACT_CHECKPOINT_SERIALIZED_BYTES = 4 * 1024 * 1024
ACTIVE_REACT_CHECKPOINT_TOOL_RESULT_CHARS = 4 * 1024 * 1024


class AgentCancelled(RuntimeError):
    """Raised when the current single-agent turn is cancelled by runtime state."""


class ModelStreamLoopStopped(RuntimeError):
    """The current model request was stopped after entering a long output loop."""


def merge_system_messages_at_start(messages: list[Message]) -> list[Message]:
    """Return a provider-safe projection with one leading system message.

    Some local chat templates require every system instruction at the very
    beginning. Merge their text in original order and leave non-system history
    messages untouched, including tool-call protocol messages.
    """

    system_parts = [
        str(message.get("content") or "").strip()
        for message in messages
        if message.get("role") == "system" and str(message.get("content") or "").strip()
    ]
    if len(system_parts) <= 1:
        return messages
    return [
        {"role": "system", "content": "\n\n".join(system_parts)},
        *(dict(message) for message in messages if message.get("role") != "system"),
    ]


def repeated_stream_span(text: str, *, window_chars: int = MODEL_STREAM_REPETITION_WINDOW_CHARS) -> str:
    """Return a repeated long suffix, or an empty string for ordinary streaming text.

    Heartbeats prove that a transport is alive, but not that generation is making
    progress. An exact, whitespace-normalized 1200-character replay is a
    deliberately conservative signal: it catches decoding loops without treating
    short rhetorical repetition or a recurring heading as a stuck model.
    """
    normalized = re.sub(r"\s+", " ", str(text or "")).strip().casefold()
    span_size = max(1, int(window_chars))
    if len(normalized) < span_size * 2:
        return ""
    suffix = normalized[-span_size:]
    return suffix if suffix in normalized[:-span_size] else ""


@dataclass(frozen=True)
class AgentResult:
    final: str
    steps_used: int
    model_profile: str
    used_tools: bool = False
    messages: list[Message] | None = None
    # Append-origin history: what actually happened, before any compaction
    # replacement shadowed part of it for the model.
    transcript: list[Message] | None = None
    artifacts: list[dict[str, Any]] | None = None


@dataclass(frozen=True)
class ActiveRuntimePressure:
    estimated_tokens: int
    token_count_source: str
    raw_estimated_tokens: int
    serialized_request_bytes: int
    tool_result_chars: int
    context_trigger_tokens: int
    reasons: tuple[str, ...]


@dataclass
class LoopHooks:
    """Where behaviour attaches to the loop instead of being welded into it.

    Hooks are generators so they can emit activity events the same way the loop
    does; ``yield from`` composes them without a separate event channel. A hook
    that raises is treated as absent — extending the loop must not be able to
    end a turn.
    """

    transform_context: Callable[..., Iterator[dict[str, Any]]] | None = None
    """Called before each model request, after built-in compaction.

    Free to append context to the runtime; the request is rebuilt afterwards.
    This is the mounting point for long-term memory injection.
    """

    should_stop_after_turn: Callable[..., bool] | None = None
    """Called after a step's tool batch. Returning True ends the turn cleanly."""

    before_tool_call: Callable[..., Iterator[dict[str, Any]]] | None = None
    """Called before a tool runs.

    Return ``{"block": True, "observation": "..."}`` to skip execution and hand
    the model that text instead. Approval does not run through this yet — see
    the note in TODO.md — but a gate no longer has to be welded into the loop.
    """

    after_tool_call: Callable[..., Iterator[dict[str, Any]]] | None = None
    """Called once a tool result is settled.

    Return ``{"observation": "..."}`` to replace what the model sees.
    """


WORKSPACE_CONTEXT_FILENAME = "AGENTS.md"
WORKSPACE_CONTEXT_MAX_CHARS = 4000
_WORKSPACE_CONTEXT_CACHE: dict[str, tuple[float, str]] = {}


def read_workspace_context(workspace_root: Any) -> str:
    """Read this workspace's own conventions, if it states any.

    Bounded and cached by mtime: it is injected on every request, so an
    unbounded file would quietly become the largest thing in the prompt.
    """

    if not workspace_root:
        return ""
    path = Path(workspace_root) / WORKSPACE_CONTEXT_FILENAME
    try:
        stamp = path.stat().st_mtime
    except OSError:
        return ""
    cached = _WORKSPACE_CONTEXT_CACHE.get(str(path))
    if cached is not None and cached[0] == stamp:
        return cached[1]
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    if len(text) > WORKSPACE_CONTEXT_MAX_CHARS:
        text = (
            text[:WORKSPACE_CONTEXT_MAX_CHARS].rstrip()
            + f"\n\n[{WORKSPACE_CONTEXT_FILENAME} 超出 {WORKSPACE_CONTEXT_MAX_CHARS} 字符，"
            f"其余部分未载入；需要时直接读取该文件。]"
        )
    _WORKSPACE_CONTEXT_CACHE[str(path)] = (stamp, text)
    return text


def _latest_user_content(runtime: "ConversationRuntime") -> str:
    for message in reversed(runtime.log.derive_messages()):
        if message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


def as_conversation_runtime(
    conversation: "list[Message] | ConversationRuntime",
) -> ConversationRuntime:
    """Accept either a live conversation or a plain message list.

    Callers that still hold flat history get a runtime seeded from it, so there
    is one loop implementation rather than one per calling convention.
    """

    if isinstance(conversation, ConversationRuntime):
        return conversation
    return ConversationRuntime.from_messages(conversation)


class ReActAgent:
    def __init__(
        self,
        *,
        client: OpenAICompatibleClient,
        profile: ModelProfile,
        tools: ToolBus,
        max_steps: int = DEFAULT_MAX_STEPS,
        system_prompt: str | None = None,
        extra_system_context: str | None = None,
        debug_trace: Any | None = None,
        cancel_check: Callable[[], bool] | None = None,
        pending_messages: Callable[[], list[str]] | None = None,
        pending_messages_committed: Callable[[list[str]], None] | None = None,
        request_transform: Callable[[list[Message]], list[Message]] | None = None,
        hooks: LoopHooks | None = None,
        workspace_root: str | Path | None = None,
        reasoning_effort: str = "medium",
        auto_approve: bool = True,
        approval_reviewer: ApprovalReviewer | None = None,
        plan_update_callback: Callable[[list[dict[str, str]], str], None] | None = None,
        usage_callback: Callable[[dict[str, Any], list[Message]], None] | None = None,
        initial_task_plan: list[dict[str, str]] | None = None,
        late_task_plan_context: bool = True,
    ) -> None:
        self.client = client
        self.profile = profile
        self.tools = tools
        self.max_steps = max_steps
        self.extra_system_context = (extra_system_context or "").strip()
        # Set before the prompt is built: the workspace's own conventions are
        # part of it.
        self.workspace_root = Path(workspace_root) if workspace_root else None
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.debug_trace = debug_trace
        self.cancel_check = cancel_check
        # Interrupting is not the only thing a person may want mid-run. Draining
        # a queue lets the user add to the work instead of killing it, which is
        # what makes an early wrap-up recoverable without restarting the turn.
        self.pending_messages = pending_messages
        self.pending_messages_committed = pending_messages_committed
        # 只在装配请求那一刻改写消息，改写结果不回写日志。图片附件走这条路：
        # 日志里留人可读的路径，base64 只活在这一次请求里。
        self.request_transform = request_transform
        self.hooks = hooks or LoopHooks()
        # 工具在本轮交回来的多模态内容块，等着被注入成一条用户消息。
        self._tool_attachments: list[dict[str, Any]] = []
        # Structured product state collected from tool progress and results.
        # Final prose may mention a file, but prose is never the source of truth
        # for whether an artifact exists or can be delivered.
        self.artifact_collector = ToolArtifactCollector(self.workspace_root)
        self.reasoning_effort = normalize_reasoning_effort(reasoning_effort)
        # The runtime policy is caller-controlled: when disabled, reviewable
        # actions go straight to the human approval card instead of the model
        # reviewer.
        self.auto_approve = bool(auto_approve)
        self.approval_reviewer = approval_reviewer or ApprovalReviewer(
            client=client,
            profile=profile,
        )
        self.plan_update_callback = plan_update_callback
        self.usage_callback = usage_callback
        self.active_runtime_was_compacted = False
        # Provider usage is authoritative for the request that just completed.
        # Keep it as an in-turn anchor; only messages added afterwards need a
        # local delta estimate before the next request is sent.
        self._active_provider_prompt_tokens = 0
        self._active_provider_anchor_estimated_tokens = 0
        self.task_plan = [
            {"step": str(item.get("step") or ""), "status": str(item.get("status") or "pending")}
            for item in (initial_task_plan or [])
            if isinstance(item, dict) and str(item.get("step") or "").strip()
        ]
        self.late_task_plan_context = bool(late_task_plan_context)

    def run(self, goal: str) -> AgentResult:
        return self.run_messages([{"role": "user", "content": goal}])

    def run_messages(
        self,
        conversation: list[Message] | ConversationRuntime,
        *,
        system_context: str = "",
    ) -> AgentResult:
        """Run one turn without streaming, over the same loop as the UI path.

        There is exactly one place with loop logic. A second implementation had
        already drifted — it never grew the local compaction fallback the
        streaming path relies on — so the non-streaming entry point is now a
        projection of the streamed event sequence rather than a copy of it.
        """

        runtime = as_conversation_runtime(conversation)
        final = ""
        steps_used = 0
        used_tools = False
        for event in self.iter_message_events(runtime, system_context=system_context):
            kind = event.get("event")
            if kind == "final":
                final = str(event.get("content") or "")
                steps_used = int(event.get("steps_used") or 0)
                used_tools = bool(event.get("used_tools"))
            elif kind == "error":
                raise RuntimeError(str(event.get("message") or "agent failed"))
        return AgentResult(
            final=final,
            steps_used=steps_used,
            model_profile=self.profile.name,
            used_tools=used_tools,
            messages=runtime.log.derive_messages(),
            transcript=runtime.log.derive_transcript(),
            artifacts=runtime.log.derive_artifacts(),
        )

    def iter_events(self, goal: str) -> Iterator[dict[str, Any]]:
        yield from self.iter_message_events([{"role": "user", "content": goal}])

    def iter_message_events(
        self,
        conversation: list[Message] | ConversationRuntime,
        *,
        system_context: str = "",
    ) -> Iterator[dict[str, Any]]:
        runtime = as_conversation_runtime(conversation)
        self.active_runtime_was_compacted = False
        self._active_provider_prompt_tokens = 0
        self._active_provider_anchor_estimated_tokens = 0
        self.artifact_collector.reset()
        messages: list[Message] = self._request_messages(runtime, system_context=system_context)
        tool_schemas = self._tool_schemas()
        used_tools = False
        visible_content_parts: list[str] = []
        last_truncation_signature: tuple[str, ...] = ()
        consecutive_truncations = 0
        last_invalid_arguments_signature: tuple[str, ...] = ()
        consecutive_invalid_arguments = 0
        self._trace(
            "agent_start",
            mode="stream",
            session_message_count=runtime.log.seq,
            model_message_count=len(messages),
            tool_schema_count=len(tool_schemas),
            messages=compact_message_summary(runtime.log.derive_messages()[-12:]),
        )

        yield {
            "event": "activity",
            "phase": "thinking",
            "title": "分析任务",
            "detail": "已进入 tool calling 模式：模型可直接用 content 回答，也可返回 tool_calls 调用工具。",
            "step": 0,
        }

        for step in range(1, self.max_steps + 1):
            self._raise_if_cancelled()
            messages = self._request_messages(runtime, system_context=system_context)
            pressure = self._active_runtime_pressure(
                messages,
                tool_schemas=tool_schemas,
            )
            if pressure.reasons and len(runtime.active_turn_surface_seqs()) >= 2:
                source_label = token_count_source_label(pressure.token_count_source)
                reason_text = "、".join(
                    {
                        "tokens": "token 达到 85% 安全线",
                        "serialized_bytes": "请求体达到进程保护线",
                        "tool_results": "工具结果达到进程保护线",
                    }.get(reason, reason)
                    for reason in pressure.reasons
                )
                self._trace(
                    "active_runtime_compaction_started",
                    step=step,
                    estimated_tokens=pressure.estimated_tokens,
                    token_count_source=pressure.token_count_source,
                    raw_estimated_tokens=pressure.raw_estimated_tokens,
                    serialized_request_bytes=pressure.serialized_request_bytes,
                    tool_result_chars=pressure.tool_result_chars,
                    context_trigger_tokens=pressure.context_trigger_tokens,
                    pressure_reasons=list(pressure.reasons),
                )
                yield {
                    "event": "activity",
                    "phase": "thinking",
                    "title": "正在压缩运行上下文",
                    "detail": (
                        f"触发项：{reason_text}；当前约 {pressure.estimated_tokens:,}/"
                        f"{pressure.context_trigger_tokens:,} tokens；依据：{source_label}。"
                        "正在生成续作检查点，原始轨迹不会删除。"
                    ),
                    "activity_type": "runtime_compaction_started",
                    "step": step,
                }
            checkpoint_event = None
            try:
                checkpoint_event = self._maybe_compact_active_runtime(
                    runtime,
                    messages,
                    tool_schemas=tool_schemas,
                    step=step,
                    pressure=pressure,
                )
            except ContextCompactionError as error:
                self._trace(
                    "active_runtime_compaction_failed",
                    step=step,
                    error=str(error),
                    fallback="none",
                    traceback=traceback.format_exc().splitlines()[-16:],
                )
                yield {
                    "event": "activity",
                    "phase": "error",
                    "title": "运行上下文整理失败",
                    "detail": (
                        f"{error} 完整原始轨迹仍保留；本轮已停止，"
                        "不会改写工作上下文、切换模型或隐式重试。"
                    ),
                    "activity_type": "runtime_compaction_failed",
                    "command_status": "error",
                    "step": step,
                }
                raise
            if checkpoint_event is not None:
                yield checkpoint_event
                messages = self._request_messages(runtime, system_context=system_context)
            if self.hooks.transform_context is not None:
                try:
                    yield from self.hooks.transform_context(runtime, step=step)
                except Exception:
                    self._trace(
                        "transform_context_hook_failed",
                        step=step,
                        traceback=traceback.format_exc().splitlines()[-8:],
                    )
                else:
                    messages = self._request_messages(runtime, system_context=system_context)
            # Record the non-history half of the request before dispatching it.
            # History is already the log; the prompt, the late blocks and the
            # route exist only at request time and are otherwise unrecoverable.
            runtime.record_request_header(
                system_prompt=self.system_prompt,
                late_blocks=self._late_system_blocks(system_context),
                tool_names=[
                    str((schema.get("function") or {}).get("name") or "")
                    for schema in tool_schemas
                ],
                profile=self.profile.name,
                model=self.profile.model,
                endpoint=str(getattr(self.profile, "base_url", "") or ""),
                step=step,
                reason="initial" if step == 1 else "change",
                reasoning_effort=self.reasoning_effort,
                max_tokens=self.profile.max_tokens,
            )
            try:
                response = yield from self._plan_with_progress(
                    messages=messages,
                    tool_schemas=tool_schemas,
                    step=step,
                    draft_prefix=visible_react_draft_prefix(visible_content_parts),
                )
                self._record_response_usage(
                    response.raw,
                    runtime,
                    request_messages=messages,
                )
            except AgentCancelled:
                self._trace("agent_cancelled", step=step)
                raise
            except ModelStreamLoopStopped as error:
                # This is a deliberate safety stop, not an unexpected Python
                # failure. Keep the technical trace in debug telemetry and send
                # the UI a recoverable state with a clear next action.
                yield {
                    "event": "error",
                    "message": str(error),
                    "type": type(error).__name__,
                    "detail": str(error),
                    "error_code": "model_loop_stopped",
                    "recoverable": True,
                    "suggested_action": "continue",
                }
                return
            except Exception as error:
                trace_lines = traceback.format_exc().splitlines()
                yield {
                    "event": "error",
                    "message": str(error),
                    "type": type(error).__name__,
                    "detail": str(error),
                    "trace": trace_lines[-12:],
                }
                return
            assistant_message = response_message(response.raw)
            tool_calls = normalize_tool_calls(assistant_message)

            finish_reason = response_finish_reason(response.raw)
            if tool_calls and finish_reason == "length":
                signature = tuple(call.name for call in tool_calls)
                consecutive_truncations = (
                    consecutive_truncations + 1
                    if signature == last_truncation_signature
                    else 1
                )
                last_truncation_signature = signature
                used_tools = True
                assistant_history, tool_messages = truncated_tool_call_messages(
                    tool_calls,
                    max_tokens=self.profile.max_tokens,
                    attempt=consecutive_truncations,
                )
                runtime.append_message(assistant_history)
                for _item in tool_messages:
                    runtime.append_message(_item)
                self._trace(
                    "tool_calls_truncated",
                    step=step,
                    finish_reason=finish_reason,
                    max_tokens=self.profile.max_tokens,
                    consecutive_count=consecutive_truncations,
                    tool_names=list(signature),
                )
                detail = str(tool_messages[0].get("content") or "")
                yield {
                    "event": "activity",
                    "phase": "error",
                    "title": "工具参数被输出上限截断",
                    "detail": detail,
                    "activity_type": "tool_arguments_truncated",
                    "command_status": "error",
                    "step": step,
                }
                if consecutive_truncations >= MAX_CONSECUTIVE_TOOL_LENGTH_TRUNCATIONS:
                    final = repeated_tool_truncation_final(
                        signature,
                        max_tokens=self.profile.max_tokens,
                        attempts=consecutive_truncations,
                    )
                    final_message: Message = {"role": "assistant", "content": final}
                    runtime.append_message(final_message)
                    display_final = merge_visible_react_content(visible_content_parts, final)
                    self._trace(
                        "tool_truncation_circuit_open",
                        step=step,
                        attempts=consecutive_truncations,
                        tool_names=list(signature),
                    )
                    yield {
                        "event": "final",
                        "content": display_final,
                        "steps_used": step,
                        "model_profile": self.profile.name,
                        "used_tools": True,
                        "tool_truncation_circuit_open": True,
                    }
                    return
                continue
            last_truncation_signature = ()
            consecutive_truncations = 0

            invalid_argument_calls = [
                tool_call for tool_call in tool_calls if tool_call.arguments_error
            ]
            if invalid_argument_calls:
                signature = tuple(call.name for call in tool_calls)
                consecutive_invalid_arguments = (
                    consecutive_invalid_arguments + 1
                    if signature == last_invalid_arguments_signature
                    else 1
                )
                last_invalid_arguments_signature = signature
                used_tools = True
                assistant_history, tool_messages = invalid_tool_call_messages(
                    tool_calls,
                    attempt=consecutive_invalid_arguments,
                )
                runtime.append_message(assistant_history)
                for tool_message in tool_messages:
                    runtime.append_message(tool_message)
                invalid_names = [call.name for call in invalid_argument_calls]
                self._trace(
                    "tool_arguments_invalid",
                    step=step,
                    consecutive_count=consecutive_invalid_arguments,
                    tool_names=list(signature),
                    invalid_tool_names=invalid_names,
                    errors=[call.arguments_error for call in invalid_argument_calls],
                )
                detail = str(tool_messages[0].get("content") or "")
                yield {
                    "event": "activity",
                    "phase": "error",
                    "title": "工具参数不是有效 JSON",
                    "detail": detail,
                    "activity_type": "tool_arguments_invalid",
                    "command_status": "error",
                    "step": step,
                }
                if consecutive_invalid_arguments >= MAX_CONSECUTIVE_INVALID_TOOL_ARGUMENTS:
                    final = repeated_invalid_tool_arguments_final(
                        signature,
                        attempts=consecutive_invalid_arguments,
                    )
                    final_message: Message = {"role": "assistant", "content": final}
                    runtime.append_message(final_message)
                    display_final = merge_visible_react_content(visible_content_parts, final)
                    self._trace(
                        "invalid_tool_arguments_circuit_open",
                        step=step,
                        attempts=consecutive_invalid_arguments,
                        tool_names=list(signature),
                    )
                    yield {
                        "event": "final",
                        "content": display_final,
                        "steps_used": step,
                        "model_profile": self.profile.name,
                        "used_tools": True,
                        "invalid_tool_arguments_circuit_open": True,
                    }
                    return
                continue
            last_invalid_arguments_signature = ()
            consecutive_invalid_arguments = 0

            # ReAct state transition is determined only by whether a tool call
            # can be parsed from this assistant message. Content may contain
            # user-visible preamble before tool calls, so content presence is
            # never an end signal.
            if not tool_calls:
                raw_content = str(assistant_message.get("content") or response.content or "")
                if contains_tool_call_markup(raw_content):
                    yield {
                        "event": "activity",
                        "phase": "thinking",
                        "title": "兼容工具调用格式",
                        "detail": "模型返回了无法解析的文本工具标签，已隐藏原始内容并要求模型改用原生 tool calling。",
                        "step": step,
                    }
                    assistant_fix, user_fix = text_tool_call_repair_messages()
                    for _fix in (assistant_fix, user_fix):
                        runtime.record_context_injection(
                            str(_fix.get('content') or ''),
                            kind='tool_call_repair',
                            role=str(_fix.get('role') or 'system'),
                        )
                    continue
                final = raw_content.strip()
                if not final:
                    yield {
                        "event": "error",
                        "message": empty_model_response_message(self.profile.name),
                        "type": "EmptyModelResponse",
                        "detail": "模型请求已经结束，但没有可写入对话气泡的正文或工具调用。",
                    }
                    return
                final_message = assistant_message_for_history(assistant_message)
                final_message["content"] = final
                runtime.append_message(final_message)
                # The agent would stop here. Anything the user queued while it
                # was working is a reason to keep going instead — no prompt
                # needs to warn the model against wrapping up early when a
                # person can simply add the next sentence.
                follow_up = self._drain_pending_messages()
                if follow_up:
                    visible_content_parts.append(final)
                    yield from self._inject_pending_messages(
                        runtime, follow_up, step=step, title="用户追加了消息，继续本轮"
                    )
                    continue
                display_final = merge_visible_react_content(visible_content_parts, final)
                self._trace(
                    "agent_final",
                    step=step,
                    used_tools=used_tools,
                    final_chars=len(display_final),
                )
                yield {
                    "event": "activity",
                    "phase": "complete",
                    "title": f"已完成 {step} 轮",
                    "detail": "本轮没有解析到 tool_calls，按最终回复结束。",
                    "step": step,
                }
                yield {
                    "event": "final",
                    "content": display_final,
                    "steps_used": step,
                    "model_profile": self.profile.name,
                    "used_tools": used_tools,
                }
                return

            raw_visible_text = str(assistant_message.get("content") or response.content or "")
            visible_text = assistant_visible_content(assistant_message).strip()
            if visible_text:
                if contains_tool_call_markup(raw_visible_text):
                    yield {
                        "event": "draft_delta",
                        "content": visible_text,
                        "step": step,
                    }
                visible_content_parts.append(visible_text)
                yield {
                    "event": "draft_delta",
                    "content": "\n\n",
                    "step": step,
                }
                yield {
                    "event": "activity",
                    "phase": "thinking",
                    "title": "实施路径",
                    "detail": visible_text,
                    "activity_type": "work_note",
                    "step": step,
                }
            # An append-only log has no rollback: a batch paused for approval
            # keeps its recorded prefix, and resume continues from the exact
            # call that needed the grant.
            completed_tool_messages: list[Message] = []
            deterministic_final: str | None = None
            assistant_history = assistant_message_for_history(assistant_message, tool_calls=tool_calls)
            runtime.append_message(assistant_history)
            for index, tool_call in enumerate(tool_calls):
                self._raise_if_cancelled()
                used_tools = True
                tool_name = tool_call.name
                tool_input = tool_call.arguments
                tool_activity_id = f"tool-{step}-{index}-{tool_call.id or tool_name}"
                yield {
                    "event": "activity",
                    "phase": "thinking",
                    "title": f"准备调用 {tool_name}",
                    "detail": (
                        "模型通过原生 tool calling 选择了这个工具。"
                        if has_native_tool_calls(assistant_message)
                        else "模型返回了文本形式的工具调用，系统已兼容解析并隐藏原始标签。"
                    ),
                    "step": step,
                    "tool_name": tool_name,
                }
                yield {
                    "event": "activity",
                    "id": tool_activity_id,
                    "phase": "action",
                    "title": f"执行工具：{tool_name}",
                    "detail": summarize_tool_input(tool_input),
                    "step": step,
                    "tool_name": tool_name,
                }

                gate = yield from self._run_tool_hook(
                    self.hooks.before_tool_call,
                    name="before_tool_call",
                    runtime=runtime,
                    step=step,
                    tool_name=tool_name,
                    tool_input=tool_input,
                )
                if gate is not None and gate.get("block"):
                    observation = str(
                        gate.get("observation") or f"{tool_name} 被运行时策略拦下，未执行。"
                    )
                else:
                    observation = yield from self._execute_tool_with_progress(
                        tool_name,
                        tool_input,
                        step,
                        tool_call_id=tool_call.id or f"call_{step}_{index}",
                    )
                self._raise_if_cancelled()
                approval_payload = parse_shell_approval_required_observation(tool_name, observation)
                review: ApprovalReview | None = None
                if (
                    approval_payload is not None
                    and self.auto_approve
                    and approval_payload.get("reviewable_by_model") is True
                ):
                    yield {
                        "event": "activity",
                        "phase": "thinking",
                        "title": "安全策略正在审查",
                        "detail": "仅审查当前精确动作；固定安全边界不会交给模型改写。",
                        "content": str(approval_payload.get("preview") or ""),
                        "activity_type": "approval_review",
                        "command": str(approval_payload.get("command") or ""),
                        "risk_category": str(approval_payload.get("risk_category") or "EXECUTE"),
                        "step": step,
                        "tool_name": tool_name,
                    }
                    review = self._review_approval(runtime.log.derive_messages(), approval_payload, step=step)
                    approval_payload = approval_payload_with_review(approval_payload, review)
                    yield {
                        "event": "activity",
                        "phase": "action" if review.approved else ("error" if review.failed else "thinking"),
                        "title": "独立审查已批准" if review.approved else "独立审查未放行",
                        "detail": review.reason,
                        "content": str(approval_payload.get("preview") or ""),
                        "activity_type": "approval_review",
                        "command": str(approval_payload.get("command") or ""),
                        "command_status": "running" if review.approved else "approval_required",
                        "risk_category": str(approval_payload.get("risk_category") or "EXECUTE"),
                        "approval_resolved": review.approved,
                        "reviewer_profile": review.reviewer_profile,
                        "step": step,
                        "tool_name": tool_name,
                    }
                if approval_payload is not None and review is not None and review.approved:
                    yield {
                        "event": "activity",
                        "phase": "action",
                        "title": "执行审查已批准的动作",
                        "detail": (
                            f"{approval_payload.get('risk_category') or 'EXECUTE'} · "
                            f"{review.reason}"
                        ),
                        "content": str(approval_payload.get("preview") or ""),
                        "activity_type": "command",
                        "command": str(approval_payload.get("command") or ""),
                        "command_status": "running",
                        "risk_category": str(approval_payload.get("risk_category") or "EXECUTE"),
                        "approval_resolved": True,
                        "step": step,
                        "tool_name": tool_name,
                    }
                    self._trace(
                        "approval_auto_approved",
                        step=step,
                        tool_name=tool_name,
                        command=approval_payload.get("command"),
                        risk_category=approval_payload.get("risk_category"),
                    )
                    observation = yield from self._execute_tool_with_progress(
                        tool_name,
                        approval_granted_tool_input(
                            tool_call,
                            approval_payload,
                            source="reviewer",
                        ),
                        step,
                        trusted_approval=True,
                        tool_call_id=tool_call.id or f"call_{step}_{index}",
                    )
                    self._raise_if_cancelled()
                    approval_payload = parse_shell_approval_required_observation(tool_name, observation)
                if approval_payload is not None:
                    pending_approval = pending_tool_batch_state(
                        runtime_messages_before_batch=[],
                        assistant_message=assistant_history,
                        tool_calls=tool_calls,
                        approval_index=index,
                        completed_tool_messages=completed_tool_messages,
                        step=step,
                        profile_name=self.profile.name,
                        model=self.profile.model,
                        max_steps=self.max_steps,
                        system_context=system_context,
                        extra_system_context=self.extra_system_context,
                        approval_payload=approval_payload,
                        reasoning_effort=self.reasoning_effort,
                        auto_approve=self.auto_approve,
                        visible_content_parts=visible_content_parts,
                    )
                    yield {
                        "event": "activity",
                        "phase": "action",
                        "title": "等待批次审批",
                        "detail": (
                            str(approval_payload.get("reason") or "该命令需要用户确认后才能执行。")
                            + " 确认后会继续执行同一批工具调用，不会让模型重新解释。"
                        ),
                        "content": str(approval_payload.get("preview") or ""),
                        "activity_type": "command",
                        "command": str(approval_payload.get("command") or ""),
                        "command_status": "approval_required",
                        "risk_category": str(approval_payload.get("risk_category") or "EXECUTE"),
                        "approval_required": True,
                        "approval_preview": str(approval_payload.get("preview") or ""),
                        "approval_batch_count": len(tool_calls),
                        "approval_batch_remaining": len(tool_calls) - index,
                        "approval_batch_commands": pending_approval.get("approval_batch_commands", []),
                        "step": step,
                        "tool_name": tool_name,
                    }
                    final = approval_required_final_text(
                        approval_payload,
                        batch_count=len(tool_calls),
                        batch_remaining=len(tool_calls) - index,
                    )
                    display_final = merge_visible_react_content(visible_content_parts, final)
                    self._trace(
                        "agent_waiting_approval",
                        step=step,
                        tool_name=tool_name,
                        command=approval_payload.get("command"),
                        batch_count=len(tool_calls),
                        batch_remaining=len(tool_calls) - index,
                    )
                    yield {
                        "event": "final",
                        "content": display_final,
                        "steps_used": step,
                        "model_profile": self.profile.name,
                        "used_tools": True,
                        "waiting_approval": True,
                        "pending_approval": pending_approval,
                    }
                    return

                observation_failed = tool_observation_failed(observation)
                if tool_name == "update_plan" and not observation_failed:
                    yield self._task_plan_activity(step)
                yield {
                    "event": "activity_delta",
                    "id": tool_activity_id,
                    "append_mode": "replace",
                    "phase": "error" if observation_failed else "observation",
                    "title": f"{tool_name} 执行失败" if observation_failed else f"{tool_name} 返回结果",
                    "content": "",
                    "detail": observation if observation_failed else truncate_text(observation, 360),
                    "input_summary": summarize_tool_input(tool_input),
                    "result_summary": truncate_text(observation, 360),
                    "command_status": "error" if observation_failed else "success",
                    "step": step,
                    "tool_name": tool_name,
                }

                revised = yield from self._run_tool_hook(
                    self.hooks.after_tool_call,
                    name="after_tool_call",
                    runtime=runtime,
                    step=step,
                    tool_name=tool_name,
                    tool_input=tool_input,
                    observation=observation,
                )
                if revised is not None and revised.get("observation") is not None:
                    observation = str(revised["observation"])

                tool_message: Message = {
                    "role": "tool",
                    "tool_call_id": tool_call.id or f"call_{step}_{index}",
                    "name": tool_name,
                    "content": observation,
                }
                runtime.append_message(tool_message)
                completed_tool_messages.append(tool_message)
                self.artifact_collector.record_tool_result(
                    runtime,
                    tool_name=tool_name,
                    tool_input=tool_input,
                    observation=observation,
                    step=step,
                )

                terminal_text = deterministic_tool_success_final(
                    tool_name,
                    tool_input,
                    observation,
                )
                if terminal_text and len(tool_calls) == 1:
                    deterministic_final = terminal_text

            if deterministic_final:
                final_message: Message = {"role": "assistant", "content": deterministic_final}
                runtime.append_message(final_message)
                display_final = merge_visible_react_content(visible_content_parts, deterministic_final)
                self._trace(
                    "agent_deterministic_final",
                    step=step,
                    used_tools=True,
                    final_chars=len(display_final),
                )
                yield {
                    "event": "activity",
                    "phase": "complete",
                    "title": "工作汇报已保存",
                    "detail": "保存工具已返回明确成功结果，无需再次请求模型组织收尾话术。",
                    "step": step,
                }
                yield {
                    "event": "final",
                    "content": display_final,
                    "steps_used": step,
                    "model_profile": self.profile.name,
                    "used_tools": True,
                    "deterministic_tool_final": True,
                }
                return

            # 工具交回来的图片在这里进上下文。tool 消息只能是字符串，所以
            # "看见"这件事只能由 harness 完成：作为一条用户消息注入，模型在
            # 下一次调用时才真正看到它。
            attachments = self._take_tool_attachments()
            if attachments:
                runtime.append_message(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "以下是刚才用 read_file 载入的图片。"},
                            *attachments,
                        ],
                    }
                )
                yield {
                    "event": "activity",
                    "phase": "observation",
                    "title": f"已载入 {len(attachments)} 张图片",
                    "detail": "图片作为用户消息进入上下文，模型在下一步看到它。",
                    "activity_type": "image_attached",
                    "step": step,
                }

            # Tools for this step are done. Anything the user typed while they
            # ran goes in before the next model call, so steering lands on the
            # next decision rather than after the turn is over.
            steering = self._drain_pending_messages()
            if steering:
                yield from self._inject_pending_messages(
                    runtime, steering, step=step, title="用户中途补充"
                )

            if self.hooks.should_stop_after_turn is not None:
                try:
                    stop_now = bool(
                        self.hooks.should_stop_after_turn(
                            runtime, step=step, used_tools=used_tools
                        )
                    )
                except Exception:
                    self._trace(
                        "should_stop_hook_failed",
                        step=step,
                        traceback=traceback.format_exc().splitlines()[-8:],
                    )
                    stop_now = False
                if stop_now:
                    final = "已按运行时要求在本轮结束。"
                    runtime.append_message({"role": "assistant", "content": final})
                    display_final = merge_visible_react_content(visible_content_parts, final)
                    self._trace("agent_stopped_by_hook", step=step)
                    yield {
                        "event": "final",
                        "content": display_final,
                        "steps_used": step,
                        "model_profile": self.profile.name,
                        "used_tools": used_tools,
                        "stopped_by_hook": True,
                    }
                    return

        final = f"Reached max ReAct steps ({self.max_steps}) without final answer."
        final_message: Message = {"role": "assistant", "content": final}
        runtime.append_message(final_message)
        display_final = merge_visible_react_content(visible_content_parts, final)
        self._trace("agent_max_steps", max_steps=self.max_steps, used_tools=used_tools)
        yield {
            "event": "activity",
            "phase": "complete",
            "title": "达到最大工具轮数",
            "detail": final,
            "step": self.max_steps,
        }
        yield {
            "event": "final",
            "content": display_final,
            "steps_used": self.max_steps,
            "model_profile": self.profile.name,
            "used_tools": used_tools,
        }

    def iter_approved_tool_batch_events(
        self,
        conversation: list[Message] | ConversationRuntime,
        pending_approval: dict[str, Any],
        *,
        system_context: str = "",
    ) -> Iterator[dict[str, Any]]:
        """Resume one exact approved action within a paused tool batch.

        A pending batch is a model transport detail, not an approval scope.  A
        user grant applies only to ``approval_index``.  If a later shell call
        needs approval, persist a new pending state for that exact call instead
        of silently replaying it under the prior grant.
        """

        assistant_history = pending_approval.get("assistant_message")
        if not isinstance(assistant_history, dict):
            raise ValueError("待审批批次缺少 assistant_message，无法恢复执行。")
        tool_calls = [
            native_tool_call_from_payload(item)
            for item in pending_approval.get("tool_calls", [])
            if isinstance(item, dict)
        ]
        if not tool_calls:
            raise ValueError("待审批批次缺少 tool_calls，无法恢复执行。")
        approval_index = max(0, min(int(pending_approval.get("approval_index") or 0), len(tool_calls) - 1))
        completed_tool_messages = [
            item
            for item in pending_approval.get("completed_tool_messages", [])
            if isinstance(item, dict) and item.get("role") == "tool"
        ]
        step = max(1, int(pending_approval.get("step") or 1))
        assistant_history = assistant_message_for_history(assistant_history, tool_calls=tool_calls)
        visible_content_parts = [
            str(item).strip()
            for item in pending_approval.get("visible_content_parts", [])
            if str(item).strip()
        ]
        if not visible_content_parts:
            legacy_visible_text = assistant_visible_content(assistant_history).strip()
            if legacy_visible_text:
                visible_content_parts.append(legacy_visible_text)

        runtime = as_conversation_runtime(conversation)
        # The paused batch was never rolled back, so its assistant message and
        # the results that already landed are still recorded. Replaying them
        # here would duplicate the call the grant was issued against.
        if runtime.log.latest(ASSISTANT_MESSAGE) is None:
            runtime.append_message(assistant_history)
            for tool_message in completed_tool_messages:
                runtime.append_message(tool_message)
        messages = self._request_messages(runtime, system_context=system_context)

        yield {
            "event": "draft_reset",
            "content": visible_react_draft_prefix(visible_content_parts),
            "step": step,
        }
        yield {
            "event": "activity",
            "phase": "action",
            "title": "终端审批已确认",
            "detail": "仅执行当前已确认的精确动作；后续命令如需权限，会单独再次确认。",
            "approval_resolved": True,
            "step": step,
        }

        for index, tool_call in enumerate(tool_calls[approval_index:], start=approval_index):
            self._raise_if_cancelled()
            tool_name = tool_call.name
            tool_input = dict(tool_call.arguments)
            tool_activity_id = f"tool-{step}-{index}-{tool_call.id or tool_name}"
            yield {
                "event": "activity",
                "phase": "thinking",
                "title": f"准备调用 {tool_name}",
                "detail": "继续执行已获批的同一批工具调用。",
                "step": step,
                "tool_name": tool_name,
            }
            yield {
                "event": "activity",
                "id": tool_activity_id,
                "phase": "action",
                "title": f"执行工具：{tool_name}",
                "detail": summarize_tool_input(tool_input),
                "step": step,
                "tool_name": tool_name,
            }
            if tool_name == "shell_exec" and index == approval_index:
                tool_input = approval_granted_tool_input(
                    tool_call,
                    pending_approval.get("approval_payload") or {},
                    source="user",
                )
            observation = yield from self._execute_tool_with_progress(
                tool_name,
                tool_input,
                step,
                trusted_approval=(tool_name == "shell_exec" and index == approval_index),
                tool_call_id=tool_call.id or f"call_{step}_{index}",
            )
            self._raise_if_cancelled()
            approval_payload = parse_shell_approval_required_observation(tool_name, observation)
            review: ApprovalReview | None = None
            if (
                approval_payload is not None
                and self.auto_approve
                and approval_payload.get("reviewable_by_model") is True
            ):
                yield {
                    "event": "activity",
                    "phase": "thinking",
                    "title": "安全策略正在审查",
                    "detail": "仅审查当前精确动作；固定安全边界不会交给模型改写。",
                    "activity_type": "approval_review",
                    "command": str(approval_payload.get("command") or ""),
                    "risk_category": str(approval_payload.get("risk_category") or "EXECUTE"),
                    "step": step,
                    "tool_name": tool_name,
                }
                review = self._review_approval(runtime.log.derive_messages(), approval_payload, step=step)
                approval_payload = approval_payload_with_review(approval_payload, review)
                yield {
                    "event": "activity",
                    "phase": "action" if review.approved else ("error" if review.failed else "thinking"),
                    "title": "独立审查已批准" if review.approved else "独立审查未放行",
                    "detail": review.reason,
                    "activity_type": "approval_review",
                    "command": str(approval_payload.get("command") or ""),
                    "command_status": "running" if review.approved else "approval_required",
                    "risk_category": str(approval_payload.get("risk_category") or "EXECUTE"),
                    "approval_resolved": review.approved,
                    "reviewer_profile": review.reviewer_profile,
                    "step": step,
                    "tool_name": tool_name,
                }
            if approval_payload is not None and review is not None and review.approved:
                tool_input = approval_granted_tool_input(
                    tool_call,
                    approval_payload,
                    source="reviewer",
                )
                observation = yield from self._execute_tool_with_progress(
                    tool_name,
                    tool_input,
                    step,
                    trusted_approval=True,
                    tool_call_id=tool_call.id or f"call_{step}_{index}",
                )
                self._raise_if_cancelled()
                approval_payload = parse_shell_approval_required_observation(tool_name, observation)
            if approval_payload is not None:
                next_pending = pending_tool_batch_state(
                    runtime_messages_before_batch=[],
                    assistant_message=assistant_history,
                    tool_calls=tool_calls,
                    approval_index=index,
                    completed_tool_messages=completed_tool_messages,
                    step=step,
                    profile_name=self.profile.name,
                    model=self.profile.model,
                    max_steps=self.max_steps,
                    system_context=system_context,
                    extra_system_context=self.extra_system_context,
                    approval_payload=approval_payload,
                    reasoning_effort=self.reasoning_effort,
                    auto_approve=self.auto_approve,
                    visible_content_parts=visible_content_parts,
                )
                yield {
                    "event": "activity",
                    "phase": "action",
                    "title": "等待单项审批",
                    "detail": (
                        str(approval_payload.get("reason") or "该命令需要用户确认后才能执行。")
                        + " 本次确认仅覆盖这一条命令。"
                    ),
                    "content": str(approval_payload.get("preview") or ""),
                    "activity_type": "command",
                    "command": str(approval_payload.get("command") or ""),
                    "command_status": "approval_required",
                    "risk_category": str(approval_payload.get("risk_category") or "EXECUTE"),
                    "approval_required": True,
                    "approval_preview": str(approval_payload.get("preview") or ""),
                    "approval_batch_count": len(tool_calls),
                    "approval_batch_remaining": len(tool_calls) - index,
                    "approval_batch_commands": next_pending.get("approval_batch_commands", []),
                    "step": step,
                    "tool_name": tool_name,
                }
                yield {
                    "event": "final",
                    "content": merge_visible_react_content(
                        visible_content_parts,
                        approval_required_final_text(
                            approval_payload,
                            batch_count=len(tool_calls),
                            batch_remaining=len(tool_calls) - index,
                        ),
                    ),
                    "steps_used": step,
                    "model_profile": self.profile.name,
                    "used_tools": True,
                    "waiting_approval": True,
                    "pending_approval": next_pending,
                }
                return
            observation_failed = tool_observation_failed(observation)
            yield {
                "event": "activity_delta",
                "id": tool_activity_id,
                "append_mode": "replace",
                "phase": "error" if observation_failed else "observation",
                "title": f"{tool_name} 执行失败" if observation_failed else f"{tool_name} 返回结果",
                "detail": observation if observation_failed else truncate_text(observation, 360),
                "input_summary": summarize_tool_input(tool_input),
                "result_summary": truncate_text(observation, 360),
                "command_status": "error" if observation_failed else "success",
                "step": step,
                "tool_name": tool_name,
            }
            tool_message: Message = {
                "role": "tool",
                "tool_call_id": tool_call.id or f"call_{step}_{index}",
                "name": tool_name,
                "content": observation,
            }
            runtime.append_message(tool_message)
            completed_tool_messages.append(tool_message)

        # The approved batch is now structurally complete: assistant(tool_calls)
        # is followed by one tool message per tool_call_id. Continue normal
        # ReAct from that state and force used_tools=true on the final event.
        runtime.end_step(step)
        for event in self.iter_message_events(runtime, system_context=system_context):
            if event.get("event") == "draft_reset":
                event["content"] = (
                    visible_react_draft_prefix(visible_content_parts)
                    + str(event.get("content") or "")
                )
            if event.get("event") == "final":
                event["used_tools"] = True
                event["content"] = merge_visible_react_content(
                    visible_content_parts,
                    str(event.get("content") or ""),
                )
                next_pending = event.get("pending_approval")
                if isinstance(next_pending, dict):
                    nested_parts = next_pending.get("visible_content_parts")
                    next_pending["visible_content_parts"] = [
                        *visible_content_parts,
                        *(
                            [str(item) for item in nested_parts]
                            if isinstance(nested_parts, list)
                            else []
                        ),
                    ]
            yield event

    def _default_system_prompt(self) -> str:
        """What the model needs from the harness, and nothing it can learn elsewhere.

        Every character here is paid on every request, so a rule only belongs in
        this text when it has nowhere closer to live. Rules about a tool live in
        that tool's description; rules about this workspace live in AGENTS.md;
        rules about a domain live in that domain's skill. Termination lives in
        the loop, which can simply be resumed.
        """

        return (
            "你是本地工作智能体。你可以使用工具读取/写入工作区文件，并调用已注册技能或 MCP 工具。\n"
            "工具定义只通过 API 的 tools 字段提供；你必须使用原生 tool calling 调用工具，"
            "不要在正文中模拟任何工具标签、XML、JSON 或伪协议。\n"
            "最终正文陈述已经发生并核验的结果，不用未来时计划冒充交付；不要把整段答复放进代码围栏。\n"
            "不要编造工具结果。最多工具调用轮数由运行时控制。\n\n"
            # Not a style preference: this note is what the UI shows as the
            # turn's 实施路径, so it is the only way that panel gets written.
            "在需要多步工具的工作中，如果你形成了会影响后续理解的路线选择、范围判断或关键发现，"
            "请在发起 tool_calls 的同一条 assistant content 中先写一小段自然语言工作说明。\n\n"
            "技能分层规则：任务明确匹配常驻技能目录时直接 sys_skill.open；"
            "无法判断对应技能时再调用 sys_skill.list。读取技能说明后按需 show / call，"
            "不得猜测技能工具名或参数。"
            "read_file（文件、图片、目录）、write_text_file、edit_text_file 和 shell_exec "
            "是常驻 core 能力，可以直接调用。外部 MCP 能力通过 mcporter 的 list/show/call 分层使用。\n\n"
            f"{self._workspace_context_block()}"
            f"{self._extra_system_context_block()}"
        )

    def _workspace_context_block(self) -> str:
        """Conventions that belong to this workspace, not to the harness.

        Mirrors what a project instruction file does elsewhere: the rules travel
        with the directory, so moving the agent to another workspace does not
        carry another project's Python layout along with it.
        """

        text = read_workspace_context(self.workspace_root)
        if not text:
            return ""
        return f"<workspace_context>\n{text}\n</workspace_context>\n\n"


    def _extra_system_context_block(self) -> str:
        if not self.extra_system_context:
            return ""
        return f"{self.extra_system_context}\n\n"

    def _late_system_blocks(self, system_context: str) -> list[Message]:
        """Per-request context that is re-rendered every step, not history."""
        blocks: list[Message] = []
        if self.task_plan and self.late_task_plan_context:
            blocks.append(
                {
                    "role": "system",
                    "content": (
                        "上一执行断点留下的活计划如下。先结合用户当前请求判断是否仍是同一任务；"
                        "若是则从未完成步骤继续，若不是则不要机械沿用，并在确有必要时用 update_plan 替换：\n"
                        + json.dumps(self.task_plan, ensure_ascii=False)
                    ),
                }
            )
        if system_context.strip():
            blocks.append({"role": "system", "content": system_context.strip()})
        return blocks

    def _request_messages(
        self,
        runtime: ConversationRuntime,
        *,
        system_context: str = "",
    ) -> list[Message]:
        """Derive this step's provider messages from the log.

        Assembly is a pure function of the log plus the blocks re-rendered for
        this request, so nothing the model receives can come from state that
        was never recorded.
        """

        messages = runtime.build_request_messages(
            self.system_prompt, self._late_system_blocks(system_context)
        )
        # LM Studio's Qwen chat template rejects every system message except
        # the opening one. Runtime context remains append-only in the durable
        # log, but the provider projection must be template-compatible.
        if self.profile.provider == "lm-studio":
            messages = merge_system_messages_at_start(messages)
        if self.request_transform is None:
            return messages
        try:
            return self.request_transform(messages)
        except Exception:
            # 富化失败就发原样的消息：宁可这次看不到图，也不要整轮失败。
            self._trace(
                "request_transform_failed",
                traceback=traceback.format_exc().splitlines()[-8:],
            )
            return messages


    def _trace(self, event: str, **payload: Any) -> None:
        tracer = self.debug_trace
        if tracer is None:
            return
        try:
            tracer.emit(event, **payload)
        except Exception:
            # Observability must never break the agent path.
            return

    def _cancel_requested(self) -> bool:
        if self.cancel_check is None:
            return False
        try:
            return bool(self.cancel_check())
        except Exception:
            return False

    def _raise_if_cancelled(self) -> None:
        if self._cancel_requested():
            self._trace("agent_cancel_requested")
            raise AgentCancelled("用户停止了当前轮。")

    def _take_tool_attachments(self) -> list[dict[str, Any]]:
        blocks = list(self._tool_attachments)
        self._tool_attachments.clear()
        return blocks

    def _drain_pending_messages(self) -> list[str]:
        if self.pending_messages is None:
            return []
        try:
            queued = self.pending_messages() or []
        except Exception:
            # A broken inbox must never take the turn down with it.
            return []
        return [str(item).strip() for item in queued if str(item or "").strip()]

    def _inject_pending_messages(
        self,
        runtime: ConversationRuntime,
        texts: list[str],
        *,
        step: int,
        title: str,
    ) -> Iterator[dict[str, Any]]:
        """Put what the user said mid-run into the conversation, as themselves."""

        for text in texts:
            runtime.append_message({"role": "user", "content": text})
        # A steering inbox is acknowledged only after the same durable log
        # contains the injected user messages.  If flushing fails, the inbox
        # stays intact and the next safe point can retry without data loss.
        runtime.flush()
        if self.pending_messages_committed is not None:
            self.pending_messages_committed(texts)
        self._trace("pending_messages_injected", step=step, count=len(texts))
        yield {
            "event": "activity",
            "phase": "thinking",
            "title": title,
            "detail": "\n\n".join(texts),
            "activity_type": "user_steering",
            "step": step,
        }

    def _run_tool_hook(
        self,
        hook: Callable[..., Iterator[dict[str, Any]]] | None,
        *,
        name: str,
        **payload: Any,
    ) -> Iterator[dict[str, Any]]:
        """Run one tool hook, forwarding its events and returning its decision.

        A hook that raises is treated as absent: extending the loop must never
        be able to fail a tool call that would otherwise have worked.
        """

        if hook is None:
            return None
        try:
            result = yield from hook(**payload)
        except Exception:
            self._trace(
                f"{name}_hook_failed",
                traceback=traceback.format_exc().splitlines()[-8:],
            )
            return None
        return result if isinstance(result, dict) else None

    def _tool_schemas(self) -> list[dict[str, Any]]:
        schemas = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self.tools.list_model_tools()
        ]
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": "update_plan",
                    "description": (
                        "Track complex work in 2-7 outcome-shaped steps. Use when multiple dependent "
                        "actions benefit from visible progress; keep at most one step in_progress and "
                        "update statuses as work completes."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "explanation": {
                                "type": "string",
                                "description": "Reason for a material plan change.",
                            },
                            "plan": {
                                "type": "array",
                                "minItems": 2,
                                "maxItems": 7,
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "step": {"type": "string"},
                                        "status": {
                                            "type": "string",
                                            "enum": ["pending", "in_progress", "completed"],
                                        },
                                    },
                                    "required": ["step", "status"],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "required": ["plan"],
                        "additionalProperties": False,
                    },
                },
            }
        )
        return schemas

    def _execute_model_tool(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        *,
        trusted_approval: bool = False,
        tool_call_id: str = "",
    ) -> str:
        previous_cancel_check = set_tool_cancel_check(self.cancel_check)
        previous_attachment_sink = set_tool_attachment_sink(self._tool_attachments.append)
        try:
            if tool_name == "update_plan":
                return self._apply_task_plan(tool_input)
            tool = self.tools.get_model_tool(tool_name)
            safe_input = dict(tool_input)
            # A model argument cannot choose the durable execution identity.  The
            # native provider tool-call ID always wins and survives turn recovery.
            safe_input.pop("_execution_tool_call_id", None)
            if tool_name == "shell_exec" and tool_call_id:
                safe_input["_execution_tool_call_id"] = str(tool_call_id)
            if tool_name == "shell_exec" and not trusted_approval:
                safe_input.pop("approved_by_user", None)
                safe_input.pop("_approval_source", None)
                safe_input.pop("_approval_action_id", None)
                safe_input.pop("_approval_grant", None)
            missing = [
                str(name)
                for name in (tool.parameters.get("required") or [])
                if name not in safe_input or safe_input.get(name) is None
            ]
            if missing:
                raise ValueError(
                    f"工具 {tool_name} 缺少必填参数：{', '.join(missing)}。"
                    "请补齐参数后重试。"
                )
            return str(tool.handler(safe_input))
        finally:
            set_tool_cancel_check(previous_cancel_check)
            set_tool_attachment_sink(previous_attachment_sink)

    def _review_approval(
        self,
        session_messages: list[Message],
        approval_payload: dict[str, Any],
        *,
        step: int,
    ) -> ApprovalReview:
        review = self.approval_reviewer.review(session_messages, approval_payload)
        self._trace(
            "approval_review_completed",
            step=step,
            command=approval_payload.get("command"),
            risk_category=approval_payload.get("risk_category"),
            decision=review.decision,
            reason=review.reason,
            action_id=review.action_id,
            reviewer_profile=review.reviewer_profile,
            failed=review.failed,
        )
        return review

    def _apply_task_plan(self, tool_input: dict[str, Any]) -> str:
        raw_plan = tool_input.get("plan")
        if not isinstance(raw_plan, list) or not 2 <= len(raw_plan) <= 7:
            return "TOOL_ERROR: ValueError: plan must contain 2 to 7 steps"
        plan: list[dict[str, str]] = []
        in_progress = 0
        for item in raw_plan:
            if not isinstance(item, dict):
                return "TOOL_ERROR: ValueError: each plan item must be an object"
            step = str(item.get("step") or "").strip()
            status = str(item.get("status") or "").strip()
            if not step or status not in {"pending", "in_progress", "completed"}:
                return "TOOL_ERROR: ValueError: invalid plan step or status"
            in_progress += int(status == "in_progress")
            plan.append({"step": step, "status": status})
        if in_progress > 1:
            return "TOOL_ERROR: ValueError: at most one plan step may be in_progress"
        explanation = str(tool_input.get("explanation") or "").strip()
        self.task_plan = plan
        if self.plan_update_callback is not None:
            self.plan_update_callback([dict(item) for item in plan], explanation)
        return json.dumps(
            {"ok": True, "plan": plan, "explanation": explanation},
            ensure_ascii=False,
        )

    def _task_plan_activity(self, step: int) -> dict[str, Any]:
        completed = sum(item["status"] == "completed" for item in self.task_plan)
        total = len(self.task_plan)
        current = next(
            (item["step"] for item in self.task_plan if item["status"] == "in_progress"),
            "计划已更新",
        )
        return {
            "event": "activity",
            "phase": "thinking",
            "title": "执行计划",
            "detail": current,
            "activity_type": "plan",
            "plan": [dict(item) for item in self.task_plan],
            "plan_completed": completed,
            "plan_total": total,
            "step": step,
        }

    def _maybe_compact_active_runtime(
        self,
        runtime: ConversationRuntime,
        messages: list[Message],
        *,
        tool_schemas: list[dict[str, Any]] | None = None,
        step: int,
        pressure: ActiveRuntimePressure | None = None,
    ) -> dict[str, Any] | None:
        measured = pressure or self._active_runtime_pressure(
            messages,
            tool_schemas=tool_schemas,
        )
        if not measured.reasons:
            return None
        folded_seqs = runtime.active_turn_surface_seqs()
        # A user prompt by itself has no completed implementation path to fold.
        if len(folded_seqs) < 2:
            return None
        active_messages = [{"role": "user", "content": _latest_user_content(runtime)}] + [
            message for message in runtime.log.derive_messages()[-len(folded_seqs):]
        ]
        checkpoint = summarize_active_react_checkpoint(
            self.client,
            self.profile,
            active_messages,
            task_plan=self.task_plan,
        )
        if not checkpoint:
            return None
        runtime.record_compaction(
            folded_seqs,
            "本轮运行上下文已压缩。以下是继续执行所需的高保真检查点；"
            "它不是最终答复：\n\n" + checkpoint,
        )
        self.active_runtime_was_compacted = True
        self._active_provider_prompt_tokens = 0
        self._active_provider_anchor_estimated_tokens = 0
        self._trace(
            "active_runtime_compacted",
            step=step,
            estimated_tokens=measured.estimated_tokens,
            token_count_source=measured.token_count_source,
            raw_estimated_tokens=measured.raw_estimated_tokens,
            serialized_request_bytes=measured.serialized_request_bytes,
            tool_result_chars=measured.tool_result_chars,
            context_trigger_tokens=measured.context_trigger_tokens,
            pressure_reasons=list(measured.reasons),
            original_message_count=len(folded_seqs),
            checkpoint_chars=len(checkpoint),
        )
        return {
            "event": "activity",
            "phase": "thinking",
            "title": "压缩运行上下文",
            "detail": (
                f"本轮工作集已达整理阈值；已将 {len(folded_seqs)} 条正在执行的 ReAct 消息整理为检查点，"
                "完整原始轨迹仍保留在事件日志中。"
            ),
            "activity_type": "runtime_summary",
            "step": step,
        }

    def _active_runtime_pressure(
        self,
        messages: list[Message],
        *,
        tool_schemas: list[dict[str, Any]] | None = None,
    ) -> ActiveRuntimePressure:
        resolved_tools = tool_schemas if tool_schemas is not None else self._tool_schemas()
        estimated, token_count_source, raw_estimated = self._active_runtime_context_tokens(
            messages,
            tool_schemas=resolved_tools,
        )
        request_payload = build_chat_tools_payload(
            messages,
            profile=self.profile,
            tools=resolved_tools,
            tool_choice="auto",
            reasoning_effort=self.reasoning_effort,
            request_usage=True,
        )
        serialized_request_bytes = len(
            json.dumps(
                request_payload,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        _message_bytes, tool_result_chars = message_pressure_metrics(messages)
        context_trigger = profile_context_trigger_tokens(self.profile)
        reasons: list[str] = []
        if estimated >= context_trigger:
            reasons.append("tokens")
        if serialized_request_bytes >= ACTIVE_REACT_CHECKPOINT_SERIALIZED_BYTES:
            reasons.append("serialized_bytes")
        if tool_result_chars >= ACTIVE_REACT_CHECKPOINT_TOOL_RESULT_CHARS:
            reasons.append("tool_results")
        return ActiveRuntimePressure(
            estimated_tokens=estimated,
            token_count_source=token_count_source,
            raw_estimated_tokens=raw_estimated,
            serialized_request_bytes=serialized_request_bytes,
            tool_result_chars=tool_result_chars,
            context_trigger_tokens=context_trigger,
            reasons=tuple(reasons),
        )

    def _active_runtime_context_tokens(
        self,
        messages: list[Message],
        *,
        tool_schemas: list[dict[str, Any]] | None = None,
    ) -> tuple[int, str, int]:
        """Count the full request and use sane provider usage as an anchor."""

        raw_estimated = estimate_messages_tokens(messages)
        request_payload = build_chat_tools_payload(
            messages,
            profile=self.profile,
            tools=tool_schemas if tool_schemas is not None else self._tool_schemas(),
            tool_choice="auto",
            reasoning_effort=self.reasoning_effort,
            request_usage=True,
        )
        serialized_request = json.dumps(
            request_payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        full_request_estimated = estimate_context_tokens(serialized_request)
        prompt_tokens = max(0, int(self._active_provider_prompt_tokens or 0))
        anchor_estimated = max(
            0,
            int(self._active_provider_anchor_estimated_tokens or 0),
        )
        if prompt_tokens <= 0 or raw_estimated < anchor_estimated:
            return full_request_estimated, "estimated_full_request", raw_estimated
        delta = raw_estimated - anchor_estimated
        source = "provider_usage" if delta == 0 else "provider_usage_plus_estimated_delta"
        return prompt_tokens + delta, source, raw_estimated

    def _record_response_usage(
        self,
        raw: dict[str, Any],
        runtime: ConversationRuntime,
        *,
        request_messages: list[Message] | None = None,
    ) -> None:
        usage = raw.get("usage") if isinstance(raw, dict) else None
        if not isinstance(usage, dict):
            return
        prompt_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        if prompt_tokens <= 0:
            return
        completion_tokens = int(
            usage.get("completion_tokens") or usage.get("output_tokens") or 0
        )
        if request_messages is not None:
            self._active_provider_prompt_tokens = prompt_tokens
            self._active_provider_anchor_estimated_tokens = estimate_messages_tokens(
                request_messages
            )
            self._trace(
                "active_provider_usage_recorded",
                usage_scope="single_request",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=int(
                    usage.get("total_tokens") or (prompt_tokens + completion_tokens)
                ),
                anchor_estimated_tokens=self._active_provider_anchor_estimated_tokens,
            )
        if self.usage_callback is None or self.active_runtime_was_compacted:
            return
        try:
            self.usage_callback(dict(usage), runtime.log.derive_messages())
        except Exception as error:
            self._trace(
                "provider_usage_callback_failed",
                error_type=type(error).__name__,
                error=str(error),
            )

    def _execute_tool_with_progress(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        step: int,
        *,
        trusted_approval: bool = False,
        tool_call_id: str = "",
    ) -> Iterator[dict[str, Any]]:
        result_queue: queue.Queue[str] = queue.Queue(maxsize=1)
        progress_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        started_at = time.monotonic()
        progress_seen = False
        self._trace("tool_start", step=step, tool_name=tool_name, arguments=tool_input)

        def run_tool() -> None:
            previous_sink = set_tool_progress_sink(progress_queue.put)
            previous_cancel_check = set_tool_cancel_check(self.cancel_check)
            previous_attachment_sink = set_tool_attachment_sink(self._tool_attachments.append)
            try:
                observation = self._execute_model_tool(
                    tool_name,
                    tool_input,
                    trusted_approval=trusted_approval,
                    tool_call_id=tool_call_id,
                )
            except Exception as error:
                observation = f"TOOL_ERROR: {type(error).__name__}: {error}"
                self._trace(
                    "tool_error",
                    step=step,
                    tool_name=tool_name,
                    elapsed_ms=int((time.monotonic() - started_at) * 1000),
                    error_type=type(error).__name__,
                    error=str(error),
                )
            else:
                self._trace(
                    "tool_end",
                    step=step,
                    tool_name=tool_name,
                    elapsed_ms=int((time.monotonic() - started_at) * 1000),
                    observation_chars=len(str(observation)),
                    observation_preview=str(observation)[:2000],
                )
            finally:
                set_tool_progress_sink(previous_sink)
                set_tool_cancel_check(previous_cancel_check)
                set_tool_attachment_sink(previous_attachment_sink)
            result_queue.put(str(observation))

        thread = threading.Thread(target=run_tool, name=f"work-agent-tool-{tool_name}", daemon=True)
        thread.start()

        last_emit = started_at
        fallback_interval = 60.0
        while True:
            self._raise_if_cancelled()
            try:
                event = progress_queue.get(timeout=0.2)
                progress_seen = True
                self.artifact_collector.capture_progress(event, tool_name=tool_name, step=step)
                yield event
            except queue.Empty:
                pass
            try:
                observation = result_queue.get_nowait()
            except queue.Empty:
                now = time.monotonic()
                if now - last_emit < (fallback_interval if progress_seen else 10):
                    continue
                last_emit = now
                elapsed_seconds = int(now - started_at)
                yield {
                    "event": "activity_delta",
                    "id": f"tool-{step}-{tool_name}-progress",
                    "phase": "action",
                    "title": "工具运行日志",
                    "content": tool_progress_line(tool_name, tool_input, elapsed_seconds),
                    "append_mode": "replace",
                    "step": step,
                    "tool_name": tool_name,
                }
            else:
                while True:
                    try:
                        event = progress_queue.get_nowait()
                    except queue.Empty:
                        break
                    progress_seen = True
                    self.artifact_collector.capture_progress(event, tool_name=tool_name, step=step)
                    yield event
                return observation

    def _plan_with_progress(
        self,
        *,
        messages: list[Message],
        tool_schemas: list[dict[str, Any]],
        step: int,
        draft_prefix: str = "",
    ) -> Iterator[dict[str, Any]]:
        result_queue: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)
        progress_queue: queue.Queue[dict[str, str]] = queue.Queue()
        started_at = time.monotonic()
        last_stream_at: float | None = None
        recovery_started_at: float | None = None
        recovery_last_stream_at: float | None = None
        last_heartbeat_at: float | None = None
        recovery_last_heartbeat_at: float | None = None
        request_cancel_event = threading.Event()
        stream_start_timeout = stream_start_timeout_seconds(self.profile)
        stream_idle_timeout = max(
            1,
            min(self.profile.timeout_seconds, MODEL_STREAM_IDLE_TIMEOUT_SECONDS),
        )
        active_stream_max_timeout = max(
            self.profile.timeout_seconds * MODEL_STREAM_MAX_MULTIPLIER,
            self.profile.timeout_seconds + MODEL_STREAM_MAX_EXTENSION_SECONDS,
        )
        recovery_timeout = recovery_request_timeout_seconds(self.profile)
        recovery_active_max_timeout = max(
            recovery_timeout * MODEL_RECOVERY_MAX_MULTIPLIER,
            recovery_timeout + MODEL_STREAM_MAX_EXTENSION_SECONDS,
        )
        request_id = f"model-plan-{step}-{int(started_at * 1000)}"
        request_payload = build_chat_tools_payload(
            messages,
            profile=self.profile,
            tools=tool_schemas,
            tool_choice="auto",
            reasoning_effort=self.reasoning_effort,
            request_usage=True,
        )
        serialized_request_bytes = len(
            json.dumps(request_payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
        _message_bytes, request_tool_result_chars = message_pressure_metrics(messages)
        self._trace(
            "llm_start",
            step=step,
            request_id=request_id,
            profile=self.profile.name,
            model=self.profile.model,
            message_count=len(messages),
            tool_schema_count=len(tool_schemas),
            serialized_request_bytes=serialized_request_bytes,
            tool_result_chars=request_tool_result_chars,
            start_timeout_seconds=stream_start_timeout,
            stream_idle_timeout_seconds=stream_idle_timeout,
            active_stream_max_seconds=active_stream_max_timeout,
            recovery_start_timeout_seconds=recovery_timeout,
            recovery_active_max_seconds=recovery_active_max_timeout,
        )

        def run_model() -> None:
            try:
                def on_delta(chunk: Any) -> None:
                    nonlocal last_stream_at, recovery_started_at, recovery_last_stream_at
                    last_stream_at = time.monotonic()
                    status = str(getattr(chunk, "status", "") or "")
                    status_detail = str(getattr(chunk, "status_detail", "") or "")
                    if status in {"recovery_started", "network_retry"}:
                        recovery_started_at = last_stream_at
                        recovery_last_stream_at = None
                    elif status == "recovery_streaming":
                        recovery_last_stream_at = last_stream_at
                    progress_queue.put(
                        {
                            "content": str(getattr(chunk, "content", "") or ""),
                            "reasoning": str(getattr(chunk, "reasoning", "") or ""),
                            "tool_name": str(getattr(chunk, "tool_name", "") or ""),
                            "tool_arguments": str(getattr(chunk, "tool_arguments", "") or ""),
                            "status": status,
                            "status_detail": status_detail,
                         }
                    )

                def on_heartbeat() -> None:
                    nonlocal last_heartbeat_at, recovery_last_heartbeat_at
                    if recovery_started_at is not None:
                        recovery_last_heartbeat_at = time.monotonic()
                    else:
                        last_heartbeat_at = time.monotonic()

                response = self.client.chat_tools_stream(
                    messages,
                    profile=self.profile,
                    reasoning_effort=self.reasoning_effort,
                    tools=tool_schemas,
                    tool_choice="auto",
                    on_delta=on_delta,
                    cancel_event=request_cancel_event,
                    on_heartbeat=on_heartbeat,
                )
                result_queue.put((True, response))
            except Exception as error:
                self._trace(
                    "llm_error",
                    step=step,
                    request_id=request_id,
                    elapsed_ms=int((time.monotonic() - started_at) * 1000),
                    error_type=type(error).__name__,
                    error=str(error),
                    traceback=traceback.format_exc().splitlines()[-16:],
                )
                result_queue.put((False, error))

        thread = threading.Thread(target=run_model, name=f"work-agent-model-plan-{step}", daemon=True)
        thread.start()

        yield {
            "event": "activity",
            "id": request_id,
            "phase": "thinking",
            "title": f"第 {step} 轮 · 模型思考",
            "detail": "正在决定下一步工具调用或最终回复",
            "content": "正在等待模型开始返回…",
            "step": step,
        }

        waiting_notice_emitted = False
        content_buffer = ""
        reasoning_buffer = ""
        tool_name_buffer = ""
        tool_arguments_buffer = ""
        stream_status = ""
        stream_status_detail = ""
        stream_seen = False
        last_preview = ""
        draft_content_chars = 0
        last_stream_signature: tuple[int, int, int, int, str, str] | None = None
        # 循环重复只在“连续多个数据块都命中同一重复尾段”时才判死循环。
        # 单次命中无法区分真死循环与长结构化生成(反复引用同构 XML/模板后仍在推进)。
        # 这项计数在每轮 while 内重置，任何未命中(新内容出现)都会清零。
        consecutive_repetition_hits = 0
        REPETITION_HITS_TO_STOP = 2
        while True:
            if self._cancel_requested():
                request_cancel_event.set()
                thread.join(timeout=0.25)
                self._trace("agent_cancel_requested", step=step, request_id=request_id)
                raise AgentCancelled("用户停止了当前轮。")
            stream_updated = False
            while True:
                try:
                    delta = progress_queue.get_nowait()
                except queue.Empty:
                    break
                delta_status = delta.get("status") or ""
                if delta_status in {"recovery_started", "network_retry"}:
                    if draft_content_chars:
                        yield {"event": "draft_reset", "content": draft_prefix, "step": step}
                    content_buffer = ""
                    reasoning_buffer = ""
                    tool_name_buffer = ""
                    tool_arguments_buffer = ""
                    draft_content_chars = 0
                content_buffer += delta.get("content") or ""
                reasoning_buffer += delta.get("reasoning") or ""
                tool_name_buffer += delta.get("tool_name") or ""
                tool_arguments_buffer += delta.get("tool_arguments") or ""
                stream_status = delta_status or stream_status
                stream_status_detail = delta.get("status_detail") or stream_status_detail
                stream_seen = True
                stream_updated = True
                repeated_channel = ""
                if repeated_stream_span(reasoning_buffer):
                    repeated_channel = "模型思考"
                elif repeated_stream_span(content_buffer):
                    repeated_channel = "回答草稿"
                if repeated_channel:
                    consecutive_repetition_hits += 1
                else:
                    consecutive_repetition_hits = 0
                if consecutive_repetition_hits >= REPETITION_HITS_TO_STOP:
                    request_cancel_event.set()
                    thread.join(timeout=0.25)
                    elapsed_ms = int((time.monotonic() - started_at) * 1000)
                    error = ModelStreamLoopStopped(
                        f"检测到{repeated_channel}陷入循环重复，系统已停止当前模型请求，"
                        "避免继续空转。已完成的操作和文件会保留，但任务尚未完成。"
                    )
                    self._trace(
                        "llm_stream_repetition",
                        step=step,
                        request_id=request_id,
                        channel=repeated_channel,
                        repeated_span_chars=MODEL_STREAM_REPETITION_WINDOW_CHARS,
                        consecutive_hits=consecutive_repetition_hits,
                        hits_required=REPETITION_HITS_TO_STOP,
                        elapsed_ms=elapsed_ms,
                        worker_stopped=not thread.is_alive(),
                    )
                    yield {
                        "event": "activity_delta",
                        "id": request_id,
                        "phase": "error",
                        "title": f"第 {step} 轮 · 模型陷入循环重复，已停止",
                        "content": str(error),
                        "reasoning_content": compact_reasoning_preview(reasoning_buffer),
                        "append_mode": "replace",
                        "activity_type": "model_loop_stopped",
                        "stream_status": "repetition_stopped",
                        "step": step,
                    }
                    raise error
            now = time.monotonic()
            stream_signature = (
                len(content_buffer),
                len(reasoning_buffer),
                len(tool_name_buffer),
                len(tool_arguments_buffer),
                stream_status,
                stream_status_detail,
            )
            if stream_seen and stream_updated and stream_signature != last_stream_signature:
                last_stream_signature = stream_signature
                elapsed_seconds = int(now - started_at)
                preview = model_stream_preview(
                    elapsed_seconds=elapsed_seconds,
                    content=content_buffer,
                    reasoning=reasoning_buffer,
                    tool_name=tool_name_buffer,
                    tool_arguments=tool_arguments_buffer,
                    status=stream_status,
                    status_detail=stream_status_detail,
                )
                if preview != last_preview:
                    last_preview = preview
                    yield {
                        "event": "activity_delta",
                        "id": request_id,
                        "phase": "thinking",
                        "title": f"第 {step} 轮 · 模型思考",
                        "content": preview,
                        "reasoning_content": compact_reasoning_preview(reasoning_buffer),
                        "append_mode": "replace",
                        # Carried as a field so the UI never has to pattern-match
                        # the preview text to know what the stream is doing.
                        "stream_status": stream_status,
                        "step": step,
                    }
                if (
                    len(content_buffer) > draft_content_chars
                    and not contains_tool_call_markup(content_buffer)
                ):
                    yield {
                        "event": "draft_delta",
                        "content": content_buffer[draft_content_chars:],
                        "step": step,
                    }
                    draft_content_chars = len(content_buffer)
            try:
                ok, value = result_queue.get_nowait()
            except queue.Empty:
                elapsed = now - started_at
                timeout_kind = ""
                if recovery_started_at is not None:
                    recovery_elapsed = now - recovery_started_at
                    if (
                        recovery_last_heartbeat_at is None
                        and recovery_elapsed
                        >= recovery_timeout + MODEL_RECOVERY_TIMEOUT_GRACE_SECONDS
                    ):
                        timeout_kind = "recovery_start"
                        value = RuntimeError(
                            f"模型兼容恢复在 {recovery_timeout} 秒内没有开始返回有效流，"
                            "系统已停止等待；本轮已完成的工具结果仍会保留。"
                        )
                    elif (
                        recovery_last_heartbeat_at is not None
                        and now - recovery_last_heartbeat_at >= stream_idle_timeout
                    ):
                        timeout_kind = "recovery_idle"
                        value = RuntimeError(
                            f"模型恢复流已连续 {stream_idle_timeout} 秒没有收到数据，系统已停止等待；"
                            "本轮已完成的工具结果仍会保留。"
                        )
                    elif (
                        recovery_last_heartbeat_at is not None
                        and recovery_elapsed >= recovery_active_max_timeout
                    ):
                        timeout_kind = "recovery_max_active"
                        value = RuntimeError(
                            f"模型恢复流虽持续返回，但已达到 {recovery_active_max_timeout} 秒安全上限，"
                            "本轮已完成的工具结果仍会保留。"
                        )
                elif last_heartbeat_at is None and elapsed >= stream_start_timeout:
                    timeout_kind = "start"
                    value = RuntimeError(
                        f"模型在 {stream_start_timeout} 秒内没有开始返回有效流，系统已停止等待；"
                        "本轮已完成的工具结果仍会保留。"
                    )
                elif last_heartbeat_at is not None and now - last_heartbeat_at >= stream_idle_timeout:
                    timeout_kind = "idle"
                    value = RuntimeError(
                        f"模型流已连续 {stream_idle_timeout} 秒没有收到数据，系统已停止等待；"
                        "本轮已完成的工具结果仍会保留。"
                    )
                elif last_heartbeat_at is not None and elapsed >= active_stream_max_timeout:
                    timeout_kind = "max_active"
                    value = RuntimeError(
                        f"模型虽持续返回流，但单次请求已达到 {active_stream_max_timeout} 秒安全上限，"
                        "系统已停止等待；本轮已完成的工具结果仍会保留。"
                    )

                if timeout_kind:
                    ok = False
                    request_cancel_event.set()
                    thread.join(timeout=0.25)
                    self._trace(
                        "llm_stream_timeout",
                        step=step,
                        request_id=request_id,
                        timeout_kind=timeout_kind,
                        elapsed_ms=int(elapsed * 1000),
                        idle_ms=(
                            int((now - last_heartbeat_at) * 1000)
                            if last_heartbeat_at is not None
                            else None
                        ),
                        recovery_elapsed_ms=(
                            int((now - recovery_started_at) * 1000)
                            if recovery_started_at is not None
                            else None
                        ),
                        recovery_idle_ms=(
                            int((now - recovery_last_heartbeat_at) * 1000)
                            if recovery_last_heartbeat_at is not None
                            else None
                        ),
                        worker_stopped=not thread.is_alive(),
                    )
                else:
                    try:
                        ok, value = result_queue.get(timeout=0.2)
                    except queue.Empty:
                        elapsed_seconds = int(now - started_at)
                        if not stream_seen and not waiting_notice_emitted and elapsed_seconds >= 5:
                            waiting_notice_emitted = True
                            last_preview = (
                                f"[{elapsed_seconds}s] 正在等待模型开始返回。\n"
                                "界面会保持计时；只有收到新内容时才更新过程记录。"
                            )
                            yield {
                                "event": "activity_delta",
                                "id": request_id,
                                "phase": "thinking",
                                "title": f"第 {step} 轮 · 模型思考",
                                "content": f"已等待 {elapsed_seconds}s，模型尚未开始返回…",
                                "append_mode": "replace",
                                "step": step,
                            }
                        continue

            # Drain any final stream fragments that arrived just before the model thread completed.
            while True:
                try:
                    delta = progress_queue.get_nowait()
                except queue.Empty:
                    break
                delta_status = delta.get("status") or ""
                if delta_status in {"recovery_started", "network_retry"}:
                    if draft_content_chars:
                        yield {"event": "draft_reset", "content": draft_prefix, "step": step}
                    content_buffer = ""
                    reasoning_buffer = ""
                    tool_name_buffer = ""
                    tool_arguments_buffer = ""
                    draft_content_chars = 0
                content_buffer += delta.get("content") or ""
                reasoning_buffer += delta.get("reasoning") or ""
                tool_name_buffer += delta.get("tool_name") or ""
                tool_arguments_buffer += delta.get("tool_arguments") or ""
                stream_status = delta_status or stream_status
                stream_status_detail = delta.get("status_detail") or stream_status_detail
                stream_seen = True
            final_stream_signature = (
                len(content_buffer),
                len(reasoning_buffer),
                len(tool_name_buffer),
                len(tool_arguments_buffer),
                stream_status,
                stream_status_detail,
            )
            if stream_seen and final_stream_signature != last_stream_signature:
                last_stream_signature = final_stream_signature
                elapsed_seconds = int(time.monotonic() - started_at)
                preview = model_stream_preview(
                    elapsed_seconds=elapsed_seconds,
                    content=content_buffer,
                    reasoning=reasoning_buffer,
                    tool_name=tool_name_buffer,
                    tool_arguments=tool_arguments_buffer,
                    status=stream_status,
                    status_detail=stream_status_detail,
                )
                if preview != last_preview:
                    last_preview = preview
                    yield {
                        "event": "activity_delta",
                        "id": request_id,
                        "phase": "thinking",
                        "title": f"第 {step} 轮 · 模型思考",
                        "content": preview,
                        "reasoning_content": compact_reasoning_preview(reasoning_buffer),
                        "append_mode": "replace",
                        # Carried as a field so the UI never has to pattern-match
                        # the preview text to know what the stream is doing.
                        "stream_status": stream_status,
                        "step": step,
                    }
                if (
                    len(content_buffer) > draft_content_chars
                    and not contains_tool_call_markup(content_buffer)
                ):
                    yield {
                        "event": "draft_delta",
                        "content": content_buffer[draft_content_chars:],
                        "step": step,
                    }
                    draft_content_chars = len(content_buffer)

            elapsed_seconds = int(time.monotonic() - started_at)
            if ok:
                parsed_message = response_message(value.raw)
                parsed_tool_calls = normalize_tool_calls(parsed_message)
                final_answer_without_tools = not parsed_tool_calls
                final_content = str(parsed_message.get("content") or value.content or "")
                final_reasoning = str(
                    parsed_message.get("reasoning_content")
                    or parsed_message.get("reasoning")
                    or reasoning_buffer
                    or ""
                )
                self._trace(
                    "llm_end",
                    step=step,
                    request_id=request_id,
                    elapsed_ms=elapsed_seconds * 1000,
                    response=response_debug_summary(value.raw, value.content),
                )
                if final_answer_without_tools and not final_content.strip():
                    failure_message = empty_model_response_message(self.profile.name)
                    yield {
                        "event": "activity_delta",
                        "id": request_id,
                        "phase": "error",
                        "title": f"第 {step} 轮 · 模型思考失败",
                        "content": f"✕ {failure_message}",
                        "reasoning_content": compact_reasoning_preview(final_reasoning),
                        "append_mode": "replace",
                        "step": step,
                    }
                    raise RuntimeError(failure_message)
                if final_content and not contains_tool_call_markup(final_content):
                    remaining_content = (
                        final_content[draft_content_chars:]
                        if final_content.startswith(content_buffer[:draft_content_chars])
                        else final_content
                    )
                    if remaining_content:
                        yield {
                            "event": "draft_delta",
                            "content": remaining_content,
                            "step": step,
                        }
                if final_answer_without_tools:
                    recovery = value.raw.get("_work_agent", {}).get("recovery", {})
                    recovery_note = (
                        "流式返回停滞后已自动切换兼容请求恢复。\n"
                        if isinstance(recovery, dict) and recovery
                        else ""
                    )
                    success_content = (
                        f"{recovery_note}✓ 最终答复已写入对话气泡，用时 {elapsed_seconds}s。"
                    )
                else:
                    tool_names = "、".join(call.name for call in parsed_tool_calls)
                    success_content = (
                        f"✓ 已确定下一步：{tool_names or '调用工具'}，用时 {elapsed_seconds}s。"
                    )
                yield {
                    "event": "activity_delta",
                    "id": request_id,
                    "phase": "thinking",
                    "title": f"第 {step} 轮 · 模型思考",
                    "content": success_content,
                    "reasoning_content": compact_reasoning_preview(final_reasoning),
                    "append_mode": "replace",
                    "step": step,
                }
                return value

            yield {
                "event": "activity_delta",
                "id": request_id,
                "phase": "error",
                "title": f"第 {step} 轮 · 模型思考失败",
                "content": f"\n\n✗ 模型规划失败：{type(value).__name__}: {value}\n",
                "reasoning_content": compact_reasoning_preview(reasoning_buffer),
                "step": step,
            }
            raise value


def model_stream_preview(
    *,
    elapsed_seconds: int,
    content: str,
    reasoning: str,
    tool_name: str,
    tool_arguments: str,
    status: str = "",
    status_detail: str = "",
) -> str:
    tool = tool_name.strip()
    if status == "network_retry":
        headline = f"[{elapsed_seconds}s] 网络未就绪，正在重试同一端点。"
    elif status == "recovery_started":
        headline = f"[{elapsed_seconds}s] 主流已结束，正在启动流式恢复。"
    elif status == "recovery_streaming":
        headline = f"[{elapsed_seconds}s] 恢复流正在返回。"
    else:
        headline = f"[{elapsed_seconds}s] 模型正在流式返回。"
    lines = [headline]
    if status == "network_retry":
        # Never silently switch endpoints: a different route is a different
        # model, and the user asked for this one.
        lines.append("连接失败通常是短暂的，正在退避后重试同一模型端点；不会更换模型。")
        if status_detail:
            lines.append(status_detail)
    elif status in {"recovery_started", "recovery_streaming"}:
        lines.append("当前请求未形成完整决策，系统正在自动恢复；已完成的工具结果会继续保留。")
        if status_detail:
            lines.append(f"主流结束信息：{status_detail}")

    if tool_arguments:
        preview_text, field_name = preview_tool_arguments_text(tool_arguments)
        label = f"{tool}.{field_name}" if field_name and tool else field_name or "工具参数"
        lines.extend(
            [
                "",
                f"--- 下一步：{label} ---",
                preview_text,
            ]
        )
    elif tool:
        lines.extend(["", "--- 下一步工具 ---", tool])
    elif content:
        if contains_tool_call_markup(content):
            lines.append("模型正在生成文本形式的工具调用标签，原始标签已隐藏；完成后会尝试兼容解析并执行。")
        else:
            lines.extend(
                [
                    "",
                    "--- 回答草稿 ---",
                    compact_preview_text(content, limit=3600),
                ]
            )
    elif not reasoning:
        lines.append("模型已建立连接，正在组织下一步。")
    return "\n".join(lines)


def compact_reasoning_preview(text: str, *, limit: int = 12000) -> str:
    """Keep the current reasoning readable without letting the activity panel grow forever."""
    value = str(text or "")
    if len(value) <= limit:
        return value
    head = max(2000, limit // 3)
    tail = max(4000, limit - head - 100)
    omitted = len(value) - head - tail
    return (
        value[:head].rstrip()
        + f"\n\n… 中间 {omitted} 字已折叠，以下继续显示最新思考 …\n\n"
        + value[-tail:].lstrip()
    )


def preview_tool_arguments_text(raw_arguments: str, *, limit: int = 3600) -> tuple[str, str | None]:
    field_priority = [
        "content",
        "markdown_content",
        "text",
        "body",
        "message",
        "command",
        "script",
        "input",
        "path",
        "output_path",
    ]
    parsed = parse_complete_json_object(raw_arguments)
    if parsed is not None:
        metadata = tool_argument_metadata(parsed)
        for key in field_priority:
            value = parsed.get(key)
            if isinstance(value, (str, int, float, bool)) and str(value):
                preview = compact_preview_text(str(value), limit=limit)
                return (f"{metadata}\n{preview}".strip() if metadata else preview, key)
        return compact_preview_text(json.dumps(parsed, ensure_ascii=False, indent=2), limit=limit), None

    for key in field_priority:
        partial = extract_partial_json_string_field(raw_arguments, key)
        if partial is not None:
            return compact_preview_text(partial, limit=limit), key
    return compact_preview_text(decode_json_string_fragment(raw_arguments), limit=limit), None


def parse_complete_json_object(raw_arguments: str) -> dict[str, Any] | None:
    text = str(raw_arguments or "").strip()
    if not text:
        return None

    for candidate in unique_candidates([text, escape_raw_control_chars_in_strings(text)]):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def tool_argument_metadata(parsed: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("path", "output_path", "markdown_path", "title"):
        value = parsed.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(f"{key}={truncate_text(value.strip(), 120)}")
    return "参数摘要：" + "；".join(parts) if parts else ""


def extract_partial_json_string_field(raw_arguments: str, field_name: str) -> str | None:
    text = str(raw_arguments or "")
    pattern = re.compile(rf'"{re.escape(field_name)}"\s*:\s*"')
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    return decode_json_string_fragment(text[matches[-1].end() :])


def decode_json_string_fragment(fragment: str) -> str:
    result: list[str] = []
    escaped = False
    index = 0
    while index < len(fragment):
        char = fragment[index]
        if escaped:
            if char == "n":
                result.append("\n")
            elif char == "r":
                result.append("\r")
            elif char == "t":
                result.append("\t")
            elif char == "b":
                result.append("\b")
            elif char == "f":
                result.append("\f")
            elif char == "u" and index + 4 < len(fragment):
                hex_value = fragment[index + 1 : index + 5]
                try:
                    result.append(chr(int(hex_value, 16)))
                    index += 4
                except ValueError:
                    result.append("\\u" + hex_value)
                    index += 4
            else:
                result.append(char)
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            break
        else:
            result.append(char)
        index += 1
    if escaped:
        result.append("\\")
    return "".join(result)


@dataclass(frozen=True)
class NativeToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    arguments_error: str = ""


def response_message(raw: dict[str, Any]) -> dict[str, Any]:
    choice = (raw.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    return message if isinstance(message, dict) else {}


def response_finish_reason(raw: dict[str, Any]) -> str:
    choice = (raw.get("choices") or [{}])[0] if isinstance(raw, dict) else {}
    if not isinstance(choice, dict):
        return ""
    return str(choice.get("finish_reason") or "").strip().lower()


def truncated_tool_call_messages(
    tool_calls: list[NativeToolCall],
    *,
    max_tokens: int,
    attempt: int,
) -> tuple[Message, list[Message]]:
    """Return compact history that tells the model a length-truncated call was not run.

    The raw arguments may contain tens of thousands of incomplete characters. Keeping
    them in active history both wastes context and encourages the model to repeat the
    same payload, so ReAct history stores only a small sentinel call plus an actionable
    tool observation. The trace keeps the finish reason, limit, tool names, and repeat
    count without copying the oversized incomplete payload again.
    """
    compact_calls: list[dict[str, Any]] = []
    tool_messages: list[Message] = []
    for index, tool_call in enumerate(tool_calls):
        call_id = tool_call.id or f"call_truncated_{index}"
        compact_calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": tool_call.name,
                    "arguments": json.dumps(
                        {"_tool_arguments_truncated": True},
                        ensure_ascii=False,
                    ),
                },
            }
        )
        tool_messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": tool_call.name,
                "content": (
                    "TOOL_ERROR: ToolArgumentsTruncated: 本次模型响应达到输出上限"
                    f"（finish_reason=length，max_tokens={max_tokens}），"
                    f"{tool_call.name} 的参数未保证完整，因此系统没有执行该工具。"
                    "这不是 path/content 是否平铺或 arguments 是否嵌套的问题。"
                    "下一轮不得重新发送同一份完整内容；请立即显著缩小单次工具参数，"
                    "把长脚本拆成较小模块/文件，或用小型分段补丁逐步完成。"
                    f"这是相同工具组合连续第 {attempt} 次被截断。"
                ),
            }
        )
    assistant_message: Message = {
        "role": "assistant",
        "content": "",
        "tool_calls": compact_calls,
    }
    return assistant_message, tool_messages


def invalid_tool_call_messages(
    tool_calls: list[NativeToolCall],
    *,
    attempt: int,
) -> tuple[Message, list[Message]]:
    """Keep malformed model arguments out of durable/provider history.

    A native tool call is still part of the provider protocol even when its
    ``function.arguments`` string is broken.  Replaying that raw string makes
    the entire next request fail with HTTP 400 before the model can repair it.
    Store a small valid sentinel call instead, pair every call in the batch
    with a tool result, and do not execute any handler from the malformed batch.
    """

    compact_calls: list[dict[str, Any]] = []
    tool_messages: list[Message] = []
    for index, tool_call in enumerate(tool_calls):
        call_id = tool_call.id or f"call_invalid_{index}"
        compact_calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": tool_call.name,
                    "arguments": json.dumps(
                        {"_tool_arguments_invalid": True},
                        ensure_ascii=False,
                    ),
                },
            }
        )
        if tool_call.arguments_error:
            detail = truncate_text(tool_call.arguments_error, 360)
            content = (
                "TOOL_ERROR: ToolArgumentsInvalid: 模型返回的 function.arguments "
                f"不是有效 JSON 对象，因此系统没有执行 {tool_call.name}。"
                f"解析信息：{detail}。请只重试这一小步，并用原生 tool calling "
                "返回完整、严格的 JSON 参数；不要把 arguments 再包一层。"
                f"这是相同工具组合连续第 {attempt} 次格式错误。"
            )
        else:
            content = (
                "TOOL_ERROR: ToolBatchSkipped: 同一批次中另一个工具的参数不是有效 JSON，"
                f"为避免部分执行，系统没有执行 {tool_call.name}。请重新提交这一批工具调用。"
            )
        tool_messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": tool_call.name,
                "content": content,
            }
        )
    return (
        {"role": "assistant", "content": "", "tool_calls": compact_calls},
        tool_messages,
    )


def repeated_invalid_tool_arguments_final(
    tool_names: tuple[str, ...],
    *,
    attempts: int,
) -> str:
    names = "、".join(tool_names) or "工具调用"
    return (
        f"{names} 连续 {attempts} 次返回了非法 JSON 工具参数。"
        "系统已停止自动重试，并且没有执行这些调用，避免污染会话历史或产生部分写入。"
        "请缩小这一步的参数后再继续。"
    )


def repeated_tool_truncation_final(
    tool_names: tuple[str, ...],
    *,
    max_tokens: int,
    attempts: int,
) -> str:
    names = "、".join(tool_names) or "工具调用"
    return (
        f"{names} 连续 {attempts} 次达到模型输出上限（max_tokens={max_tokens}），"
        "系统已停止自动重试，且没有执行这些不完整的工具调用，以免反复消耗和写入残缺文件。"
        "需要把单次生成内容进一步拆小后再继续。"
    )


def empty_model_response_message(profile_name: str) -> str:
    return (
        f"{profile_name} 没有返回正文或工具调用。系统已自动尝试恢复一次但仍为空，"
        "请重试；如需更换模型，请在设置中手动选择。"
    )


def response_debug_summary(raw: dict[str, Any], content: str) -> dict[str, Any]:
    choice = (raw.get("choices") or [{}])[0] if isinstance(raw, dict) else {}
    message = choice.get("message") if isinstance(choice, dict) else {}
    if not isinstance(message, dict):
        message = {}
    usage = raw.get("usage") if isinstance(raw, dict) and isinstance(raw.get("usage"), dict) else {}
    return {
        "finish_reason": choice.get("finish_reason") if isinstance(choice, dict) else None,
        "content_chars": len(str(message.get("content") or content or "")),
        "native_tool_call_count": count_native_tool_calls(message),
        "tool_names": [
            str((call.get("function") or {}).get("name") or "")
            for call in (message.get("tool_calls") or [])
            if isinstance(call, dict)
        ],
        "usage": {
            key: usage.get(key)
            for key in (
                "prompt_tokens",
                "input_tokens",
                "completion_tokens",
                "output_tokens",
                "total_tokens",
            )
            if usage.get(key) is not None
        },
    }


def count_native_tool_calls(message: dict[str, Any]) -> int:
    raw_calls = message.get("tool_calls") or []
    if not isinstance(raw_calls, list):
        return 0
    return sum(1 for item in raw_calls if isinstance(item, dict))


def normalize_tool_calls(message: dict[str, Any]) -> list[NativeToolCall]:
    raw_calls = message.get("tool_calls") or []
    calls: list[NativeToolCall] = []
    if isinstance(raw_calls, list):
        for index, raw_call in enumerate(raw_calls):
            if not isinstance(raw_call, dict):
                continue
            function = raw_call.get("function") or {}
            if not isinstance(function, dict):
                function = {}
            name = str(function.get("name") or raw_call.get("name") or "").strip()
            if not name:
                continue
            arguments, arguments_error = parse_tool_arguments_result(
                function.get("arguments") or raw_call.get("arguments") or {}
            )
            calls.append(
                NativeToolCall(
                    id=str(raw_call.get("id") or f"call_{index}"),
                    name=name,
                    arguments=arguments,
                    arguments_error=arguments_error,
                )
            )
    if calls:
        return calls
    return parse_text_tool_calls(str(message.get("content") or ""))


def has_native_tool_calls(message: dict[str, Any]) -> bool:
    raw_calls = message.get("tool_calls") or []
    return isinstance(raw_calls, list) and any(isinstance(item, dict) for item in raw_calls)


TOOL_CALL_TAG_NAME_PATTERN = r"(?:tool_calls?|工具调用(?:列表)?)"
TOOL_CALL_TAG_END_PATTERN = r"(?=[\s>/])"
TOOL_CALL_MARKUP_RE = re.compile(
    rf"</?\s*{TOOL_CALL_TAG_NAME_PATTERN}{TOOL_CALL_TAG_END_PATTERN}",
    flags=re.IGNORECASE,
)
TOOL_CALL_OPEN_RE = re.compile(
    rf"<\s*(?P<tag>{TOOL_CALL_TAG_NAME_PATTERN}){TOOL_CALL_TAG_END_PATTERN}(?P<attrs>[^>]*)/?>",
    flags=re.IGNORECASE | re.DOTALL,
)
TOOL_CALL_CLOSE_RE = re.compile(
    rf"</\s*{TOOL_CALL_TAG_NAME_PATTERN}\s*>",
    flags=re.IGNORECASE,
)


def contains_tool_call_markup(text: str) -> bool:
    return bool(TOOL_CALL_MARKUP_RE.search(str(text or "")))


def parse_text_tool_calls(content: str) -> list[NativeToolCall]:
    """Compat parser for models that incorrectly emit tool calls as XML-ish text."""
    text = str(content or "")
    if not contains_tool_call_markup(text):
        return []
    calls: list[NativeToolCall] = []
    attr_pattern = re.compile(
        r"([A-Za-z_][\w:-]*)\s*=\s*(\"[^\"]*\"|'[^']*')",
        flags=re.DOTALL,
    )
    for index, match in enumerate(TOOL_CALL_OPEN_RE.finditer(text)):
        attrs_text = match.group("attrs") or ""
        attrs: dict[str, str] = {}
        for attr_match in attr_pattern.finditer(attrs_text):
            raw_value = attr_match.group(2)
            attrs[attr_match.group(1).lower()] = html.unescape(raw_value[1:-1])
        name = (attrs.get("name") or attrs.get("tool_name") or "").strip()
        if not name:
            continue
        raw_arguments = (
            attrs.get("arguments")
            or attrs.get("args")
            or attrs.get("input")
            or attrs.get("parameters")
            or ""
        )
        if not raw_arguments:
            close_match = TOOL_CALL_CLOSE_RE.search(text, match.end())
            close_index = close_match.start() if close_match else -1
            if close_index > match.end():
                raw_arguments = html.unescape(strip_xmlish_tags(text[match.end() : close_index]).strip())
        arguments, arguments_error = parse_tool_arguments_result(raw_arguments)
        calls.append(
            NativeToolCall(
                id=(attrs.get("tool_call_id") or attrs.get("id") or f"call_text_{index}").strip(),
                name=name,
                arguments=arguments,
                arguments_error=arguments_error,
            )
        )
    return calls


def strip_xmlish_tags(text: str) -> str:
    return re.sub(
        rf"</?\s*{TOOL_CALL_TAG_NAME_PATTERN}{TOOL_CALL_TAG_END_PATTERN}[^>]*>",
        "",
        str(text or ""),
        flags=re.IGNORECASE | re.DOTALL,
    )


def strip_tool_call_markup(text: str) -> str:
    cleaned = re.sub(
        rf"<\s*(?:tool_calls|工具调用列表){TOOL_CALL_TAG_END_PATTERN}[^>]*>.*?</\s*(?:tool_calls|工具调用列表)\s*>",
        "",
        str(text or ""),
        flags=re.IGNORECASE | re.DOTALL,
    )
    cleaned = re.sub(
        rf"<\s*(?:tool_call|工具调用){TOOL_CALL_TAG_END_PATTERN}[^>]*>.*?</\s*{TOOL_CALL_TAG_NAME_PATTERN}\s*>",
        "",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    cleaned = re.sub(
        rf"<\s*(?:tool_call|工具调用){TOOL_CALL_TAG_END_PATTERN}[^>]*>.*?</\s*(?:tool_calls|工具调用列表)\s*>",
        "",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    cleaned = re.sub(
        rf"<\s*(?:tool_call|工具调用){TOOL_CALL_TAG_END_PATTERN}[^>]*/\s*>",
        "",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    cleaned = re.sub(
        rf"<\s*{TOOL_CALL_TAG_NAME_PATTERN}{TOOL_CALL_TAG_END_PATTERN}[^>]*>.*$",
        "",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return strip_xmlish_tags(cleaned)


def assistant_visible_content(message: dict[str, Any]) -> str:
    content = str(message.get("content") or "")
    if not content:
        return ""
    if contains_tool_call_markup(content):
        return strip_tool_call_markup(content).strip()
    return content.strip()


def merge_visible_react_content(parts: list[str], final_content: str = "") -> str:
    visible_parts = [str(part or "").strip() for part in parts if str(part or "").strip()]
    final = str(final_content or "").strip()
    if final:
        accumulated = "\n\n".join(visible_parts)
        if not accumulated:
            return final
        if final == accumulated or final.startswith(accumulated):
            return final
        if accumulated.endswith(final):
            return accumulated
        visible_parts.append(final)
    return "\n\n".join(visible_parts)


def visible_react_draft_prefix(parts: list[str]) -> str:
    content = merge_visible_react_content(parts)
    return f"{content}\n\n" if content else ""


def assistant_message_for_history(
    message: dict[str, Any],
    *,
    tool_calls: list[NativeToolCall] | None = None,
) -> Message:
    clean: Message = {
        "role": "assistant",
        "content": assistant_visible_content(message) if tool_calls else (message.get("content") or ""),
    }
    normalized_calls = tool_calls
    if normalized_calls is None and message.get("tool_calls"):
        candidate_calls = normalize_tool_calls(message)
        if candidate_calls and not any(call.arguments_error for call in candidate_calls):
            normalized_calls = candidate_calls
    if normalized_calls:
        clean["tool_calls"] = [
            {
                "id": tool_call.id or f"call_{index}",
                "type": "function",
                "function": {
                    "name": tool_call.name,
                    "arguments": json.dumps(tool_call.arguments, ensure_ascii=False),
                },
            }
            for index, tool_call in enumerate(normalized_calls)
        ]
    reasoning_content = str(message.get("reasoning_content") or message.get("reasoning") or "")
    if reasoning_content:
        clean["reasoning_content"] = reasoning_content
    return clean


def text_tool_call_repair_messages() -> tuple[Message, Message]:
    return (
        {"role": "assistant", "content": ""},
        {
            "role": "user",
            "content": (
                "你上一条返回了文本形式的 <tool_call> 标签，但系统无法解析。"
                "请使用原生 tool calling 调用工具；如果不需要工具，请直接给出最终Markdown答复。"
                "不要把工具调用写成XML或JSON正文。"
            ),
        },
    )


def parse_tool_arguments(value: Any) -> dict[str, Any]:
    parsed, error = parse_tool_arguments_result(value)
    if not error:
        return parsed
    return {"value": str(value or "").strip()}


def parse_tool_arguments_result(value: Any) -> tuple[dict[str, Any], str]:
    if isinstance(value, dict):
        return value, ""
    if value is None:
        return {}, ""
    text = str(value or "").strip()
    if not text:
        return {}, ""
    last_error: Exception | None = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        last_error = error
        try:
            parsed = json.loads(escape_raw_control_chars_in_strings(text))
        except json.JSONDecodeError as error:
            last_error = error
            try:
                parsed = parse_react_json(text)
            except ValueError as error:
                last_error = error
                preview = truncate_text(text.replace("\n", "\\n"), 240)
                return {}, f"{last_error}; preview={preview}"
    if isinstance(parsed, dict):
        return parsed, ""
    return {}, f"JSON root must be an object, got {type(parsed).__name__}"


def parse_shell_approval_required_observation(tool_name: str, observation: str) -> dict[str, Any] | None:
    if tool_name != "shell_exec":
        return None
    try:
        payload = json.loads(str(observation or ""))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("status") != "approval_required":
        return None
    return payload


def approval_payload_with_review(
    payload: dict[str, Any],
    review: ApprovalReview,
) -> dict[str, Any]:
    enriched = dict(payload)
    enriched["review_decision"] = review.decision
    enriched["review_reason"] = review.reason
    enriched["reviewer_profile"] = review.reviewer_profile
    enriched["review_failed"] = review.failed
    enriched["action_id"] = review.action_id
    return enriched


def approval_required_final_text(
    payload: dict[str, Any],
    *,
    batch_count: int = 1,
    batch_remaining: int = 1,
) -> str:
    command = str(payload.get("command") or "").strip()
    reason = str(payload.get("reason") or payload.get("detail") or "该命令需要用户确认后才能执行。").strip()
    review_reason = str(payload.get("review_reason") or "").strip()
    risk = str(payload.get("risk_category") or "EXECUTE").strip()
    lines = [
        "需要你确认后我才能继续执行这批工具调用。",
        "",
        f"- 风险类别：{risk}",
        f"- 原因：{reason}",
        f"- 本批工具调用：共 {max(1, batch_count)} 个，确认后将从当前待审批命令开始继续执行剩余 {max(1, batch_remaining)} 个",
    ]
    if command:
        lines.extend(["", "当前待审批命令：", "", f"```bash\n{command}\n```"])
    if review_reason:
        lines.extend(["", f"独立审查未自动放行：{review_reason}"])
    lines.append("")
    lines.append("点击下面的“确认执行”后，后端会恢复同一个 pending batch；不会让模型重新生成或改写这批工具调用。")
    return "\n".join(lines)


def pending_tool_batch_state(
    *,
    runtime_messages_before_batch: list[Message],
    assistant_message: Message,
    tool_calls: list[NativeToolCall],
    approval_index: int,
    completed_tool_messages: list[Message],
    step: int,
    profile_name: str,
    model: str,
    max_steps: int,
    system_context: str,
    extra_system_context: str,
    approval_payload: dict[str, Any],
    reasoning_effort: str = "medium",
    auto_approve: bool = True,
    visible_content_parts: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "kind": "tool_batch",
        "runtime_messages_before_batch": runtime_messages_before_batch,
        "assistant_message": assistant_message,
        "tool_calls": [native_tool_call_to_payload(item) for item in tool_calls],
        "approval_index": max(0, approval_index),
        "completed_tool_messages": completed_tool_messages,
        "step": step,
        "profile_name": profile_name,
        "model": model,
        "max_steps": max_steps,
        "system_context": system_context,
        "extra_system_context": extra_system_context,
        "approval_payload": approval_payload,
        "reasoning_effort": normalize_reasoning_effort(reasoning_effort),
        "auto_approve": bool(auto_approve),
        "visible_content_parts": [
            str(item).strip()
            for item in (visible_content_parts or [])
            if str(item).strip()
        ],
        "approval_batch_commands": approval_batch_commands(tool_calls, start_index=approval_index),
    }


def native_tool_call_to_payload(tool_call: NativeToolCall) -> dict[str, Any]:
    return {
        "id": tool_call.id,
        "name": tool_call.name,
        "arguments": tool_call.arguments,
    }


def native_tool_call_from_payload(payload: dict[str, Any]) -> NativeToolCall:
    return NativeToolCall(
        id=str(payload.get("id") or ""),
        name=str(payload.get("name") or ""),
        arguments=payload.get("arguments") if isinstance(payload.get("arguments"), dict) else {},
    )


def approval_granted_tool_input(
    tool_call: NativeToolCall,
    approval_payload: dict[str, Any],
    *,
    source: str,
) -> dict[str, Any]:
    arguments = dict(tool_call.arguments)
    if tool_call.name == "shell_exec":
        arguments.pop("approved_by_user", None)
        arguments["_approval_source"] = source
        arguments["_approval_action_id"] = str(approval_payload.get("action_id") or "")
        arguments["_approval_grant"] = issue_internal_approval_grant(
            action_id=str(approval_payload.get("action_id") or ""),
            source=source,
        )
    return arguments


def approval_batch_commands(tool_calls: list[NativeToolCall], *, start_index: int = 0) -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    for index, tool_call in enumerate(tool_calls[start_index:], start=start_index):
        if tool_call.name != "shell_exec":
            continue
        command = str(tool_call.arguments.get("command") or "").strip()
        if not command:
            continue
        commands.append(
            {
                "index": index,
                "command": command,
                "cwd": str(tool_call.arguments.get("cwd") or "."),
                "timeout_seconds": int(tool_call.arguments.get("timeout_seconds") or 120),
            }
        )
    return commands


def tool_progress_line(tool_name: str, tool_input: dict[str, Any], elapsed_seconds: int) -> str:
    if tool_name in {"generate_meeting_minutes", "transcribe_meeting_audio"}:
        return meeting_minutes_progress_line(tool_input, elapsed_seconds)
    if tool_name == "shell_exec":
        return f"[{elapsed_seconds}s] 终端命令仍在执行，完成后会自动继续。\n"
    if tool_name == "create_docx_from_markdown":
        return f"[{elapsed_seconds}s] 正在生成 Word 文档，完成后会返回文件路径。\n"
    return f"[{elapsed_seconds}s] {tool_name} 仍在执行，完成后会自动继续。\n"


def meeting_minutes_progress_line(tool_input: dict[str, Any], elapsed_seconds: int) -> str:
    input_path = Path(
        str(
            tool_input.get("input_path")
            or tool_input.get("audio_path")
            or tool_input.get("transcript_path")
            or ""
        )
    )
    meeting_name = str(tool_input.get("meeting_name") or input_path.stem or "会议").strip()
    output_dir = Path(str(tool_input.get("output_dir") or "meet_files"))
    stem = sanitize_progress_name(input_path.stem)
    asr_root = Path("meet_files") / "asr_full" / stem if stem else None

    signals: list[str] = []
    if asr_root and asr_root.exists():
        signals.append("已创建 ASR 工作目录")
        audio_root = asr_root / "audio"
        if audio_root.exists():
            signals.append("音频预处理/降噪已启动")
        deepfilter_inputs = list(asr_root.rglob("*.deepfilter_input.wav"))
        enhanced_outputs = list(asr_root.rglob("enhanced/*.wav")) + list(asr_root.rglob("*.enhanced.wav"))
        meeting_ready_outputs = list(asr_root.rglob("*.meeting_ready.wav"))
        standardized_outputs = list(asr_root.rglob("*.standardized_16k.wav"))
        if deepfilter_inputs:
            signals.append("DeepFilterNet 输入音频已生成")
            if not enhanced_outputs and not meeting_ready_outputs and elapsed_seconds >= 60:
                signals.append("DeepFilterNet 增强结果尚未出现，auto 模式超时后会降级")
        if enhanced_outputs:
            signals.append("DeepFilterNet 增强结果已生成")
        if meeting_ready_outputs:
            signals.append("音频预处理结果已生成")
        if standardized_outputs:
            signals.append("已降级为 FFmpeg 预处理音频")
        asr_progress = describe_qwen3_asr_progress(asr_root)
        if asr_progress:
            signals.append(asr_progress)
        elif meeting_ready_outputs or standardized_outputs:
            signals.append("正在等待 VAD 分块计划或 Qwen3-ASR 模型加载输出")

    archive_dir = output_dir / "会议项目" / sanitize_progress_name(meeting_name)
    internal_path = archive_dir / f"{sanitize_progress_name(meeting_name)}_会议沟通内容整理_内部留档版.md"
    work_path = archive_dir / f"{sanitize_progress_name(meeting_name)}_会议纪要_工作提交版.md"
    work_docx_path = archive_dir / f"{sanitize_progress_name(meeting_name)}会议纪要.docx"
    manifest_path = archive_dir / "manifest.json"
    if internal_path.exists():
        signals.append("内部留档版已写出")
    if work_path.exists():
        signals.append("工作提交版Markdown已写出")
    if work_docx_path.exists():
        signals.append("工作提交版DOCX已写出")
    if manifest_path.exists():
        signals.append("会议归档清单已更新")

    if not signals:
        signals.append("工具已启动，正在初始化模型或准备音频")

    status = "\n- ".join(signals[-5:])
    return (
        f"[{elapsed_seconds}s] 会议纪要工具仍在执行。\n"
        f"- {status}\n"
        "说明：当前等待的是本地音频/ASR/文档生成子流程结束；有新分块结果或文件写出后会继续自动推进。\n"
    )


def describe_qwen3_asr_progress(asr_root: Path) -> str:
    output_root = latest_parent_with_file(asr_root, "chunk_plan.json")
    if output_root is None:
        if list(asr_root.rglob("qwen3")):
            return "Qwen3-ASR 输出目录已创建，正在准备 VAD 分块计划"
        return ""

    plan = read_json_file(output_root / "chunk_plan.json")
    summary = read_json_file(output_root / "summary.json")
    progress_rows = read_progress_jsonl(output_root / "progress.jsonl")
    chunk_count = int(plan.get("chunk_count") or summary.get("completed_chunks") or 0)
    chunks = plan.get("chunks") if isinstance(plan.get("chunks"), list) else []
    completed_from_items = len(list((output_root / "items").glob("chunk_*/transcript.txt")))
    completed = max(
        safe_int(summary.get("completed_chunks")),
        len(progress_rows),
        completed_from_items,
    )
    complete = bool(summary.get("complete")) or (
        chunk_count > 0 and completed >= chunk_count and (output_root / "transcript.txt").is_file()
    )
    duration = str(plan.get("duration") or summary.get("duration") or "")
    mode = str((plan.get("chunk_plan") or {}).get("effective_chunk_mode") or summary.get("chunk_mode") or "")
    progress_age = seconds_since_latest_file(
        [
            output_root / "progress.jsonl",
            output_root / "summary.json",
            output_root / "transcript.txt",
        ]
    )

    prefix_parts = []
    if duration:
        prefix_parts.append(f"音频时长 {duration}")
    if chunk_count:
        prefix_parts.append(f"共 {chunk_count} 个分块")
    if mode:
        prefix_parts.append(f"模式 {mode}")
    prefix = "，".join(prefix_parts)

    if complete:
        return (
            f"Qwen3-ASR 转写已完成"
            f"{f'（{prefix}）' if prefix else ''}，正在写出标准 ASR 文稿或进入纪要生成"
        )

    if chunk_count <= 0:
        return "Qwen3-ASR 脚本已启动，正在探测音频时长并生成 VAD 分块计划"

    next_index = min(completed + 1, chunk_count)
    next_chunk = chunk_by_index(chunks, next_index)
    next_range = chunk_range_text(next_chunk)
    last_row = progress_rows[-1] if progress_rows else {}
    last_text = ""
    if last_row:
        last_index = safe_int(last_row.get("index"))
        infer_seconds = last_row.get("infer_seconds")
        char_count = last_row.get("char_count")
        last_text = (
            f"；最近完成第 {last_index} 块"
            f"{f'，推理 {infer_seconds}s' if infer_seconds not in (None, '') else ''}"
            f"{f'，{char_count} 字' if char_count not in (None, '') else ''}"
        )
    age_text = f"；最近进度文件更新 {progress_age}s 前" if progress_age is not None else ""
    chunk_files = len(list((output_root / "chunks").glob("*.wav")))
    chunk_file_text = f"；已导出 {chunk_files} 个分块音频" if chunk_files else ""

    return (
        f"Qwen3-ASR 正在识别第 {next_index}/{chunk_count} 块"
        f"{f'（{next_range}）' if next_range else ''}"
        f"；已完成 {completed}/{chunk_count} 块"
        f"{last_text}{age_text}{chunk_file_text}。"
        "当前主要等待：本地 MLX/Qwen3 对当前分块返回识别结果。"
    )


def latest_parent_with_file(root: Path, filename: str) -> Path | None:
    candidates = sorted(
        root.rglob(filename),
        key=lambda path: path.stat().st_mtime if path.exists() else 0,
        reverse=True,
    )
    if not candidates:
        return None
    return candidates[0].parent


def read_json_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def read_progress_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            rows.append(parsed)
    return rows


def safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def chunk_by_index(chunks: list[Any], index: int) -> dict[str, Any]:
    for item in chunks:
        if isinstance(item, dict) and safe_int(item.get("index")) == index:
            return item
    return {}


def chunk_range_text(chunk: dict[str, Any]) -> str:
    start = str(chunk.get("start") or "").strip()
    end = str(chunk.get("end") or "").strip()
    if start and end:
        return f"{start}-{end}"
    return ""


def seconds_since_latest_file(paths: list[Path]) -> int | None:
    mtimes = [path.stat().st_mtime for path in paths if path.is_file()]
    if not mtimes:
        return None
    return max(0, int(time.time() - max(mtimes)))


def sanitize_progress_name(name: str) -> str:
    cleaned = "".join(char if char not in '/\\:*?"<>|' else "_" for char in str(name or "")).strip()
    return cleaned or "meeting"


def parse_react_json(content: str) -> dict[str, Any]:
    candidates = unique_candidates(
        [
            content.strip(),
            strip_markdown_json_fence(content.strip()),
            extract_first_json_object(content),
        ]
    )
    last_error: Exception | None = None
    for candidate in candidates:
        if not candidate:
            continue
        for source in unique_candidates([candidate, escape_raw_control_chars_in_strings(candidate)]):
            try:
                parsed = json.loads(source)
            except json.JSONDecodeError as error:
                last_error = error
                continue
            if isinstance(parsed, dict):
                return parsed
            last_error = ValueError("JSON root is not an object")
    detail = str(last_error) if last_error else "没有找到JSON对象"
    preview = truncate_text(content.replace("\n", "\\n"), 500)
    raise ValueError(f"Model did not return a valid ReAct JSON object: {detail}; preview={preview}")


def unique_candidates(values: list[str]) -> list[str]:
    seen: set[str] = set()
    candidates: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        candidates.append(value)
    return candidates


def strip_markdown_json_fence(content: str) -> str:
    text = content.strip()
    if not text.startswith("```"):
        return text
    text = re.sub(r"^```(?:json|JSON)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_first_json_object(content: str) -> str:
    start = content.find("{")
    if start < 0:
        return ""

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(content)):
        char = content[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return content[start : index + 1].strip()
    return content[start:].strip()


def escape_raw_control_chars_in_strings(content: str) -> str:
    result: list[str] = []
    in_string = False
    escaped = False
    changed = False
    for char in content:
        if in_string:
            if escaped:
                result.append(char)
                escaped = False
                continue
            if char == "\\":
                result.append(char)
                escaped = True
                continue
            if char == '"':
                result.append(char)
                in_string = False
                continue
            if char == "\n":
                result.append("\\n")
                changed = True
                continue
            if char == "\r":
                result.append("\\r")
                changed = True
                continue
            if char == "\t":
                result.append("\\t")
                changed = True
                continue
            if ord(char) < 0x20:
                result.append(f"\\u{ord(char):04x}")
                changed = True
                continue
            result.append(char)
            continue

        result.append(char)
        if char == '"':
            in_string = True
    return "".join(result) if changed else content


def react_json_repair_prompt(error: Exception) -> str:
    return (
        "你的上一条输出不是合法的ReAct JSON对象，无法执行。"
        f"错误摘要：{truncate_text(str(error), 220)}\n"
        "请根据当前任务重新输出，只能输出一个合法JSON对象：\n"
        "{\"status\":\"一句用户可见状态\",\"action\":\"工具名\",\"action_input\":{...}}\n"
        "或\n"
        "{\"final\":\"最终Markdown答复\"}\n"
        "注意：JSON字符串里的换行必须写成\\n。"
    )


def visible_summary(value: Any, *, default: str) -> str:
    text = str(value or "").strip()
    if not text:
        return default
    return truncate_text(text.replace("\n", " "), 96)


def summarize_tool_input(tool_input: dict[str, Any]) -> str:
    if not tool_input:
        return "无参数"
    compact = json.dumps(tool_input, ensure_ascii=False)
    return truncate_text(compact, 220)


def truncate_text(value: str, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def deterministic_tool_success_final(
    tool_name: str,
    tool_input: dict[str, Any],
    observation: str,
) -> str | None:
    """Close a turn when a delivery tool already returned an authoritative result.

    A successful report save is itself the completion signal. Asking the model
    for a third turn merely to paraphrase this structured result can turn a
    completed local write into a visible failure when the provider is flaky.
    Keep this deliberately narrow so ordinary tools continue through ReAct.
    """

    is_report_save = str(tool_name or "").strip() == "save_work_report"
    if str(tool_name or "").strip() == "sys_skill":
        is_report_save = (
            str(tool_input.get("op") or "").strip().lower() == "call"
            and str(tool_input.get("skill_id") or "").strip() == "work-reports"
            and str(tool_input.get("tool_name") or "").strip() == "save_work_report"
        )
    if not is_report_save:
        return None

    try:
        payload = json.loads(str(observation or ""))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if (
        not isinstance(payload, dict)
        or payload.get("ok") is not True
        or payload.get("verified") is not True
    ):
        return None

    report_labels = {"daily": "日报", "weekly": "周报", "biweekly": "双周报"}
    report_type = str(payload.get("report_type") or "").strip()
    report_label = report_labels.get(report_type, "工作汇报")
    start_date = str(payload.get("start_date") or "").strip()
    end_date = str(payload.get("end_date") or "").strip()
    date_text = start_date if start_date == end_date else f"{start_date} 至 {end_date}"
    path = str(payload.get("content_path") or "").strip()
    coverage_labels = {"full": "完整", "partial": "部分", "external_gap": "存在外部工作缺口"}
    coverage = coverage_labels.get(str(payload.get("source_coverage") or "").strip(), "未标注")

    lines = [
        f"已完成并核验保存{date_text + ' ' if date_text else ''}{report_label}。",
        f"证据覆盖：{coverage}。",
    ]
    if payload.get("needs_user_input") is True:
        lines.append("当前版本已保存；仍有线下或外部工作信息需要补充。")
    if path:
        lines.append(f"## 交付文件\n\n- `{path}`")
    return "\n\n".join(lines)


def tool_observation_failed(observation: str) -> bool:
    """Identify tool-level failures without hiding them in a normal result."""

    text = str(observation or "").strip()
    upper = text.upper()
    if upper.startswith(("TOOL_ERROR:", "MCP_TOOL_ERROR:", "ERROR:", "TRACEBACK")):
        return True
    try:
        payload = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    if payload.get("ok") is False or payload.get("success") is False:
        return True
    return str(payload.get("status") or "").strip().lower() in {"error", "failed", "failure"}
