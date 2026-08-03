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
from .config import ModelProfile
from .debug_trace import compact_message_summary
from .llm import (
    RECOVERY_REQUEST_TIMEOUT_SECONDS,
    Message,
    OpenAICompatibleClient,
    normalize_reasoning_effort,
)
from .memory import (
    ACTIVE_REACT_CHECKPOINT_TRIGGER_TOKENS,
    ContextCompactionError,
    estimate_messages_tokens,
    summarize_active_react_checkpoint,
)
from .progress import compact_preview_text, set_tool_progress_sink
from .session_store import repair_runtime_message_sequence
from .shell_tools import issue_internal_approval_grant
from .tool_bus import ToolBus


DEFAULT_MAX_STEPS = 50
MODEL_STREAM_IDLE_TIMEOUT_SECONDS = 45
MODEL_STREAM_MAX_MULTIPLIER = 4
MODEL_STREAM_MAX_EXTENSION_SECONDS = 60
MODEL_RECOVERY_TIMEOUT_GRACE_SECONDS = 5
MODEL_RECOVERY_MAX_MULTIPLIER = 4


class AgentCancelled(RuntimeError):
    """Raised when the current single-agent turn is cancelled by runtime state."""


@dataclass(frozen=True)
class AgentResult:
    final: str
    steps_used: int
    model_profile: str
    used_tools: bool = False
    messages: list[Message] | None = None


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
        reasoning_effort: str = "medium",
        auto_approve: bool = False,
        approval_reviewer: ApprovalReviewer | None = None,
        plan_update_callback: Callable[[list[dict[str, str]], str], None] | None = None,
        initial_task_plan: list[dict[str, str]] | None = None,
    ) -> None:
        self.client = client
        self.profile = profile
        self.tools = tools
        self.max_steps = max_steps
        self.extra_system_context = (extra_system_context or "").strip()
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.debug_trace = debug_trace
        self.cancel_check = cancel_check
        self.reasoning_effort = normalize_reasoning_effort(reasoning_effort)
        self.auto_approve = bool(auto_approve)
        self.approval_reviewer = approval_reviewer or ApprovalReviewer(
            client=client,
            profile=profile,
        )
        self.plan_update_callback = plan_update_callback
        self.task_plan = [
            {"step": str(item.get("step") or ""), "status": str(item.get("status") or "pending")}
            for item in (initial_task_plan or [])
            if isinstance(item, dict) and str(item.get("step") or "").strip()
        ]

    def run(self, goal: str) -> AgentResult:
        return self.run_messages([{"role": "user", "content": goal}])

    def run_messages(
        self,
        session_messages: list[Message],
        *,
        system_context: str = "",
    ) -> AgentResult:
        messages: list[Message] = self._model_messages(session_messages, system_context=system_context)
        tool_schemas = self._tool_schemas()
        used_tools = False
        self._trace(
            "agent_start",
            mode="sync",
            session_message_count=len(session_messages),
            model_message_count=len(messages),
            tool_schema_count=len(tool_schemas),
            messages=compact_message_summary(session_messages[-12:]),
        )

        for step in range(1, self.max_steps + 1):
            self._raise_if_cancelled()
            self._maybe_compact_active_runtime(messages, step=step)
            started_at = time.monotonic()
            self._trace(
                "llm_start",
                step=step,
                message_count=len(messages),
                tool_schema_count=len(tool_schemas),
            )
            response = self.client.chat(
                messages,
                profile=self.profile,
                reasoning_effort=self.reasoning_effort,
                tools=tool_schemas,
                tool_choice="auto",
            )
            self._trace(
                "llm_end",
                step=step,
                elapsed_ms=int((time.monotonic() - started_at) * 1000),
                response=response_debug_summary(response.raw, response.content),
            )
            assistant_message = response_message(response.raw)
            tool_calls = normalize_tool_calls(assistant_message)
            self._trace(
                "llm_decision",
                step=step,
                content_chars=len(str(assistant_message.get("content") or response.content or "")),
                native_tool_call_count=count_native_tool_calls(assistant_message),
                parsed_tool_calls=[{"id": call.id, "name": call.name} for call in tool_calls],
            )

            # ReAct state transition is determined only by whether a tool call
            # can be parsed from this assistant message. Content may contain
            # user-visible preamble before tool calls, so content presence is
            # never an end signal.
            if not tool_calls:
                raw_content = str(assistant_message.get("content") or response.content or "")
                if contains_tool_call_markup(raw_content):
                    assistant_fix, user_fix = text_tool_call_repair_messages()
                    messages.extend([assistant_fix, user_fix])
                    continue
                final = raw_content.strip()
                if not final:
                    raise RuntimeError(empty_model_response_message(self.profile.name))
                final_message: Message = {
                    "role": "assistant",
                    "content": final,
                }
                session_messages.append(final_message)
                messages.append(final_message)
                self._trace(
                    "agent_final",
                    step=step,
                    used_tools=used_tools,
                    final_chars=len(str(final_message["content"])),
                )
                return AgentResult(
                    final=str(final_message["content"]),
                    steps_used=step,
                    model_profile=self.profile.name,
                    used_tools=used_tools,
                    messages=session_messages,
                )

            batch_session_start = len(session_messages)
            batch_model_start = len(messages)
            completed_tool_messages: list[Message] = []
            deterministic_final: str | None = None
            assistant_history = assistant_message_for_history(assistant_message, tool_calls=tool_calls)
            session_messages.append(assistant_history)
            messages.append(assistant_history)
            for index, tool_call in enumerate(tool_calls):
                self._raise_if_cancelled()
                used_tools = True
                tool_name = tool_call.name
                tool_input = tool_call.arguments
                tool_started_at = time.monotonic()
                self._trace(
                    "tool_start",
                    step=step,
                    tool_name=tool_name,
                    tool_call_id=tool_call.id,
                    arguments=tool_input,
                )
                try:
                    observation = self._execute_model_tool(tool_name, tool_input)
                except Exception as error:
                    observation = f"TOOL_ERROR: {type(error).__name__}: {error}"
                    self._trace(
                        "tool_error",
                        step=step,
                        tool_name=tool_name,
                        tool_call_id=tool_call.id,
                        elapsed_ms=int((time.monotonic() - tool_started_at) * 1000),
                        error_type=type(error).__name__,
                        error=str(error),
                    )
                else:
                    self._trace(
                        "tool_end",
                        step=step,
                        tool_name=tool_name,
                        tool_call_id=tool_call.id,
                        elapsed_ms=int((time.monotonic() - tool_started_at) * 1000),
                        observation_chars=len(str(observation)),
                        observation_preview=str(observation)[:2000],
                    )

                approval_payload = parse_shell_approval_required_observation(tool_name, observation)
                review: ApprovalReview | None = None
                if (
                    approval_payload is not None
                    and self.auto_approve
                    and approval_payload.get("reviewable_by_model") is True
                ):
                    review = self._review_approval(session_messages, approval_payload, step=step)
                    approval_payload = approval_payload_with_review(approval_payload, review)
                    if review.approved:
                        observation = self._execute_model_tool(
                            tool_name,
                            approval_granted_tool_input(
                                tool_call,
                                approval_payload,
                                source="reviewer",
                            ),
                            trusted_approval=True,
                        )
                        approval_payload = parse_shell_approval_required_observation(tool_name, observation)
                tool_message: Message = {
                    "role": "tool",
                    "tool_call_id": tool_call.id or f"call_{step}_{index}",
                    "name": tool_name,
                    "content": observation,
                }
                if approval_payload is not None:
                    del session_messages[batch_session_start:]
                    del messages[batch_model_start:]
                    final = approval_required_final_text(
                        approval_payload,
                        batch_count=len(tool_calls),
                        batch_remaining=len(tool_calls) - index,
                    )
                    self._trace(
                        "agent_waiting_approval",
                        step=step,
                        tool_name=tool_name,
                        command=approval_payload.get("command"),
                        batch_count=len(tool_calls),
                        batch_remaining=len(tool_calls) - index,
                    )
                    return AgentResult(
                        final=final,
                        steps_used=step,
                        model_profile=self.profile.name,
                        used_tools=True,
                        messages=session_messages,
                    )
                session_messages.append(tool_message)
                messages.append(tool_message)
                terminal_text = deterministic_tool_success_final(
                    tool_name,
                    tool_input,
                    observation,
                )
                if terminal_text and len(tool_calls) == 1:
                    deterministic_final = terminal_text

            if deterministic_final:
                final_message: Message = {"role": "assistant", "content": deterministic_final}
                session_messages.append(final_message)
                messages.append(final_message)
                self._trace(
                    "agent_deterministic_final",
                    step=step,
                    used_tools=True,
                    final_chars=len(deterministic_final),
                )
                return AgentResult(
                    final=deterministic_final,
                    steps_used=step,
                    model_profile=self.profile.name,
                    used_tools=True,
                    messages=session_messages,
                )

        final = f"Reached max ReAct steps ({self.max_steps}) without final answer."
        final_message: Message = {"role": "assistant", "content": final}
        session_messages.append(final_message)
        messages.append(final_message)
        self._trace("agent_max_steps", max_steps=self.max_steps, used_tools=used_tools)
        return AgentResult(
            final=final,
            steps_used=self.max_steps,
            model_profile=self.profile.name,
            used_tools=used_tools,
            messages=session_messages,
        )

    def iter_events(self, goal: str) -> Iterator[dict[str, Any]]:
        yield from self.iter_message_events([{"role": "user", "content": goal}])

    def iter_message_events(
        self,
        session_messages: list[Message],
        *,
        system_context: str = "",
    ) -> Iterator[dict[str, Any]]:
        messages: list[Message] = self._model_messages(session_messages, system_context=system_context)
        tool_schemas = self._tool_schemas()
        used_tools = False
        self._trace(
            "agent_start",
            mode="stream",
            session_message_count=len(session_messages),
            model_message_count=len(messages),
            tool_schema_count=len(tool_schemas),
            messages=compact_message_summary(session_messages[-12:]),
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
            try:
                checkpoint_event = self._maybe_compact_active_runtime(messages, step=step)
            except ContextCompactionError as error:
                self._trace("active_runtime_compaction_failed", step=step, error=str(error))
                yield {
                    "event": "error",
                    "message": str(error),
                    "type": type(error).__name__,
                    "detail": "本轮未继续调用模型；完整原始运行轨迹仍保留，可在模型恢复后继续本轮。",
                }
                return
            if checkpoint_event is not None:
                yield checkpoint_event
            try:
                response = yield from self._plan_with_progress(
                    messages=messages,
                    tool_schemas=tool_schemas,
                    step=step,
                )
            except AgentCancelled:
                self._trace("agent_cancelled", step=step)
                raise
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
                    messages.extend([assistant_fix, user_fix])
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
                final_message: Message = {
                    "role": "assistant",
                    "content": final,
                }
                session_messages.append(final_message)
                messages.append(final_message)
                self._trace(
                    "agent_final",
                    step=step,
                    used_tools=used_tools,
                    final_chars=len(str(final_message["content"])),
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
                    "content": str(final_message["content"]),
                    "steps_used": step,
                    "model_profile": self.profile.name,
                    "used_tools": used_tools,
                }
                return

            visible_text = assistant_visible_content(assistant_message).strip()
            yield {"event": "draft_reset", "step": step}
            if visible_text:
                yield {
                    "event": "activity",
                    "phase": "thinking",
                    "title": "实施路径",
                    "detail": visible_text,
                    "activity_type": "work_note",
                    "step": step,
                }
            batch_session_start = len(session_messages)
            batch_model_start = len(messages)
            completed_tool_messages: list[Message] = []
            deterministic_final: str | None = None
            assistant_history = assistant_message_for_history(assistant_message, tool_calls=tool_calls)
            session_messages.append(assistant_history)
            messages.append(assistant_history)
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

                observation = yield from self._execute_tool_with_progress(tool_name, tool_input, step)
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
                        "title": "独立审查智能体正在审批",
                        "detail": "仅审查当前精确动作；固定安全边界不会交给模型改写。",
                        "content": str(approval_payload.get("preview") or ""),
                        "activity_type": "approval_review",
                        "command": str(approval_payload.get("command") or ""),
                        "risk_category": str(approval_payload.get("risk_category") or "EXECUTE"),
                        "step": step,
                        "tool_name": tool_name,
                    }
                    review = self._review_approval(session_messages, approval_payload, step=step)
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
                    )
                    self._raise_if_cancelled()
                    approval_payload = parse_shell_approval_required_observation(tool_name, observation)
                if approval_payload is not None:
                    pending_approval = pending_tool_batch_state(
                        runtime_messages_before_batch=session_messages[:batch_session_start],
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
                    )
                    del session_messages[batch_session_start:]
                    del messages[batch_model_start:]
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
                        "content": final,
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

                tool_message: Message = {
                    "role": "tool",
                    "tool_call_id": tool_call.id or f"call_{step}_{index}",
                    "name": tool_name,
                    "content": observation,
                }
                session_messages.append(tool_message)
                messages.append(tool_message)
                completed_tool_messages.append(tool_message)

                terminal_text = deterministic_tool_success_final(
                    tool_name,
                    tool_input,
                    observation,
                )
                if terminal_text and len(tool_calls) == 1:
                    deterministic_final = terminal_text

            if deterministic_final:
                final_message: Message = {"role": "assistant", "content": deterministic_final}
                session_messages.append(final_message)
                messages.append(final_message)
                self._trace(
                    "agent_deterministic_final",
                    step=step,
                    used_tools=True,
                    final_chars=len(deterministic_final),
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
                    "content": deterministic_final,
                    "steps_used": step,
                    "model_profile": self.profile.name,
                    "used_tools": True,
                    "deterministic_tool_final": True,
                }
                return

        final = f"Reached max ReAct steps ({self.max_steps}) without final answer."
        final_message: Message = {"role": "assistant", "content": final}
        session_messages.append(final_message)
        messages.append(final_message)
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
            "content": final,
            "steps_used": self.max_steps,
            "model_profile": self.profile.name,
            "used_tools": used_tools,
        }

    def iter_approved_tool_batch_events(
        self,
        session_messages: list[Message],
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

        runtime_messages_before_batch = list(session_messages)
        messages = self._model_messages(session_messages, system_context=system_context)
        session_messages.append(assistant_history)
        messages.append(assistant_history)
        for tool_message in completed_tool_messages:
            session_messages.append(tool_message)
            messages.append(tool_message)

        yield {"event": "draft_reset", "step": step}
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
                    "title": "独立审查智能体正在审批",
                    "detail": "仅审查当前精确动作；固定安全边界不会交给模型改写。",
                    "activity_type": "approval_review",
                    "command": str(approval_payload.get("command") or ""),
                    "risk_category": str(approval_payload.get("risk_category") or "EXECUTE"),
                    "step": step,
                    "tool_name": tool_name,
                }
                review = self._review_approval(session_messages, approval_payload, step=step)
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
                )
                self._raise_if_cancelled()
                approval_payload = parse_shell_approval_required_observation(tool_name, observation)
            if approval_payload is not None:
                next_pending = pending_tool_batch_state(
                    runtime_messages_before_batch=runtime_messages_before_batch,
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
                    "content": approval_required_final_text(
                        approval_payload,
                        batch_count=len(tool_calls),
                        batch_remaining=len(tool_calls) - index,
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
            session_messages.append(tool_message)
            messages.append(tool_message)
            completed_tool_messages.append(tool_message)

        # The approved batch is now structurally complete: assistant(tool_calls)
        # is followed by one tool message per tool_call_id. Continue normal
        # ReAct from that state and force used_tools=true on the final event.
        session_messages[:] = repair_runtime_message_sequence(session_messages)
        for event in self.iter_message_events(session_messages, system_context=system_context):
            if event.get("event") == "final":
                event["used_tools"] = True
            yield event

    def _default_system_prompt(self) -> str:
        return (
            "你是本地工作智能体。你可以使用工具读取/写入工作区文件，并调用已注册技能或 MCP 工具。\n"
            "工具定义只通过 API 的 tools 字段提供；你必须使用原生 tool calling 调用工具，"
            "不要在正文中模拟任何工具标签、XML、JSON 或伪协议。\n"
            "必须牢记 ReAct 的终止语义：只输出 assistant content 而不输出 tool_calls，会被运行时立即视为最终答复并结束整个 ReAct。"
            "因此，只要用户要求的工作仍有任何一步未实际执行，或你的正文中还会出现‘我会’‘现在开始’‘下面’‘接下来’‘随后’"
            "等尚待执行的动作，就不得只输出 content；必须在同一条 assistant 消息中同时发起完成下一步所需的原生 tool_calls。"
            "只有任务已经交付并验证、无需工具即可完整回答，或确实需要用户补充信息/批准而无法继续时，才可以只输出 Markdown 正文结束本轮。"
            "最终正文应陈述已经发生并核验的结果，不得用未来时计划冒充交付；不要把整段答复放进代码围栏。\n\n"
            "文件交付任务还有更严格的完成条件：只要用户要求生成、整理、修改或交付文件，而本轮尚未成功执行写入/生成类工具并完成相应核验，"
            "就绝对不得输出 content-only 最终答复。读取文件、打开技能、确定文种或方案、查看工具参数、环境预检以及描述‘会保留/将记录/按某方式处理’，"
            "都不构成交付；必须继续在同一条 assistant 消息中发起实际写入、生成或核验所需的原生 tool_calls。"
            "文件任务的最终答复必须引用已经存在且已核验的产物路径。\n\n"
            "最多工具调用轮数由运行时控制。不要编造工具结果。\n\n"
            "展示规则：在需要工具的复杂工作中，如果你形成了会影响后续理解的路线选择、范围判断、"
            "关键发现或修正，请在发起 tool_calls 的同一条 assistant content 中先写一小段自然语言工作说明；"
            "它是用户可见、可长期保留的实施路径原文。不要逐条播报机械操作，也不要输出隐私思维链。"
            "具体命令、参数、回显和中间观察只应体现在活动/工具调用中。"
            "最终答复只写用户要的结论、摘要、文件路径或下一步建议；不要在最终答复中重复整条过程。\n\n"
            "计划执行规则：简单问答、单一读取、单文件小改和一步即可验证的任务直接处理，不要建立计划。"
            "当任务包含三个及以上相互依赖的动作、跨多个文件或材料、研究后还要形成产物、存在显著不确定性，"
            "或预计需要较长时间执行时，先调用 update_plan 建立 2 至 7 个以结果为导向的步骤，再继续调用实际工具。"
            "计划不是最终答复，也不是 DAG：任何时刻至多一个步骤为 in_progress；完成并验证后标 completed，"
            "再推进下一步。只有新证据使原路线失效时才修改计划；不得为展示进度而频繁改写。复杂任务结束前，"
            "必须把所有已完成步骤更新为 completed；确实无法完成的步骤保留 pending，并在最终答复说明原因。\n\n"
            "终端权限规则：需要终端时调用 shell_exec。只读查看类命令会自动执行；脚本、安装、长任务或写入类命令"
            "会返回 approval_required，并附带风险类别、工作目录、超时和命令预览；"
            "系统会交由独立审查智能体或用户审批；模型不得自行伪造审批状态或重试绕过；"
            "不得因为预计某个后续动作可能需要权限，就提前在正文中要求用户回复‘允许执行’‘确认’等口令。"
            "必须先优先调用现有 core 文件工具或技能专用工具；只有确实必须使用 shell_exec 时，先实际调用 shell_exec，"
            "并且仅当工具真实返回 approval_required 后才暂停。审批由系统审批卡处理，不得用 content-only 正文模拟审批、"
            "不得让用户手工输入许可，也不得把尚未发起的命令描述成待审批状态。"
            "被拒绝的危险命令不能绕过。shell_exec 是 argv 执行，不是完整 shell：不要传 2>&1、管道、重定向或依赖 ls 的通配符；"
            "需要按文件名模式检查时使用 find 或 rg --files。任何非零 returncode 都是失败，最终答复不得把含失败命令的验证概括为“全部通过”。\n\n"
            "运行环境规则：本项目只有一套受支持的 Python 环境，即工作区根目录 `.venv`；"
            "Python、pip、Office 脚本、会议 ASR 和 VAD 都必须使用它。不要创建或调用 `.venv_agent`、"
            "`meeting_audio_minutes/.venv_project`、`.venv_deepfilter`、Conda、系统 Python 或临时 venv，"
            "也不要用 `--user` 安装包。环境检查使用 `scripts/runtime_env.sh check`，缺依赖时说明原因并请求用户批准后"
            "使用 `scripts/runtime_env.sh bootstrap`；Node/npm 使用该脚本的 node/npm 子命令。"
            "FFmpeg、LibreOffice、Poppler 属于声明的原生工具，不代表额外 Python 环境。"
            "DeepFilterNet 因 Python 版本冲突不进入主运行环境，音频降噪默认使用 FFmpeg。\n\n"
            "文件使用规则：如果用户消息、参考附件、历史对话或上一步工具结果里已经出现明确文件路径，"
            "必须优先读取/写入这些明确路径；不要为了确认而先扫描工作区。"
            "修改现有文本文件时，小范围改动优先使用 edit_text_file；跨多处或多文件改动优先使用 apply_unified_patch；"
            "只有创建完整新文件或确实需要重写成品时才使用 write_text_file。"
            "只有缺少路径且任务确实依赖本地文件时，才可以调用 list_workspace_files，并且必须限定到最小目录"
            "（例如 meet_files 或 meet_files/attachments）。"
            "除非用户明确要求查看整个项目结构，否则不要调用 list_workspace_files(path='.') 或扫描工作区根目录。\n\n"
            "技能分层规则：领域任务先根据系统提示中的技能索引判断是否已有对应技能。"
            "匹配时先调用 sys_skill 的 open 读取该技能说明；关闭的技能不能在对话中 activate，"
            "必须提示用户先在网页“技能”页启用并开始新对话。需要技能专用工具时，"
            "用 sys_skill 的 show 查看参数，再用 sys_skill 的 call 执行。"
            "不要猜测或直接调用未出现在顶层 tools 中的技能工具名。"
            "read_text_file、write_text_file、edit_text_file、apply_unified_patch、list_workspace_files 和 shell_exec "
            "是常驻 core 能力，可以直接调用。外部 MCP 能力通过 mcporter 的 list/show/call 分层使用。\n\n"
            f"{self._extra_system_context_block()}"
        )

    def _extra_system_context_block(self) -> str:
        if not self.extra_system_context:
            return ""
        return f"{self.extra_system_context}\n\n"

    def _model_messages(self, session_messages: list[Message], *, system_context: str = "") -> list[Message]:
        messages: list[Message] = [{"role": "system", "content": self.system_prompt}]
        if self.task_plan:
            messages.append(
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
            messages.append({"role": "system", "content": system_context.strip()})
        messages.extend(session_messages)
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
                        "Create or update a short living execution plan for a genuinely complex task. "
                        "Do not use for simple questions or one-step work."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "explanation": {
                                "type": "string",
                                "description": "Only explain a material plan change; otherwise keep empty.",
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
    ) -> str:
        if tool_name == "update_plan":
            return self._apply_task_plan(tool_input)
        tool = self.tools.get_model_tool(tool_name)
        safe_input = dict(tool_input)
        if tool_name == "shell_exec" and not trusted_approval:
            safe_input.pop("approved_by_user", None)
            safe_input.pop("_approval_source", None)
            safe_input.pop("_approval_action_id", None)
            safe_input.pop("_approval_grant", None)
        return str(tool.handler(safe_input))

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
        messages: list[Message],
        *,
        step: int,
    ) -> dict[str, Any] | None:
        estimated = estimate_messages_tokens(messages)
        if estimated < ACTIVE_REACT_CHECKPOINT_TRIGGER_TOKENS:
            return None
        user_indexes = [index for index, message in enumerate(messages) if message.get("role") == "user"]
        if not user_indexes:
            return None
        active_start = user_indexes[-1]
        active_messages = messages[active_start:]
        # A user prompt by itself has no completed implementation path to fold.
        if len(active_messages) < 3:
            return None
        checkpoint = summarize_active_react_checkpoint(
            self.client,
            self.profile,
            active_messages,
            task_plan=self.task_plan,
        )
        if not checkpoint:
            return None
        original_count = len(active_messages)
        messages[active_start:] = [
            active_messages[0],
            {
                "role": "assistant",
                "content": (
                    "本轮运行上下文已压缩。以下是继续执行所需的高保真检查点；"
                    "它不是最终答复：\n\n" + checkpoint
                ),
            },
        ]
        self._trace(
            "active_runtime_compacted",
            step=step,
            estimated_tokens=estimated,
            original_message_count=original_count,
            checkpoint_chars=len(checkpoint),
        )
        return {
            "event": "activity",
            "phase": "thinking",
            "title": "压缩运行上下文",
            "detail": (
                f"本轮约 {estimated} tokens；已将 {original_count} 条正在执行的 ReAct 消息整理为检查点，"
                "完整原始轨迹仍保留在后端日志。"
            ),
            "activity_type": "runtime_summary",
            "step": step,
        }

    def _execute_tool_with_progress(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        step: int,
        *,
        trusted_approval: bool = False,
    ) -> Iterator[dict[str, Any]]:
        result_queue: queue.Queue[str] = queue.Queue(maxsize=1)
        progress_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        started_at = time.monotonic()
        progress_seen = False
        self._trace("tool_start", step=step, tool_name=tool_name, arguments=tool_input)

        def run_tool() -> None:
            previous_sink = set_tool_progress_sink(progress_queue.put)
            try:
                observation = self._execute_model_tool(
                    tool_name,
                    tool_input,
                    trusted_approval=trusted_approval,
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
                    yield event
                return observation

    def _plan_with_progress(
        self,
        *,
        messages: list[Message],
        tool_schemas: list[dict[str, Any]],
        step: int,
    ) -> Iterator[dict[str, Any]]:
        result_queue: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)
        progress_queue: queue.Queue[dict[str, str]] = queue.Queue()
        started_at = time.monotonic()
        last_stream_at: float | None = None
        recovery_started_at: float | None = None
        recovery_last_stream_at: float | None = None
        request_cancel_event = threading.Event()
        stream_idle_timeout = max(
            1,
            min(self.profile.timeout_seconds, MODEL_STREAM_IDLE_TIMEOUT_SECONDS),
        )
        active_stream_max_timeout = max(
            self.profile.timeout_seconds * MODEL_STREAM_MAX_MULTIPLIER,
            self.profile.timeout_seconds + MODEL_STREAM_MAX_EXTENSION_SECONDS,
        )
        recovery_timeout = max(
            10,
            min(RECOVERY_REQUEST_TIMEOUT_SECONDS, self.profile.timeout_seconds),
        )
        recovery_active_max_timeout = max(
            recovery_timeout * MODEL_RECOVERY_MAX_MULTIPLIER,
            recovery_timeout + MODEL_STREAM_MAX_EXTENSION_SECONDS,
        )
        request_id = f"model-plan-{step}-{int(started_at * 1000)}"
        self._trace(
            "llm_start",
            step=step,
            request_id=request_id,
            profile=self.profile.name,
            model=self.profile.model,
            message_count=len(messages),
            tool_schema_count=len(tool_schemas),
            start_timeout_seconds=self.profile.timeout_seconds,
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
                    if status == "recovery_started":
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

                response = self.client.chat_tools_stream(
                    messages,
                    profile=self.profile,
                    reasoning_effort=self.reasoning_effort,
                    tools=tool_schemas,
                    tool_choice="auto",
                    on_delta=on_delta,
                    cancel_event=request_cancel_event,
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
                if delta_status == "recovery_started":
                    if draft_content_chars:
                        yield {"event": "draft_reset", "step": step}
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
                        "append_mode": "replace",
                        "step": step,
                    }
                if (
                    len(content_buffer) > draft_content_chars
                    and not tool_name_buffer
                    and not tool_arguments_buffer
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
                        recovery_last_stream_at is None
                        and recovery_elapsed
                        >= recovery_timeout + MODEL_RECOVERY_TIMEOUT_GRACE_SECONDS
                    ):
                        timeout_kind = "recovery_start"
                        value = RuntimeError(
                            f"模型兼容恢复在 {recovery_timeout} 秒内没有开始返回有效流，"
                            "系统已停止等待；本轮已完成的工具结果仍会保留。"
                        )
                    elif (
                        recovery_last_stream_at is not None
                        and now - recovery_last_stream_at >= stream_idle_timeout
                    ):
                        timeout_kind = "recovery_idle"
                        value = RuntimeError(
                            f"模型恢复流已连续 {stream_idle_timeout} 秒没有新内容，系统已停止等待；"
                            "本轮已完成的工具结果仍会保留。"
                        )
                    elif (
                        recovery_last_stream_at is not None
                        and recovery_elapsed >= recovery_active_max_timeout
                    ):
                        timeout_kind = "recovery_max_active"
                        value = RuntimeError(
                            f"模型恢复流虽持续返回，但已达到 {recovery_active_max_timeout} 秒安全上限，"
                            "本轮已完成的工具结果仍会保留。"
                        )
                elif last_stream_at is None and elapsed >= self.profile.timeout_seconds:
                    timeout_kind = "start"
                    value = RuntimeError(
                        f"模型在 {self.profile.timeout_seconds} 秒内没有开始返回有效流，系统已停止等待；"
                        "本轮已完成的工具结果仍会保留。"
                    )
                elif last_stream_at is not None and now - last_stream_at >= stream_idle_timeout:
                    timeout_kind = "idle"
                    value = RuntimeError(
                        f"模型流已连续 {stream_idle_timeout} 秒没有新内容，系统已停止等待；"
                        "本轮已完成的工具结果仍会保留。"
                    )
                elif last_stream_at is not None and elapsed >= active_stream_max_timeout:
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
                            int((now - last_stream_at) * 1000)
                            if last_stream_at is not None
                            else None
                        ),
                        recovery_elapsed_ms=(
                            int((now - recovery_started_at) * 1000)
                            if recovery_started_at is not None
                            else None
                        ),
                        recovery_idle_ms=(
                            int((now - recovery_last_stream_at) * 1000)
                            if recovery_last_stream_at is not None
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
                if delta_status == "recovery_started":
                    if draft_content_chars:
                        yield {"event": "draft_reset", "step": step}
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
                        "append_mode": "replace",
                        "step": step,
                    }
                if (
                    len(content_buffer) > draft_content_chars
                    and not tool_name_buffer
                    and not tool_arguments_buffer
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
                        "append_mode": "replace",
                        "step": step,
                    }
                    raise RuntimeError(failure_message)
                if final_answer_without_tools:
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
    if status == "recovery_started":
        headline = f"[{elapsed_seconds}s] 主流已结束，正在启动流式恢复。"
    elif status == "recovery_streaming":
        headline = f"[{elapsed_seconds}s] 恢复流正在返回。"
    else:
        headline = f"[{elapsed_seconds}s] 模型正在流式返回。"
    lines = [headline]
    if status in {"recovery_started", "recovery_streaming"}:
        lines.append("当前请求未形成完整决策，系统正在自动恢复；已完成的工具结果会继续保留。")
        if status_detail:
            lines.append(f"主流结束信息：{status_detail}")

    if reasoning:
        lines.extend(
            [
                "",
                "--- 模型思考 ---",
                compact_reasoning_preview(reasoning),
            ]
        )

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


def response_message(raw: dict[str, Any]) -> dict[str, Any]:
    choice = (raw.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    return message if isinstance(message, dict) else {}


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
    return {
        "finish_reason": choice.get("finish_reason") if isinstance(choice, dict) else None,
        "content_chars": len(str(message.get("content") or content or "")),
        "native_tool_call_count": count_native_tool_calls(message),
        "tool_names": [
            str((call.get("function") or {}).get("name") or "")
            for call in (message.get("tool_calls") or [])
            if isinstance(call, dict)
        ],
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
            arguments = parse_tool_arguments(function.get("arguments") or raw_call.get("arguments") or {})
            calls.append(
                NativeToolCall(
                    id=str(raw_call.get("id") or f"call_{index}"),
                    name=name,
                    arguments=arguments,
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
        calls.append(
            NativeToolCall(
                id=(attrs.get("tool_call_id") or attrs.get("id") or f"call_text_{index}").strip(),
                name=name,
                arguments=parse_tool_arguments(raw_arguments),
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


def assistant_message_for_history(
    message: dict[str, Any],
    *,
    tool_calls: list[NativeToolCall] | None = None,
) -> Message:
    clean: Message = {
        "role": "assistant",
        "content": assistant_visible_content(message) if tool_calls else (message.get("content") or ""),
    }
    raw_tool_calls = message.get("tool_calls")
    if raw_tool_calls:
        clean["tool_calls"] = raw_tool_calls
    elif tool_calls:
        clean["tool_calls"] = [
            {
                "id": tool_call.id or f"call_{index}",
                "type": "function",
                "function": {
                    "name": tool_call.name,
                    "arguments": json.dumps(tool_call.arguments, ensure_ascii=False),
                },
            }
            for index, tool_call in enumerate(tool_calls)
        ]
    reasoning_content = str(message.get("reasoning_content") or "")
    if clean.get("tool_calls") and reasoning_content:
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
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    text = str(value or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = json.loads(escape_raw_control_chars_in_strings(text))
        except json.JSONDecodeError:
            try:
                parsed = parse_react_json(text)
            except ValueError:
                return {"value": text}
    if isinstance(parsed, dict):
        return parsed
    return {"value": parsed}


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
    auto_approve: bool = False,
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

    lines = [f"已完成并核验保存{date_text + ' ' if date_text else ''}{report_label}。"]
    if path:
        lines.append(f"文件：`{path}`")
    lines.append(f"证据覆盖：{coverage}。")
    if payload.get("needs_user_input") is True:
        lines.append("当前版本已保存；仍有线下或外部工作信息需要补充。")
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
