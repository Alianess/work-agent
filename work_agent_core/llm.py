from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterator
import json
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from .config import ModelProfile


Message = dict[str, Any]
REASONING_EFFORTS = {"light", "medium", "high", "very_high"}
STREAM_IDLE_TIMEOUT_SECONDS = 45
RECOVERY_REQUEST_TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class LLMResponse:
    content: str
    raw: dict[str, Any]


@dataclass(frozen=True)
class LLMStreamChunk:
    content: str = ""
    reasoning: str = ""
    tool_name: str = ""
    tool_arguments: str = ""
    status: str = ""
    status_detail: str = ""


def chat_completions_endpoint(base_url: str) -> str:
    endpoint = str(base_url).rstrip("/")
    if endpoint.endswith("/chat/completions"):
        return endpoint
    return f"{endpoint}/chat/completions"


def normalize_reasoning_effort(value: str | None) -> str:
    effort = str(value or "medium").strip().lower().replace("-", "_")
    return effort if effort in REASONING_EFFORTS else "medium"


def reduced_recovery_reasoning_effort(value: str | None) -> str:
    """Recovery should produce an executable result, not repeat deep planning."""
    normalize_reasoning_effort(value)
    return "light"


def is_deepseek_profile(profile: ModelProfile) -> bool:
    identity = " ".join([profile.name, profile.provider, profile.base_url, profile.model]).lower()
    return "deepseek" in identity


def should_prefer_direct_connection(profile: ModelProfile) -> bool:
    """Identify the official DeepSeek endpoint for proxy-independent access.

    Official DeepSeek API requests must not be handed to HTTP_PROXY or
    HTTPS_PROXY.  Model selection is a user decision and is not a substitute
    for a stable route to this endpoint.
    """
    host = (urllib.parse.urlparse(chat_completions_endpoint(profile.base_url)).hostname or "").lower()
    return is_deepseek_profile(profile) and host in {"api.deepseek.com", "api.deepseek.cn"}


def supports_reasoning_effort(profile: ModelProfile) -> bool:
    identity = " ".join([profile.name, profile.provider, profile.model]).lower()
    return is_deepseek_profile(profile) or any(
        marker in identity for marker in ("gpt-5", "o3", "o4")
    )


def apply_reasoning_controls(
    payload: dict[str, Any],
    *,
    profile: ModelProfile,
    reasoning_effort: str | None,
) -> dict[str, Any]:
    """Map the UI's four generic levels to provider-specific API controls."""
    if reasoning_effort is None or not supports_reasoning_effort(profile):
        return payload
    effort = normalize_reasoning_effort(reasoning_effort)
    if is_deepseek_profile(profile):
        if effort == "light":
            payload["thinking"] = {"type": "disabled"}
            return payload
        payload.pop("temperature", None)  # DeepSeek ignores sampling controls in thinking mode.
        payload["thinking"] = {"type": "enabled"}
        payload["reasoning_effort"] = "max" if effort == "very_high" else "high"
        return payload
    payload["reasoning_effort"] = {
        "light": "low",
        "medium": "medium",
        "high": "high",
        "very_high": "max",
    }[effort]
    return payload


class OpenAICompatibleClient:
    """Minimal OpenAI-compatible chat completions client.

    This deliberately uses the standard library so any endpoint that implements
    `/chat/completions` can be added through `config/model_profiles.json`.
    """

    def __init__(self) -> None:
        # An empty ProxyHandler is the documented urllib way to bypass both
        # upper- and lower-case proxy environment variables for one request.
        self._direct_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _open_request(
        self,
        request: urllib.request.Request,
        *,
        profile: ModelProfile,
        timeout: float | int,
    ) -> Any:
        """Open one request without changing the selected model or route.

        The official DeepSeek endpoint always uses the no-proxy opener.  It
        never tries the process proxy first and never falls back to it after a
        direct failure.  This also avoids reusing a Request that urllib has
        mutated with proxy/tunnel state.
        """
        if should_prefer_direct_connection(profile):
            return self._direct_opener.open(request, timeout=timeout)
        return urllib.request.urlopen(request, timeout=timeout)

    def chat(
        self,
        messages: list[Message],
        *,
        profile: ModelProfile,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        reasoning_effort: str | None = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": profile.model,
            "messages": prepare_messages_for_profile(messages, profile),
            "temperature": profile.temperature if temperature is None else temperature,
            "max_tokens": profile.max_tokens if max_tokens is None else max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"
        apply_reasoning_controls(payload, profile=profile, reasoning_effort=reasoning_effort)

        endpoint = chat_completions_endpoint(profile.base_url)
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=data,
            headers={
                "Authorization": f"Bearer {profile.api_key()}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with self._open_request(request, profile=profile, timeout=profile.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM request failed with HTTP {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"LLM request failed: {error}") from error
        except (TimeoutError, socket.timeout) as error:
            raise RuntimeError(
                llm_timeout_message(
                    action="request",
                    profile=profile,
                    endpoint=endpoint,
                    timeout_seconds=profile.timeout_seconds,
                )
            ) from error
        except OSError as error:
            if "timed out" in str(error).lower():
                raise RuntimeError(
                    llm_timeout_message(
                        action="request",
                        profile=profile,
                        endpoint=endpoint,
                        timeout_seconds=profile.timeout_seconds,
                    )
                ) from error
            raise

        parsed = json.loads(body)
        content = coerce_content_text(parsed["choices"][0]["message"].get("content"))
        return LLMResponse(content=content, raw=parsed)

    def chat_stream(
        self,
        messages: list[Message],
        *,
        profile: ModelProfile,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        for chunk in self.chat_stream_chunks(
            messages,
            profile=profile,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            if chunk.content:
                yield chunk.content

    def chat_stream_chunks(
        self,
        messages: list[Message],
        *,
        profile: ModelProfile,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[LLMStreamChunk]:
        yield from self._chat_stream_chunks_with_retry(
            messages,
            profile=profile,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def chat_tools_stream(
        self,
        messages: list[Message],
        *,
        profile: ModelProfile,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        on_delta: Any | None = None,
        reasoning_effort: str | None = None,
        cancel_event: threading.Event | None = None,
        _allow_recovery: bool = True,
        _request_usage: bool = True,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": profile.model,
            "messages": prepare_messages_for_profile(messages, profile),
            "temperature": profile.temperature if temperature is None else temperature,
            "max_tokens": profile.max_tokens if max_tokens is None else max_tokens,
            "stream": True,
        }
        if _request_usage:
            payload["stream_options"] = {"include_usage": True}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"
        apply_reasoning_controls(payload, profile=profile, reasoning_effort=reasoning_effort)

        endpoint = chat_completions_endpoint(profile.base_url)
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=data,
            headers={
                "Authorization": f"Bearer {profile.api_key()}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
            method="POST",
        )
        started_at = time.monotonic()
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: dict[int, dict[str, Any]] = {}
        finish_reason = None
        usage: dict[str, Any] = {}
        stream_error: RuntimeError | None = None
        idle_timeout_seconds = min(profile.timeout_seconds, STREAM_IDLE_TIMEOUT_SECONDS)
        response_finished = threading.Event()
        try:
            with self._open_request(request, profile=profile, timeout=idle_timeout_seconds) as response:
                def close_response_when_cancelled() -> None:
                    if cancel_event is None:
                        return
                    while not response_finished.wait(0.1):
                        if not cancel_event.is_set():
                            continue
                        close = getattr(response, "close", None)
                        if callable(close):
                            try:
                                close()
                            except Exception:
                                pass
                        return

                cancel_watcher = threading.Thread(
                    target=close_response_when_cancelled,
                    name="work-agent-llm-stream-cancel",
                    daemon=True,
                )
                cancel_watcher.start()
                for raw_line in response:
                    if cancel_event is not None and cancel_event.is_set():
                        raise RuntimeError("模型流请求已取消。")
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_line = line.removeprefix("data:").strip()
                    if data_line == "[DONE]":
                        break
                    parsed = json.loads(data_line)
                    if isinstance(parsed.get("usage"), dict):
                        usage = dict(parsed["usage"])
                    choice = (parsed.get("choices") or [{}])[0]
                    finish_reason = choice.get("finish_reason") or finish_reason
                    delta = choice.get("delta") or {}
                    choice_message = choice.get("message") or {}
                    if not isinstance(choice_message, dict):
                        choice_message = {}
                    content = coerce_content_text(delta.get("content") or choice.get("text"))
                    # A few OpenAI-compatible relays stream reasoning deltas,
                    # then put the final answer only in a full `message` object
                    # on the last SSE frame. Do not silently drop that answer.
                    if not content and not content_parts:
                        content = coerce_content_text(choice_message.get("content"))
                    reasoning = extract_reasoning_delta(delta, choice)
                    if not reasoning and choice_message:
                        reasoning = extract_reasoning_delta(choice_message, {})
                    if content:
                        content_parts.append(content)
                    if reasoning:
                        reasoning_parts.append(reasoning)
                    tool_name_delta = ""
                    tool_args_delta = ""
                    raw_tool_calls = delta.get("tool_calls") or []
                    if not raw_tool_calls and not tool_calls:
                        raw_tool_calls = choice_message.get("tool_calls") or []
                    for raw_call in raw_tool_calls:
                        if not isinstance(raw_call, dict):
                            continue
                        index = int(raw_call.get("index") or 0)
                        current = tool_calls.setdefault(
                            index,
                            {
                                "id": str(raw_call.get("id") or f"call_{index}"),
                                "type": str(raw_call.get("type") or "function"),
                                "function": {"name": "", "arguments": ""},
                            },
                        )
                        if raw_call.get("id"):
                            current["id"] = str(raw_call.get("id"))
                        if raw_call.get("type"):
                            current["type"] = str(raw_call.get("type"))
                        function = raw_call.get("function") or {}
                        if not isinstance(function, dict):
                            continue
                        name_part = str(function.get("name") or "")
                        args_part = str(function.get("arguments") or "")
                        if name_part:
                            current["function"]["name"] += name_part
                            tool_name_delta += name_part
                        if args_part:
                            current["function"]["arguments"] += args_part
                            tool_args_delta += args_part
                    if on_delta and (content or reasoning or tool_name_delta or tool_args_delta):
                        on_delta(
                            LLMStreamChunk(
                                content=content,
                                reasoning=reasoning,
                                tool_name=tool_name_delta,
                                tool_arguments=tool_args_delta,
                            )
                        )
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            stream_error = RuntimeError(f"LLM stream failed with HTTP {error.code}: {detail}")
            if error.code == 400 and _request_usage and stream_usage_option_rejected(detail):
                return self.chat_tools_stream(
                    messages,
                    profile=profile,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools,
                    tool_choice=tool_choice,
                    on_delta=on_delta,
                    reasoning_effort=reasoning_effort,
                    cancel_event=cancel_event,
                    _allow_recovery=_allow_recovery,
                    _request_usage=False,
                )
            if error.code not in {408, 429, 500, 502, 503, 504}:
                raise stream_error from error
        except urllib.error.URLError as error:
            stream_error = RuntimeError(f"LLM stream failed: {error}")
        except AttributeError as error:
            # Some OpenAI-compatible proxy stacks leak an internal
            # ``NoneType.peek`` parser error instead of an URLError. Treat only
            # that known transport symptom as recoverable; do not hide other
            # programming errors behind a model retry.
            if "peek" not in str(error).lower():
                raise
            stream_error = RuntimeError(
                f"LLM stream transport parser failed: {error}"
            )
        except (TimeoutError, socket.timeout) as error:
            stream_error = RuntimeError(
                llm_timeout_message(
                    action="stream idle",
                    profile=profile,
                    endpoint=endpoint,
                    timeout_seconds=idle_timeout_seconds,
                )
            )
        except OSError as error:
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("模型流请求已取消。") from error
            if "timed out" in str(error).lower():
                stream_error = RuntimeError(
                    llm_timeout_message(
                        action="stream idle",
                        profile=profile,
                        endpoint=endpoint,
                        timeout_seconds=idle_timeout_seconds,
                    )
                )
            else:
                raise
        finally:
            response_finished.set()

        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("模型流请求已取消。")
        if stream_error is not None:
            if not _allow_recovery:
                raise stream_error
            return self._recover_tools_response(
                messages,
                profile=profile,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                tool_choice=tool_choice,
                reasoning_effort=reasoning_effort,
                started_at=started_at,
                reason="stream_interrupted",
                cause=stream_error,
                on_delta=on_delta,
                cancel_event=cancel_event,
                primary_finish_reason=finish_reason,
                primary_usage=usage,
                request_usage=_request_usage,
            )

        valid_tool_calls = [
            tool_calls[index]
            for index in sorted(tool_calls)
            if tool_calls[index].get("function", {}).get("name")
        ]
        message: dict[str, Any] = {
            "role": "assistant",
            "content": "".join(content_parts),
        }
        if valid_tool_calls:
            if reasoning_parts:
                message["reasoning_content"] = "".join(reasoning_parts)
            message["tool_calls"] = valid_tool_calls
        raw = {
            "choices": [
                {
                    "finish_reason": finish_reason,
                    "message": message,
                }
            ]
        }
        if usage:
            raw["usage"] = usage
        if not message["content"].strip() and not valid_tool_calls:
            diagnostics = stream_end_diagnostics(finish_reason, usage)
            if not _allow_recovery:
                raise RuntimeError(
                    "模型恢复流结束后仍没有返回正文或工具调用"
                    f"（{diagnostics}）。"
                )
            return self._recover_tools_response(
                messages,
                profile=profile,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                tool_choice=tool_choice,
                reasoning_effort=reasoning_effort,
                started_at=started_at,
                reason="empty_stream",
                cause=None,
                on_delta=on_delta,
                cancel_event=cancel_event,
                primary_finish_reason=finish_reason,
                primary_usage=usage,
                request_usage=_request_usage,
            )
        return LLMResponse(content=message["content"], raw=raw)

    def _recover_tools_response(
        self,
        messages: list[Message],
        *,
        profile: ModelProfile,
        temperature: float | None,
        max_tokens: int | None,
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        reasoning_effort: str | None,
        started_at: float,
        reason: str,
        cause: Exception | None,
        on_delta: Any | None,
        cancel_event: threading.Event | None,
        primary_finish_reason: str | None = None,
        primary_usage: dict[str, Any] | None = None,
        request_usage: bool = True,
    ) -> LLMResponse:
        elapsed_seconds = max(0, int(time.monotonic() - started_at))
        # The profile timeout governs how long the primary stream may remain
        # silent before it starts. Once the provider has actively streamed
        # reasoning, subtracting the whole primary-stream duration from that
        # timeout incorrectly makes recovery impossible. Recovery is a separate,
        # bounded phase with its own request budget.
        recovery_timeout = max(
            10,
            min(RECOVERY_REQUEST_TIMEOUT_SECONDS, profile.timeout_seconds),
        )
        recovery_profile = replace(profile, timeout_seconds=recovery_timeout)
        recovery_effort = reduced_recovery_reasoning_effort(reasoning_effort)
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("模型流请求已取消。") from cause
        if on_delta:
            on_delta(
                LLMStreamChunk(
                    status="recovery_started",
                    status_detail=stream_end_diagnostics(
                        primary_finish_reason,
                        primary_usage,
                    ),
                )
            )

        recovery_instruction = (
            "恢复阶段：不要重新分析此前已经完成的文件读取和工具结果。"
            "请立即返回下一步原生 tool_call；如果无需工具，则直接返回最终答复。"
            "不要只返回内部推理。"
        )
        recovery_messages = [dict(message) for message in messages]
        if recovery_messages and recovery_messages[0].get("role") == "system":
            recovery_messages[0]["content"] = (
                f"{str(recovery_messages[0].get('content') or '').rstrip()}\n\n"
                f"{recovery_instruction}"
            )
        else:
            recovery_messages.insert(0, {"role": "system", "content": recovery_instruction})

        def forward_recovery_delta(chunk: LLMStreamChunk) -> None:
            if on_delta:
                on_delta(replace(chunk, status="recovery_streaming"))

        recovery_error: Exception | None = None
        try:
            recovered = self.chat_tools_stream(
                recovery_messages,
                profile=recovery_profile,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                tool_choice=tool_choice,
                reasoning_effort=recovery_effort,
                on_delta=forward_recovery_delta,
                cancel_event=cancel_event,
                _allow_recovery=False,
                _request_usage=request_usage,
            )
        except Exception as error:
            recovery_error = error

        if recovery_error is not None:
            prefix = "模型流式响应中断" if cause else "模型流式响应没有正文或工具调用"
            raise RuntimeError(
                f"{prefix}，当前模型流式恢复也失败：{recovery_error}"
            ) from recovery_error

        recovered_message = (recovered.raw.get("choices") or [{}])[0].get("message") or {}
        recovered_calls = (
            recovered_message.get("tool_calls") if isinstance(recovered_message, dict) else []
        )
        if not recovered.content.strip() and not (
            isinstance(recovered_calls, list) and recovered_calls
        ):
            raise RuntimeError(
                "模型自动恢复请求结果仍为空，没有返回正文或工具调用，请重试；"
                "如需更换模型，请在设置中手动选择。"
            )

        recovered.raw["_work_agent"] = {
            "recovery": {
                "mode": "stream",
                "reason": reason,
                "reasoning_effort": recovery_effort,
                "timeout_seconds": recovery_timeout,
                "primary_stream_elapsed_seconds": elapsed_seconds,
                "primary_finish_reason": primary_finish_reason,
                "primary_usage": primary_usage or {},
            }
        }
        return recovered

    def _chat_stream_chunks_with_retry(
        self,
        messages: list[Message],
        *,
        profile: ModelProfile,
        temperature: float | None,
        max_tokens: int | None,
    ) -> Iterator[LLMStreamChunk]:
        last_error: Exception | None = None
        for attempt in range(2):
            emitted_any = False
            try:
                for chunk in self._chat_stream_chunks_once(
                    messages,
                    profile=profile,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ):
                    emitted_any = True
                    yield chunk
                return
            except RuntimeError as error:
                last_error = error
                if emitted_any:
                    raise
                if attempt == 0 and is_retryable_stream_error(error):
                    time.sleep(0.6)
                    continue
                break

        try:
            response = self.chat(
                messages,
                profile=profile,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as recovery_error:
            if last_error is not None:
                raise RuntimeError(
                    f"{last_error}; same-model non-stream recovery also failed: {recovery_error}"
                ) from recovery_error
            raise
        if response.content:
            yield LLMStreamChunk(content=response.content)

    def _chat_stream_chunks_once(
        self,
        messages: list[Message],
        *,
        profile: ModelProfile,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[LLMStreamChunk]:
        payload: dict[str, Any] = {
            "model": profile.model,
            "messages": prepare_messages_for_profile(messages, profile),
            "temperature": profile.temperature if temperature is None else temperature,
            "max_tokens": profile.max_tokens if max_tokens is None else max_tokens,
            "stream": True,
        }

        endpoint = chat_completions_endpoint(profile.base_url)
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=data,
            headers={
                "Authorization": f"Bearer {profile.api_key()}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
            method="POST",
        )
        try:
            with self._open_request(request, profile=profile, timeout=profile.timeout_seconds) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_line = line.removeprefix("data:").strip()
                    if data_line == "[DONE]":
                        break
                    parsed = json.loads(data_line)
                    choice = (parsed.get("choices") or [{}])[0]
                    delta = choice.get("delta") or {}
                    content = delta.get("content") or choice.get("text") or ""
                    reasoning = extract_reasoning_delta(delta, choice)
                    if content or reasoning:
                        yield LLMStreamChunk(content=str(content or ""), reasoning=reasoning)
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM stream failed with HTTP {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"LLM stream failed: {error}") from error
        except (TimeoutError, socket.timeout) as error:
            raise RuntimeError(
                llm_timeout_message(
                    action="stream",
                    profile=profile,
                    endpoint=endpoint,
                    timeout_seconds=profile.timeout_seconds,
                )
            ) from error
        except OSError as error:
            if "timed out" in str(error).lower():
                raise RuntimeError(
                    llm_timeout_message(
                        action="stream",
                        profile=profile,
                        endpoint=endpoint,
                        timeout_seconds=profile.timeout_seconds,
                    )
                ) from error
            raise


def extract_reasoning_delta(delta: dict[str, Any], choice: dict[str, Any]) -> str:
    for container in (delta, choice):
        for key in (
            "reasoning_content",
            "reasoning",
            "reasoning_summary",
            "thinking",
            "thought",
        ):
            value = container.get(key)
            text = stringify_reasoning_value(value)
            if text:
                return text
    return ""


def coerce_content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""
    parts: list[str] = []
    for item in value:
        if isinstance(item, str):
            parts.append(item)
            continue
        if not isinstance(item, dict):
            continue
        text = item.get("text") or item.get("content")
        if isinstance(text, str):
            parts.append(text)
        elif isinstance(text, dict) and isinstance(text.get("value"), str):
            parts.append(text["value"])
    return "".join(parts)


def stream_end_diagnostics(
    finish_reason: str | None,
    usage: dict[str, Any] | None,
) -> str:
    reason = str(finish_reason or "unknown")
    details = [f"finish_reason={reason}"]
    if isinstance(usage, dict) and usage:
        for key in (
            "completion_tokens",
            "reasoning_tokens",
            "output_tokens",
            "total_tokens",
        ):
            value = usage.get(key)
            if value is not None:
                details.append(f"{key}={value}")
        completion_details = usage.get("completion_tokens_details")
        if isinstance(completion_details, dict):
            reasoning_tokens = completion_details.get("reasoning_tokens")
            if reasoning_tokens is not None and not any(
                item.startswith("reasoning_tokens=") for item in details
            ):
                details.append(f"reasoning_tokens={reasoning_tokens}")
    return ", ".join(details)


def stream_usage_option_rejected(detail: str) -> bool:
    text = str(detail or "").lower()
    return "stream_options" in text or "include_usage" in text


def prepare_messages_for_profile(
    messages: list[Message],
    profile: ModelProfile,
) -> list[Message]:
    """Repair legacy DeepSeek tool-call history for thinking-mode replay."""
    if not profile_requires_reasoning_content(profile):
        return messages

    prepared: list[Message] = []
    for message in messages:
        if not isinstance(message, dict):
            prepared.append(message)
            continue
        clean = dict(message)
        if (
            clean.get("role") == "assistant"
            and clean.get("tool_calls")
            and not str(clean.get("reasoning_content") or "").strip()
        ):
            clean["reasoning_content"] = (
                "Tool-call reasoning was not retained by an earlier client version."
            )
        prepared.append(clean)
    return prepared


def profile_requires_reasoning_content(profile: ModelProfile) -> bool:
    identity = " ".join(
        [profile.name, profile.provider, profile.base_url, profile.model]
    ).lower()
    return "deepseek" in identity


def llm_timeout_message(
    *,
    action: str,
    profile: ModelProfile,
    endpoint: str,
    timeout_seconds: int,
) -> str:
    return (
        f"LLM {action} timed out after {timeout_seconds}s "
        f"(profile={profile.name}, model={profile.model}, endpoint={endpoint}). "
        "这通常是模型服务响应过慢或网络连接中断，不是本地文件库读取失败。"
    )


def is_retryable_stream_error(error: Exception) -> bool:
    text = str(error).lower()
    retryable_markers = (
        "unexpected_eof",
        "eof occurred",
        "connection reset",
        "connection aborted",
        "remote end closed",
        "temporarily unavailable",
        "service unavailable",
        "http 503",
        "timed out",
        "timeout",
    )
    return any(marker in text for marker in retryable_markers)


def stringify_reasoning_value(value: Any) -> str:
    if not value:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
        return "".join(parts)
    if isinstance(value, dict):
        return str(value.get("text") or value.get("content") or "")
    return ""
