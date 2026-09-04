from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Iterator
import ipaddress
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
"""流开始返回之前允许的静默时长。

对纯文本够用，对"深度思考 + 附件"远远不够：模型在出第一个 token 之前要先读完
几张图和一份文档。实测 45 秒把 4 轮正常请求判成了超时。profile 可以调高它。
"""
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


def endpoint_host(profile: ModelProfile) -> str:
    return (urllib.parse.urlparse(chat_completions_endpoint(profile.base_url)).hostname or "").lower()


DEEPSEEK_OFFICIAL_HOSTS = frozenset({"api.deepseek.com", "api.deepseek.cn"})


def is_deepseek_profile(profile: ModelProfile) -> bool:
    """参数方言属于**端点**，不属于模型名。

    DeepSeek 官方要 thinking:{type:enabled} 加 reasoning_effort:max；同一个
    deepseek-v4-flash 挂在商汤的 token.sensenova.cn 上，要的却是普通的
    reasoning_effort:low/medium/high/none。按模型名认，就会把官方的方言发给
    转售方，参数直接被拒。
    """

    return endpoint_host(profile) in DEEPSEEK_OFFICIAL_HOSTS


def is_local_or_private_endpoint(profile: ModelProfile) -> bool:
    """Return whether the endpoint is reached over a local/private network."""

    host = endpoint_host(profile).rstrip(".")
    if host == "localhost" or host.endswith((".localhost", ".local", ".ts.net")):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    tailscale_cgnat = ipaddress.ip_network("100.64.0.0/10")
    return bool(
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address in tailscale_cgnat
    )


def should_prefer_direct_connection(profile: ModelProfile) -> bool:
    """Identify endpoints that must never be sent through the process proxy.

    Official DeepSeek and local/private endpoints need a stable direct route.
    Tailscale IPv4 addresses live in 100.64.0.0/10: handing them to a desktop
    HTTP proxy produces a proxy-generated blank HTTP 503 even when the peer is
    reachable directly.
    """

    return is_deepseek_profile(profile) or is_local_or_private_endpoint(profile)


def recovery_request_timeout_seconds(profile: ModelProfile) -> int:
    """Give a local model enough time to prefill the same context again.

    A cloud recovery is deliberately short. A local/private model can spend
    longer than that merely rebuilding a large prompt KV cache, so a fixed
    60-second recovery turns a recoverable transport interruption into a
    guaranteed second timeout.
    """

    if is_local_or_private_endpoint(profile):
        return max(10, stream_start_timeout_seconds(profile))
    return max(10, min(RECOVERY_REQUEST_TIMEOUT_SECONDS, profile.timeout_seconds))


def stream_start_timeout_seconds(profile: ModelProfile) -> int:
    """Return the budget for connect, queueing, and prompt prefill.

    ``stream_idle_timeout_seconds`` used to be passed directly to urllib,
    accidentally shortening a larger request timeout. Treat an explicit value
    only as an extension: it must never reduce the model's start budget.
    """

    configured = int(getattr(profile, "stream_idle_timeout_seconds", 0) or 0)
    return max(1, profile.timeout_seconds, configured)


def is_dots_profile(profile: ModelProfile) -> bool:
    identity = " ".join([profile.name, profile.provider, profile.model]).lower()
    return "dots" in identity or "askdiandian" in profile.base_url.lower()


def is_lmstudio_qwen38_profile(profile: ModelProfile) -> bool:
    """Qwen3.8 sampling and thinking controls served through LM Studio."""

    provider = profile.provider.strip().lower().replace("_", "-")
    model = profile.model.strip().lower().replace("_", "-")
    return provider == "lm-studio" and "qwen3.8" in model


SENSENOVA_HOSTS = frozenset({"token.sensenova.cn"})


def is_sensenova_profile(profile: ModelProfile) -> bool:
    return endpoint_host(profile) in SENSENOVA_HOSTS


def is_minimax_m3_profile(profile: ModelProfile) -> bool:
    """MiniMax M3 uses ``thinking`` plus ``reasoning_split`` controls."""

    return profile.model.strip().lower() in {"minimax-m3", "minimaxai/minimax-m3"}


def is_glm_thinking_profile(profile: ModelProfile) -> bool:
    """GLM reasoning models that accept the ``thinking.type`` control."""

    model = profile.model.strip().lower().split("/")[-1]
    return model.startswith(("glm-4.5", "glm-4.6", "glm-4.7", "glm-5"))


def supports_reasoning_effort(profile: ModelProfile) -> bool:
    identity = " ".join([profile.name, profile.provider, profile.model]).lower()
    return (
        is_deepseek_profile(profile)
        or is_lmstudio_qwen38_profile(profile)
        or is_dots_profile(profile)
        or is_sensenova_profile(profile)
        or is_minimax_m3_profile(profile)
        or is_glm_thinking_profile(profile)
        or any(marker in identity for marker in ("gpt-5", "o3", "o4"))
    )


def apply_reasoning_controls(
    payload: dict[str, Any],
    *,
    profile: ModelProfile,
    reasoning_effort: str | None,
) -> dict[str, Any]:
    """Map the UI's four generic levels to provider-specific API controls."""
    # MiniMax M3 supports thinking independently of the generic
    # ``reasoning_effort`` field. Always request split reasoning so the model's
    # internal trace is returned in ``reasoning_details`` rather than mixed
    # into the visible answer (unless the caller explicitly asks for light).
    if is_minimax_m3_profile(profile):
        effort = normalize_reasoning_effort(reasoning_effort or "medium")
        payload["reasoning_split"] = True
        payload["thinking"] = {"type": "disabled" if effort == "light" else "adaptive"}
        return payload
    if is_lmstudio_qwen38_profile(profile):
        prepared_messages = payload.get("messages")
        preserve_thinking = qwen_history_has_reasoning(
            prepared_messages if isinstance(prepared_messages, list) else []
        )
        payload["chat_template_kwargs"] = {
            "enable_thinking": True,
            # Qwen3.8 requires earlier assistant reasoning to be replayed when
            # preservation is enabled.  Old Work Agent sessions retained only
            # the visible answer, so keep those conversations usable instead
            # of asking LM Studio to preserve history that is no longer there.
            "preserve_thinking": preserve_thinking,
        }
        if reasoning_effort is None:
            return payload
        effort = normalize_reasoning_effort(reasoning_effort)
        payload.update(
            {
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 20,
                "min_p": 0.0,
                "presence_penalty": 0.0,
                # LM Studio names Qwen's repetition_penalty field repeat_penalty.
                "repeat_penalty": 1.0,
            }
        )
        payload["reasoning_effort"] = {
            "light": "low",
            "medium": "low",
            "high": "medium",
            "very_high": "xhigh",
        }[effort]
        return payload
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
    if is_sensenova_profile(profile):
        # 商汤走朴素的 reasoning_effort，四档直接对上，关闭思考用 none。
        payload["reasoning_effort"] = {
            "light": "none",
            "medium": "medium",
            "high": "high",
            "very_high": "high",
        }[effort]
        return payload
    if is_dots_profile(profile):
        # Dots 只有开/关两档（该模型固定 max 思考档），没有 reasoning_effort。
        payload["chat_template_kwargs"] = {"enable_thinking": effort != "light"}
        return payload
    if is_glm_thinking_profile(profile):
        # GLM-5 is thinking-only on current providers (including routes whose
        # configured model name is still ``glm-5.2`` but whose upstream has
        # advanced to GLM-5.3).  Sending ``disabled`` is rejected with HTTP
        # 400, so its compaction safety must come from the larger output budget
        # rather than pretending that the model supports no-thinking mode.
        model = profile.model.strip().lower().split("/")[-1]
        payload["thinking"] = {
            "type": "enabled" if model.startswith("glm-5") or effort != "light" else "disabled"
        }
        return payload
    payload["reasoning_effort"] = {
        "light": "low",
        "medium": "medium",
        "high": "high",
        "very_high": "max",
    }[effort]
    return payload


def build_chat_tools_payload(
    messages: list[Message],
    *,
    profile: ModelProfile,
    temperature: float | None = None,
    max_tokens: int | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    reasoning_effort: str | None = None,
    request_usage: bool = True,
) -> dict[str, Any]:
    """Build the exact JSON body used by ``chat_tools_stream``.

    Diagnostics and production share this function so a context-analysis
    script cannot drift away from what the frontend really sends. Authentication
    headers are deliberately not part of the returned body.
    """

    payload: dict[str, Any] = {
        "model": profile.model,
        "messages": prepare_messages_for_profile(messages, profile),
        "temperature": profile.temperature if temperature is None else temperature,
        "max_tokens": profile.max_tokens if max_tokens is None else max_tokens,
        "stream": True,
    }
    if request_usage:
        payload["stream_options"] = {"include_usage": True}
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice or "auto"
    apply_reasoning_controls(payload, profile=profile, reasoning_effort=reasoning_effort)
    return payload


TRANSPORT_RETRIES = 3
"""How many extra attempts a transport failure gets. Not counting the first."""

TRANSPORT_RETRY_BACKOFF_SECONDS = (0.5, 1.5, 3.0)
RATE_LIMIT_BACKOFF_SECONDS = (5.0, 15.0, 30.0)
"""限流要等的是配额窗口，不是网络抖动。半秒后重试只会再撞一次。"""


RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
"""服务端明说"稍后再试"的状态码。

429 的报文里写的就是 Please try again later——把它当成永久失败，等于把一次
可恢复的限流变成一轮彻底失败。实测一次会话 6 轮全废，其中 2 轮就是 429。
"""


def retryable_status(error: BaseException | None) -> int:
    """错误链里第一个可重试的 HTTP 状态码，没有则返回 0。"""

    seen: set[int] = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        code = getattr(error, "code", None)
        if isinstance(error, urllib.error.HTTPError) and code in RETRYABLE_STATUS_CODES:
            return int(code)
        text = str(error)
        for status in RETRYABLE_STATUS_CODES:
            if f"HTTP {status}" in text:
                return status
        error = error.__cause__ or error.__context__
    return 0


def is_transport_failure(error: BaseException | None) -> bool:
    """这次失败值不值得原样再试一次？

    分两类：一类是根本没连上（DNS、TCP、超时），另一类是服务端答了但说"稍后
    再试"（429/5xx）。两者都该退避后重试同一个端点——而"服务端答了但拒绝了你"
    （401、400）不该重试，再试一百次也一样。
    """

    seen: set[int] = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, (socket.gaierror, ConnectionError, TimeoutError)):
            return True
        if isinstance(error, urllib.error.HTTPError):
            return int(getattr(error, "code", 0)) in RETRYABLE_STATUS_CODES
        if isinstance(error, urllib.error.URLError):
            return True
        error = error.__cause__ or error.__context__
    return bool(retryable_status(error))


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

        Endpoints selected by ``should_prefer_direct_connection`` always use
        the no-proxy opener. They never try the process proxy first and never
        fall back to it after a direct failure. This also avoids reusing a
        Request that urllib has mutated with proxy/tunnel state.
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
            headers=profile.auth_headers(),
            method="POST",
        )
        return self._send_with_retry(request, profile=profile)

    def _send_with_retry(self, request: Any, *, profile: ModelProfile) -> LLMResponse:
        """非流式请求也要退避重试。

        重试原来只做在流式恢复路径上，而摘要生成、审批审查、标题生成这些都走
        非流式——它们撞上 429 时直接失败，等于把一次可恢复的限流变成一次彻底
        失败。同一套判据：连不上或服务端说"稍后再试"就重试，说"不行"就不重试。
        """

        last_error: Exception | None = None
        for attempt in range(TRANSPORT_RETRIES + 1):
            if attempt:
                schedule = (
                    RATE_LIMIT_BACKOFF_SECONDS
                    if retryable_status(last_error) in {429, 503}
                    else TRANSPORT_RETRY_BACKOFF_SECONDS
                )
                time.sleep(schedule[min(attempt - 1, len(schedule) - 1)])
            try:
                return self._send_once(request, profile=profile)
            except Exception as error:
                last_error = error
                if not is_transport_failure(error):
                    raise
        raise last_error if last_error else RuntimeError("LLM request failed")

    def _send_once(self, request: Any, *, profile: ModelProfile) -> LLMResponse:
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
        on_heartbeat: Callable[[], None] | None = None,
        _allow_recovery: bool = True,
        _request_usage: bool = True,
    ) -> LLMResponse:
        payload = build_chat_tools_payload(
            messages,
            profile=profile,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            reasoning_effort=reasoning_effort,
            request_usage=_request_usage,
        )

        endpoint = chat_completions_endpoint(profile.base_url)
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=data,
            headers={**profile.auth_headers(), "Accept": "text/event-stream"},
            method="POST",
        )
        started_at = time.monotonic()
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: dict[int, dict[str, Any]] = {}
        finish_reason = None
        usage: dict[str, Any] = {}
        stream_error: BaseException | None = None
        # urllib applies this value while waiting for response headers / the
        # first body byte as well as between later reads. Before the first SSE
        # line, a local model is still doing prompt prefill, not an idle stream.
        # The ReAct watchdog enforces the shorter raw-SSE idle lease after data
        # starts arriving, so the transport must retain the full start budget.
        request_timeout_seconds = stream_start_timeout_seconds(profile)
        response_finished = threading.Event()
        try:
            with self._open_request(
                request,
                profile=profile,
                timeout=request_timeout_seconds,
            ) as response:
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
                    if on_heartbeat is not None:
                        on_heartbeat()
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
                    action="stream request",
                    profile=profile,
                    endpoint=endpoint,
                    timeout_seconds=request_timeout_seconds,
                )
            )
        except OSError as error:
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("模型流请求已取消。") from error
            if "timed out" in str(error).lower():
                stream_error = RuntimeError(
                    llm_timeout_message(
                        action="stream request",
                        profile=profile,
                        endpoint=endpoint,
                        timeout_seconds=request_timeout_seconds,
                    )
                )
            elif is_transport_failure(error):
                # ``http.client.RemoteDisconnected`` is both an OSError and a
                # ConnectionError.  Let it enter the same-endpoint recovery
                # path instead of escaping before the retry policy can see it.
                stream_error = error
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
                on_heartbeat=on_heartbeat,
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
        if reasoning_parts:
            message["reasoning_content"] = "".join(reasoning_parts)
        if valid_tool_calls:
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
                on_heartbeat=on_heartbeat,
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
        on_heartbeat: Callable[[], None] | None = None,
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
        recovery_timeout = recovery_request_timeout_seconds(profile)
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

        # Never switch routes: a different endpoint is a different model, and
        # substituting one silently would change the work the user asked for.
        # A DNS or TCP blip clears in a second or two, so the same endpoint is
        # simply asked again, with a short backoff, a bounded number of times.
        # 重试预算属于**这个恢复请求**，不属于主流为什么结束。原来写的是
        # `TRANSPORT_RETRIES if is_transport_failure(cause) else 0`——主流"返回了
        # 但没正文"时 cause 为空，恢复请求就只剩一发子弹，正好撞上 429 就是一轮
        # 彻底失败。而 429 恰恰是最该退避重试的那类。
        retries = TRANSPORT_RETRIES

        recovered = None
        recovery_error: Exception | None = None
        for attempt in range(retries + 1):
            if attempt > 0:
                # 退避档位看**上一次恢复请求**的错，不看主流的：限流要等几十秒，
                # DNS 抖动等一秒就够，拿错了对象就等错了时间。
                last_status = retryable_status(recovery_error) or retryable_status(cause)
                schedule = (
                    RATE_LIMIT_BACKOFF_SECONDS
                    if last_status in {429, 503}
                    else TRANSPORT_RETRY_BACKOFF_SECONDS
                )
                delay = schedule[min(attempt - 1, len(schedule) - 1)]
                if cancel_event is not None:
                    if cancel_event.wait(delay):
                        break
                else:
                    time.sleep(delay)
                if on_delta:
                    on_delta(
                        LLMStreamChunk(
                            status="network_retry",
                            status_detail=f"第 {attempt} / {retries} 次重试，等待 {delay:g} 秒后再试。",
                        )
                    )
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
                    on_heartbeat=on_heartbeat,
                    _allow_recovery=False,
                    _request_usage=request_usage,
                )
            except Exception as error:
                recovery_error = error
                if cancel_event is not None and cancel_event.is_set():
                    break
                if not is_transport_failure(error):
                    break
                continue
            recovery_error = None
            break

        if recovery_error is not None or recovered is None:
            prefix = "模型流式响应中断" if cause else "模型流式响应没有正文或工具调用"
            if is_transport_failure(cause) or is_transport_failure(recovery_error):
                host = urllib.parse.urlparse(profile.base_url).hostname or profile.base_url
                raise RuntimeError(
                    f"{prefix}：连不上 {host}（DNS 或网络故障），"
                    f"已重试 {retries} 次仍未恢复。"
                ) from (recovery_error or cause)
            status = retryable_status(recovery_error) or retryable_status(cause)
            if status in {429, 503}:
                # 限流和额度用尽都走 429，但对用户是两件事：一个等一会儿就好，
                # 一个要去后台加额度。报文里已经写明了是哪一种，把那句话拎出来，
                # 而不是把整段 JSON 糊上去。
                detail = str(recovery_error or cause)
                quota = "insufficient_quota" in detail or "quota exceeded" in detail
                raise RuntimeError(
                    f"{prefix}，{'该模型额度已用尽' if quota else '该模型正在限流'}"
                    f"（HTTP {status}），已退避重试 {retries} 次仍未通过。"
                    + ("请到服务商后台提额，或在设置里换一个模型。" if quota else "请稍后重试，或换一个模型。")
                ) from recovery_error
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
            headers={**profile.auth_headers(), "Accept": "text/event-stream"},
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
            "reasoning_details",
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
    """Prepare provider-specific assistant history without mutating storage."""
    needs_deepseek_repair = profile_requires_reasoning_content(profile)
    needs_qwen_reasoning_alias = is_lmstudio_qwen38_profile(profile)
    if not needs_deepseek_repair and not needs_qwen_reasoning_alias:
        return messages

    prepared: list[Message] = []
    for message in messages:
        if not isinstance(message, dict):
            prepared.append(message)
            continue
        clean = dict(message)
        if (
            needs_deepseek_repair
            and
            clean.get("role") == "assistant"
            and clean.get("tool_calls")
            and not str(clean.get("reasoning_content") or "").strip()
        ):
            clean["reasoning_content"] = (
                "Tool-call reasoning was not retained by an earlier client version."
            )
        if needs_qwen_reasoning_alias and clean.get("role") == "assistant":
            reasoning = str(
                clean.get("reasoning_content") or clean.get("reasoning") or ""
            )
            if reasoning:
                # Qwen's official multi-turn example sends both aliases.  Keep
                # one canonical field on disk and add the compatibility alias
                # only to the provider request.
                clean["reasoning_content"] = reasoning
                clean["reasoning"] = reasoning
        prepared.append(clean)
    return prepared


def qwen_history_has_reasoning(messages: list[Message]) -> bool:
    """Whether every historical assistant answer can preserve its thinking."""
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        if not str(message.get("reasoning_content") or message.get("reasoning") or "").strip():
            return False
    return True


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
