from __future__ import annotations

from hashlib import sha256
from typing import Any, Iterable
import json
import re
import time


RECALL_ARCHIVE_VERSION = 2
SMALL_TURN_UNITS = 150
MIN_MULTI_TURN_UNITS = 220
MAX_MULTI_TURN_UNITS = 520
MAX_SMALL_TURNS_PER_EPISODE = 4
MAX_TOOL_RESULT_CHARS = 520
MAX_ACTION_CHARS = 280

PATH_KEYS = (
    "path",
    "output_path",
    "markdown_path",
    "input_path",
    "file_path",
    "url",
)
COUNT_KEYS = (
    "count",
    "copied_count",
    "created_count",
    "updated_count",
    "matched_count",
)
REFERENCE_CONTAINER_KEYS = (
    "files",
    "outputs",
    "artifacts",
    "generated_files",
    "created_files",
)


def build_recall_episodes(
    messages: list[dict[str, Any]],
    *,
    start_message_index: int = 0,
    existing: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Project completed ReAct turns into dense, human-recallable episodes.

    This is deliberately deterministic. It preserves the original conversational
    order: user text, public assistant implementation-path notes, compact tool
    actions/results, and the final answer remain interleaved exactly where they
    occurred. Only reproducible tool payload bulk is folded.
    """

    turns = extract_completed_turns(messages, start_message_index=start_message_index)
    episodes = group_turns_into_episodes(turns)
    combined = [normalize_episode(item) for item in (existing or []) if isinstance(item, dict)]
    for episode in episodes:
        if combined and should_merge_boundary(combined[-1], episode):
            combined[-1] = merge_episodes(combined[-1], episode)
        else:
            combined.append(episode)
    return combined


def extract_completed_turns(
    messages: list[dict[str, Any]],
    *,
    start_message_index: int = 0,
) -> list[dict[str, Any]]:
    user_indexes = [
        index for index, message in enumerate(messages) if message.get("role") == "user"
    ]
    turns: list[dict[str, Any]] = []
    pending_user_texts: list[str] = []
    pending_start: int | None = None
    for ordinal, start in enumerate(user_indexes):
        end = user_indexes[ordinal + 1] if ordinal + 1 < len(user_indexes) else len(messages)
        turn_messages = messages[start:end]
        final = find_final_answer(turn_messages)
        if final is None:
            # Imported/legacy sessions can contain a superseded user message
            # immediately followed by another user message. Carry the earlier
            # instruction into the next completed turn instead of indexing it
            # as an unrelated orphan.
            pending_user_texts.append(str(turn_messages[0].get("content") or "").strip())
            if pending_start is None:
                pending_start = start
            continue
        path_notes: list[str] = []
        actions: list[str] = []
        evidence: list[str] = []
        events: list[dict[str, str]] = [
            {"role": "user", "text": text}
            for text in pending_user_texts
            if text
        ]
        for message in turn_messages:
            role = str(message.get("role") or "")
            if role == "user":
                visible = str(message.get("content") or "").strip()
                if visible:
                    events.append({"role": "user", "text": visible})
            elif role == "assistant":
                visible = str(message.get("content") or "").strip()
                if visible:
                    events.append({"role": "assistant", "text": visible})
                    if message.get("tool_calls"):
                        path_notes.append(visible)
                for call in message.get("tool_calls") or []:
                    action, refs = summarize_tool_call(call)
                    if action:
                        actions.append(action)
                        events.append({
                            "role": "tool_call",
                            "text": with_inline_refs(action, refs),
                        })
                    evidence.extend(refs)
            elif role == "tool":
                summary, refs = summarize_tool_observation(
                    str(message.get("name") or ""),
                    str(message.get("content") or ""),
                )
                if summary:
                    actions.append(summary)
                    events.append({
                        "role": "tool_result",
                        "text": with_inline_refs(summary, refs),
                    })
                evidence.extend(refs)
        current_user_text = str(turn_messages[0].get("content") or "").strip()
        user_texts = [
            text for text in [*pending_user_texts, current_user_text] if text
        ]
        effective_start = pending_start if pending_start is not None else start
        pending_user_texts = []
        pending_start = None
        final_text = str(final.get("content") or "").strip()
        units = approx_units("\n".join([*user_texts, *path_notes, *actions, final_text]))
        turns.append(
            {
                "start_message_index": start_message_index + effective_start,
                "end_message_index": start_message_index + end,
                "user_texts": user_texts,
                "path_texts": dedupe(path_notes),
                "actions": dedupe(actions),
                "evidence_refs": dedupe(evidence),
                "final_texts": [final_text] if final_text else [],
                "turn_count": 1,
                "units": units,
                "has_tools": bool(actions),
                "turns": [
                    {
                        "user": "\n\n".join(user_texts),
                        "path_texts": dedupe(path_notes),
                        "actions": dedupe(actions),
                        "evidence_refs": dedupe(evidence),
                        "final": final_text,
                        "events": events,
                    }
                ],
            }
        )
    return turns


def group_turns_into_episodes(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []

    def flush_pending(*, merge_down: dict[str, Any] | None = None) -> None:
        nonlocal pending
        if not pending:
            if merge_down is not None:
                groups.append(finalize_episode(merge_down))
            return
        grouped = combine_turns(pending)
        pending = []
        if merge_down is not None:
            grouped = merge_episodes(grouped, finalize_episode(merge_down))
        elif grouped["turn_count"] == 1 and groups:
            groups[-1] = merge_episodes(groups[-1], grouped)
            return
        groups.append(grouped)

    for turn in turns:
        substantive = bool(turn.get("has_tools")) or int(turn.get("units") or 0) >= SMALL_TURN_UNITS
        if substantive:
            flush_pending(merge_down=turn)
            continue
        pending.append(turn)
        pending_units = sum(int(item.get("units") or 0) for item in pending)
        if (
            len(pending) >= MAX_SMALL_TURNS_PER_EPISODE
            or pending_units >= MIN_MULTI_TURN_UNITS
        ):
            flush_pending()
    flush_pending()
    return groups


def compact_messages_for_archive(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Remove reproducible execution bulk after a compression checkpoint.

    User text, public assistant path notes and final answers remain verbatim.
    Tool calls stay protocol-valid but keep only route-defining arguments.
    Tool observations become short outcome/evidence records.
    """

    compacted: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role") or "")
        if role != "assistant" or not message.get("tool_calls"):
            if role == "tool":
                summary, refs = summarize_tool_observation(
                    str(message.get("name") or ""),
                    str(message.get("content") or ""),
                )
                content = summary or "工具已执行；详细运行回显已在上下文压缩时折叠。"
                if refs:
                    content += "\n关键引用：" + "；".join(refs)
                compacted.append({**message, "content": content})
            else:
                compacted.append(dict(message))
            continue

        clean = dict(message)
        clean.pop("reasoning_content", None)
        clean["tool_calls"] = [
            compact_tool_call(call) for call in message.get("tool_calls") or []
            if isinstance(call, dict)
        ]
        compacted.append(clean)
    return compacted


def compact_tool_call(call: dict[str, Any]) -> dict[str, Any]:
    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    name = str(function.get("name") or call.get("name") or "")
    arguments = parse_arguments(function.get("arguments", call.get("arguments", {})))
    compact_args: dict[str, Any] = {}
    for key, value in arguments.items():
        if key in PATH_KEYS or key in COUNT_KEYS or key in {
            "query",
            "project_id",
            "conversation_id",
            "skill",
            "action",
            "op",
            "name",
        }:
            compact_args[key] = clip_value(value, 320)
        elif key in {"command", "cmd", "script"}:
            compact_args[key] = first_meaningful_line(str(value or ""), MAX_ACTION_CHARS)
    return {
        "id": str(call.get("id") or ""),
        "type": str(call.get("type") or "function"),
        "function": {
            "name": name,
            "arguments": json.dumps(compact_args, ensure_ascii=False),
        },
    }


def summarize_tool_call(call: dict[str, Any]) -> tuple[str, list[str]]:
    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    name = str(function.get("name") or call.get("name") or "工具").strip()
    arguments = parse_arguments(function.get("arguments", call.get("arguments", {})))
    refs = extract_refs(arguments)
    if name == "shell_exec":
        command = arguments.get("command") or arguments.get("cmd") or arguments.get("script")
        detail = first_meaningful_line(str(command or ""), MAX_ACTION_CHARS)
        return (f"运行命令：{detail}" if detail else "运行终端命令", refs)
    action_parts: list[str] = []
    for key in ("action", "op", "query", "path", "output_path", "file_path", "url"):
        value = arguments.get(key)
        if value not in (None, "", [], {}):
            action_parts.append(f"{key}={clip_value(value, 160)}")
    detail = "；".join(action_parts[:3])
    return (f"调用 {name}" + (f"：{detail}" if detail else ""), refs)


def summarize_tool_observation(tool_name: str, raw_content: str) -> tuple[str, list[str]]:
    text = str(raw_content or "").strip()
    refs: list[str] = []
    if not text:
        return (f"{tool_name or '工具'} 已完成，未返回正文。", refs)
    try:
        payload = json.loads(text)
    except Exception:
        payload = None
    if isinstance(payload, dict):
        refs = extract_refs(payload)
        status = payload.get("status")
        ok = payload.get("ok")
        summary = payload.get("summary") or payload.get("message") or payload.get("detail")
        counts = [
            f"{key}={payload[key]}" for key in COUNT_KEYS
            if key in payload and isinstance(payload[key], (int, float, str))
        ]
        state = (
            "完成" if ok is True or status in {"ok", "success", "completed", "done"}
            else "失败" if ok is False or status in {"error", "failed"}
            else "返回"
        )
        parts = [f"{tool_name or '工具'} {state}"]
        if summary:
            parts.append(clip_value(summary, 320))
        if counts:
            parts.append("，".join(counts))
        return ("；".join(parts)[:MAX_TOOL_RESULT_CHARS], refs)
    if looks_like_error(text):
        return (f"{tool_name or '工具'} 失败：{compact_line(text, 420)}", refs)
    return (f"{tool_name or '工具'} 已完成。", refs)


def render_episode_text(episode: dict[str, Any]) -> str:
    turns = episode.get("turns")
    if isinstance(turns, list) and turns:
        rendered_turns: list[str] = []
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            events = normalize_events(turn.get("events"))
            if not events:
                events = legacy_turn_events(turn)
            rendered = render_turn_events(events)
            if rendered:
                rendered_turns.append(rendered)
        if rendered_turns:
            return "\n\n---\n\n".join(rendered_turns)

    return render_turn_events(legacy_turn_events({
        "user": "\n\n".join(clean_text_list(episode.get("user_texts"))),
        "path_texts": episode.get("path_texts"),
        "actions": episode.get("actions"),
        "evidence_refs": episode.get("evidence_refs"),
        "final": "\n\n".join(clean_text_list(episode.get("final_texts"))),
    }))


def render_turn_events(events: list[dict[str, str]]) -> str:
    """Render one turn as a single chronological conversation stream."""

    blocks: list[str] = []
    for event in events:
        role = event["role"]
        text = event["text"]
        if role == "user":
            blocks.append(f"用户：\n{text}")
        elif role == "assistant":
            blocks.append(f"智能体：\n{text}")
        elif role == "tool_call":
            blocks.append(f"  ↳ 工具调用：{text}")
        elif role == "tool_result":
            blocks.append(f"  ↳ 工具结果：{text}")
    return "\n\n".join(blocks).strip()


def legacy_turn_events(turn: dict[str, Any]) -> list[dict[str, str]]:
    """Render v1 archives without reviving their old split-section layout.

    Exact interleaving is recovered by rebuilding from session messages whenever
    those messages are available. This fallback only keeps old standalone data
    readable as one conversation stream.
    """

    events: list[dict[str, str]] = []
    user = str(turn.get("user") or "").strip()
    if user:
        events.append({"role": "user", "text": user})
    events.extend(
        {"role": "assistant", "text": text}
        for text in clean_text_list(turn.get("path_texts"))
    )
    actions = clean_text_list(turn.get("actions"))
    refs = clean_text_list(turn.get("evidence_refs"))
    for index, action in enumerate(actions):
        events.append({
            "role": "tool_result",
            "text": with_inline_refs(action, refs if index == len(actions) - 1 else []),
        })
    final = str(turn.get("final") or "").strip()
    if final:
        events.append({"role": "assistant", "text": final})
    return events


def should_merge_boundary(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_units = int(left.get("units") or approx_units(render_episode_text(left)))
    right_units = int(right.get("units") or approx_units(render_episode_text(right)))
    if bool(left.get("open")):
        return True
    if int(right.get("turn_count") or 1) == 1 and right_units < SMALL_TURN_UNITS:
        return left_units + right_units <= MAX_MULTI_TURN_UNITS
    return False


def combine_turns(turns: list[dict[str, Any]]) -> dict[str, Any]:
    episode = finalize_episode(turns[0])
    for turn in turns[1:]:
        episode = merge_episodes(episode, finalize_episode(turn))
    episode["open"] = (
        int(episode.get("turn_count") or 0) == 1
        and int(episode.get("units") or 0) < SMALL_TURN_UNITS
    )
    return episode


def finalize_episode(turn: dict[str, Any]) -> dict[str, Any]:
    episode = normalize_episode(turn)
    episode["id"] = episode_id(episode)
    episode["created_at"] = int(time.time())
    episode["archive_version"] = RECALL_ARCHIVE_VERSION
    episode["open"] = (
        int(episode.get("turn_count") or 1) == 1
        and int(episode.get("units") or 0) < SMALL_TURN_UNITS
    )
    return episode


def merge_episodes(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = {
        "start_message_index": min(
            int(left.get("start_message_index") or 0),
            int(right.get("start_message_index") or 0),
        ),
        "end_message_index": max(
            int(left.get("end_message_index") or 0),
            int(right.get("end_message_index") or 0),
        ),
        "user_texts": [*(left.get("user_texts") or []), *(right.get("user_texts") or [])],
        "path_texts": dedupe([*(left.get("path_texts") or []), *(right.get("path_texts") or [])]),
        "actions": dedupe([*(left.get("actions") or []), *(right.get("actions") or [])]),
        "evidence_refs": dedupe([
            *(left.get("evidence_refs") or []),
            *(right.get("evidence_refs") or []),
        ]),
        "final_texts": [*(left.get("final_texts") or []), *(right.get("final_texts") or [])],
        "turn_count": int(left.get("turn_count") or 0) + int(right.get("turn_count") or 0),
        "has_tools": bool(left.get("has_tools") or right.get("has_tools")),
        "turns": [
            *list(left.get("turns") or []),
            *list(right.get("turns") or []),
        ],
        "created_at": min(
            int(left.get("created_at") or time.time()),
            int(right.get("created_at") or time.time()),
        ),
        "archive_version": RECALL_ARCHIVE_VERSION,
    }
    merged["units"] = approx_units(render_episode_text(merged))
    merged["open"] = (
        merged["turn_count"] == 1 and merged["units"] < SMALL_TURN_UNITS
    )
    merged["id"] = episode_id(merged)
    return merged


def normalize_episode(value: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        "id": str(value.get("id") or ""),
        "start_message_index": max(0, int(value.get("start_message_index") or 0)),
        "end_message_index": max(0, int(value.get("end_message_index") or 0)),
        "user_texts": clean_text_list(value.get("user_texts")),
        "path_texts": clean_text_list(value.get("path_texts")),
        "actions": clean_text_list(value.get("actions")),
        "evidence_refs": clean_text_list(value.get("evidence_refs")),
        "final_texts": clean_text_list(value.get("final_texts")),
        "turn_count": max(1, int(value.get("turn_count") or 1)),
        "units": max(0, int(value.get("units") or 0)),
        "has_tools": bool(value.get("has_tools")),
        "open": bool(value.get("open")),
        "created_at": max(0, int(value.get("created_at") or 0)),
        "archive_version": RECALL_ARCHIVE_VERSION,
        "turns": [
            {
                "user": str(item.get("user") or "").strip(),
                "path_texts": clean_text_list(item.get("path_texts")),
                "actions": clean_text_list(item.get("actions")),
                "evidence_refs": clean_text_list(item.get("evidence_refs")),
                "final": str(item.get("final") or "").strip(),
                "events": normalize_events(item.get("events")),
            }
            for item in (value.get("turns") or [])
            if isinstance(item, dict)
        ],
    }
    if not normalized["units"]:
        normalized["units"] = approx_units(render_episode_text(normalized))
    if not normalized["id"]:
        normalized["id"] = episode_id(normalized)
    return normalized


def normalize_events(value: Any) -> list[dict[str, str]]:
    events: list[dict[str, str]] = []
    if not isinstance(value, list):
        return events
    allowed_roles = {"user", "assistant", "tool_call", "tool_result"}
    for item in value:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip()
        text = str(item.get("text") or "").strip()
        if role in allowed_roles and text:
            events.append({"role": role, "text": text})
    return events


def with_inline_refs(text: str, refs: Iterable[str]) -> str:
    """Keep evidence attached to the action/result that produced it."""

    base = str(text or "").strip()
    missing = [
        ref for ref in dedupe(refs)
        if ref and ref not in base
    ]
    if not missing:
        return base
    return f"{base}（引用：{'；'.join(missing)}）"


def find_final_answer(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    for message in reversed(messages):
        if (
            message.get("role") == "assistant"
            and not message.get("tool_calls")
            and str(message.get("content") or "").strip()
        ):
            return message
    return None


def episode_id(episode: dict[str, Any]) -> str:
    material = json.dumps(
        {
            "start": episode.get("start_message_index"),
            "end": episode.get("end_message_index"),
            "users": episode.get("user_texts"),
            "finals": episode.get("final_texts"),
            "turns": episode.get("turns"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return "episode-" + sha256(material.encode("utf-8")).hexdigest()[:16]


def parse_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(str(value or "{}"))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def extract_refs(value: Any) -> list[str]:
    refs: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in PATH_KEYS and isinstance(item, str) and item.strip():
                refs.append(f"{key}={item.strip()}")
            elif key in REFERENCE_CONTAINER_KEYS:
                refs.extend(extract_reference_container(item))
    elif isinstance(value, list):
        refs.extend(extract_reference_container(value))
    return dedupe(refs)


def extract_reference_container(value: Any) -> list[str]:
    refs: list[str] = []
    if isinstance(value, list):
        for item in value[:24]:
            if isinstance(item, str):
                refs.extend(extract_path_like_strings(item))
            elif isinstance(item, dict):
                refs.extend(extract_refs(item))
    elif isinstance(value, dict):
        refs.extend(extract_refs(value))
    elif isinstance(value, str):
        refs.extend(extract_path_like_strings(value))
    return dedupe(refs)


def extract_path_like_strings(text: str) -> list[str]:
    candidates = re.findall(
        r"(?:/[^\s\"'，。；]+|[A-Za-z0-9_.-]+\.(?:md|txt|json|csv|xlsx|docx|pdf|png|jpg))",
        str(text or ""),
        flags=re.IGNORECASE,
    )
    return dedupe(item.rstrip("):]}>") for item in candidates)[:12]


def approx_units(text: str) -> int:
    value = str(text or "")
    cjk = len(re.findall(r"[\u3400-\u9fff\uf900-\ufaff]", value))
    words = len(re.findall(r"[A-Za-z0-9_][A-Za-z0-9_.+-]*", value))
    return cjk + words


def first_meaningful_line(text: str, limit: int) -> str:
    for raw in str(text or "").splitlines():
        line = " ".join(raw.split()).strip()
        if line:
            return line[:limit] + ("…" if len(line) > limit else "")
    return ""


def compact_line(text: str, limit: int) -> str:
    value = " ".join(str(text or "").split())
    return value[:limit] + ("…" if len(value) > limit else "")


def clip_value(value: Any, limit: int) -> Any:
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [clip_value(item, max(40, limit // 2)) for item in value[:8]]
    if isinstance(value, dict):
        return {str(key): clip_value(item, max(40, limit // 2)) for key, item in list(value.items())[:8]}
    text = str(value)
    return text[:limit] + ("…" if len(text) > limit else "")


def looks_like_error(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(marker in lowered for marker in ("error", "failed", "traceback", "timed out", "失败", "异常"))


def clean_text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item or "").strip()]


def dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for raw in items:
        item = str(raw or "").strip()
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result
