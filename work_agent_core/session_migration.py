"""Backfill existing conversation history into the append-only event log.

The old store keeps one flat message array per conversation with no turn or
step structure, so a faithful conversion has to reconstruct that structure
rather than invent detail it never recorded. Every event produced here carries
``seed: true``: the history is real, its boundaries are inferred.

A turn boundary is placed at each user message, which is what a turn actually
means — one user submission and everything the agent did to answer it. Tool
calls are recovered from the ``tool_calls`` an assistant message already
carries, so call/result pairing holds and the migrated log passes the same
invariants as a natively recorded one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence
import json

from .session_log import (
    ASSISTANT_MESSAGE,
    SESSION_CREATED,
    STEP_END,
    STEP_START,
    SessionHeader,
    SessionLog,
    TOOL_CALL,
    TOOL_RESULT,
    TURN_END,
    TURN_END_COMPLETED,
    TURN_START,
    USER_MESSAGE,
    check_session_invariants,
)
from .session_log_store import SessionLogStore
from .session_store import SessionStore, sanitize_runtime_message


SEED_TURN_PREFIX = "seed-turn"


def build_seed_log(
    conversation_id: str,
    messages: Sequence[dict[str, Any]],
    *,
    project_id: str = "",
    cwd: str = "",
) -> SessionLog:
    """Convert a flat message array into a structurally valid event log."""

    header = SessionHeader(session_id=conversation_id, project_id=project_id, cwd=cwd)
    log = SessionLog(header)
    log.append(
        SESSION_CREATED,
        {"session_id": conversation_id, "project_id": project_id, "cwd": cwd, "seed": True},
    )

    turn_index = 0
    turn_open = False

    def close_turn() -> None:
        nonlocal turn_open
        if not turn_open:
            return
        log.append(STEP_END, {"step": 1, "seed": True})
        log.append(
            TURN_END,
            {
                "turn_id": f"{SEED_TURN_PREFIX}-{turn_index}",
                "reason": {"kind": TURN_END_COMPLETED},
                "seed": True,
            },
        )
        turn_open = False

    def open_turn() -> None:
        nonlocal turn_open, turn_index
        turn_index += 1
        log.append(TURN_START, {"turn_id": f"{SEED_TURN_PREFIX}-{turn_index}", "seed": True})
        log.append(STEP_START, {"step": 1, "seed": True})
        turn_open = True

    for raw_message in messages:
        message = sanitize_runtime_message(raw_message)
        if not message:
            continue
        role = message.get("role")
        if role == "user":
            close_turn()
            open_turn()
            log.append(USER_MESSAGE, {"content": message.get("content") or "", "seed": True})
            continue
        if not turn_open:
            # History that starts mid-conversation still needs an enclosure.
            open_turn()
        if role == "assistant":
            tool_calls = message.get("tool_calls") or []
            log.append(
                ASSISTANT_MESSAGE,
                {
                    "content": message.get("content") or "",
                    "tool_calls": tool_calls,
                    "reasoning_content": message.get("reasoning_content") or "",
                    "seed": True,
                },
            )
            for call in tool_calls:
                function = call.get("function") if isinstance(call.get("function"), dict) else {}
                log.append(
                    TOOL_CALL,
                    {
                        "call_id": str(call.get("id") or ""),
                        "name": str(function.get("name") or ""),
                        "arguments": function.get("arguments") or "{}",
                        "seed": True,
                    },
                )
        elif role == "tool":
            log.append(
                TOOL_RESULT,
                {
                    "call_id": str(message.get("tool_call_id") or ""),
                    "name": str(message.get("name") or ""),
                    "content": message.get("content") or "",
                    "seed": True,
                },
            )
    close_turn()
    header.seed_length = log.seq
    return log


def migrate_conversation(
    session_store: SessionStore,
    log_store: SessionLogStore,
    conversation_id: str,
    *,
    project_id: str = "",
) -> dict[str, Any]:
    """Convert one conversation and report what happened to it."""

    session = session_store.load(conversation_id)
    log = build_seed_log(
        conversation_id,
        session.messages,
        project_id=project_id or str(session.metadata.get("project_id") or ""),
    )
    problems = check_session_invariants(log)
    existing = log_store.next_seq(conversation_id)
    if existing:
        return {
            "conversation_id": conversation_id,
            "status": "skipped",
            "reason": f"日志已存在 {existing} 个事件",
            "source_messages": len(session.messages),
            "problems": problems,
        }
    log_store.put_header(log.header)
    log_store.append(conversation_id, log.events)
    derived = log.derive_messages()
    return {
        "conversation_id": conversation_id,
        "status": "migrated",
        "source_messages": len(session.messages),
        "events": log.seq,
        "derived_messages": len(derived),
        "problems": problems,
    }


def migrate_all(
    session_store: SessionStore,
    log_store: SessionLogStore,
    *,
    conversation_ids: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    ids = list(conversation_ids) if conversation_ids is not None else discover_conversations(session_store)
    return [migrate_conversation(session_store, log_store, item) for item in sorted(ids)]


def discover_conversations(session_store: SessionStore) -> list[str]:
    directory = Path(session_store.session_dir)
    if not directory.is_dir():
        return []
    return [path.stem for path in directory.glob("*.json") if path.is_file()]


def compare_derived_messages(
    session_store: SessionStore,
    log_store: SessionLogStore,
    conversation_id: str,
) -> list[str]:
    """Report where the log-derived message list differs from stored history.

    A migration that silently changes what the model would receive is worse
    than one that fails loudly, so this compares role-by-role and reports the
    first structural difference rather than a boolean.
    """

    session = session_store.load(conversation_id)
    expected = [message for message in (sanitize_runtime_message(item) for item in session.messages) if message]
    actual = log_store.load(conversation_id).derive_messages()
    differences: list[str] = []
    if len(expected) != len(actual):
        differences.append(f"消息条数不同：原 {len(expected)}，派生 {len(actual)}")
    for index, (left, right) in enumerate(zip(expected, actual)):
        if left.get("role") != right.get("role"):
            differences.append(f"第 {index} 条 role 不同：{left.get('role')} vs {right.get('role')}")
            continue
        if str(left.get("content") or "") != str(right.get("content") or ""):
            differences.append(f"第 {index} 条 content 不同（role={left.get('role')}）")
        if json.dumps(left.get("tool_calls") or [], sort_keys=True, ensure_ascii=False) != json.dumps(
            right.get("tool_calls") or [], sort_keys=True, ensure_ascii=False
        ):
            differences.append(f"第 {index} 条 tool_calls 不同")
        if str(left.get("tool_call_id") or "") != str(right.get("tool_call_id") or ""):
            differences.append(f"第 {index} 条 tool_call_id 不同")
    return differences
