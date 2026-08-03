from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json
import re
import threading
import time


SESSION_SCHEMA_VERSION = 2
DEFAULT_SESSION_DIR = Path("meet_files/conversation_history/sessions")


@dataclass
class ConversationSession:
    id: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    summary_message_count: int = 0
    recall_episodes: list[dict[str, Any]] = field(default_factory=list)
    compaction_events: list[dict[str, Any]] = field(default_factory=list)
    created_at: int = field(default_factory=lambda: int(time.time()))
    updated_at: int = field(default_factory=lambda: int(time.time()))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SESSION_SCHEMA_VERSION,
            "id": self.id,
            "created_at": self.created_at,
            "updated_at": int(time.time()),
            "summary": self.summary,
            "summary_message_count": self.summary_message_count,
            "recall_episodes": self.recall_episodes,
            "compaction_events": self.compaction_events,
            "messages": repair_runtime_message_sequence(
                [message for message in (sanitize_runtime_message(item) for item in self.messages) if message]
            ),
            "metadata": self.metadata,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any], *, fallback_id: str) -> ConversationSession:
        messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []
        cleaned_messages = [message for message in (sanitize_runtime_message(item) for item in messages) if message]
        return cls(
            id=sanitize_conversation_id(payload.get("id")) or fallback_id,
            messages=repair_runtime_message_sequence(cleaned_messages),
            summary=str(payload.get("summary") or ""),
            summary_message_count=max(0, int(payload.get("summary_message_count") or 0)),
            recall_episodes=[
                item for item in (payload.get("recall_episodes") or [])
                if isinstance(item, dict)
            ],
            compaction_events=[
                item for item in (payload.get("compaction_events") or [])
                if isinstance(item, dict)
            ],
            created_at=max(0, int(payload.get("created_at") or time.time())),
            updated_at=max(0, int(payload.get("updated_at") or time.time())),
            metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
        )


class SessionStore:
    """Persistent per-conversation working memory store.

    This is deliberately conversation-scoped runtime state, not long-term user
    memory. Active messages keep exact ReAct state. Once a compression checkpoint
    covers a completed turn, reproducible tool bulk may be folded while a dense
    recall episode preserves the public implementation path and outcome.
    """

    def __init__(self, workspace_root: str | Path, *, session_dir: str | Path = DEFAULT_SESSION_DIR) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        directory = Path(session_dir)
        if not directory.is_absolute():
            directory = self.workspace_root / directory
        self.session_dir = directory
        self._lock = threading.RLock()

    def load(self, conversation_id: str) -> ConversationSession:
        safe_id = sanitize_conversation_id(conversation_id)
        if not safe_id:
            raise ValueError("conversation_id is required")
        path = self._path_for(safe_id)
        with self._lock:
            if not path.is_file():
                return ConversationSession(id=safe_id)
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return ConversationSession(id=safe_id)
            return ConversationSession.from_payload(payload, fallback_id=safe_id)

    def save(self, session: ConversationSession) -> ConversationSession:
        session.id = sanitize_conversation_id(session.id)
        if not session.id:
            raise ValueError("conversation_id is required")
        session.messages = [
            message for message in (sanitize_runtime_message(item) for item in session.messages) if message
        ]
        session.messages = repair_runtime_message_sequence(session.messages)
        session.summary = str(session.summary or "").strip()
        session.summary_message_count = max(
            0,
            min(int(session.summary_message_count or 0), len(session.messages)),
        )
        session.recall_episodes = [
            item for item in session.recall_episodes if isinstance(item, dict)
        ]
        session.compaction_events = [
            item for item in session.compaction_events if isinstance(item, dict)
        ][-64:]
        session.updated_at = int(time.time())
        path = self._path_for(session.id)
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_suffix(".tmp")
            tmp_path.write_text(
                json.dumps(session.to_payload(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp_path.replace(path)
        return session

    def bootstrap_from_display_messages(
        self,
        session: ConversationSession,
        display_messages: list[dict[str, Any]],
        *,
        exclude_last_user: bool = True,
    ) -> bool:
        if session.messages:
            return False
        cleaned = [message for message in (display_message_to_runtime(item) for item in display_messages) if message]
        cleaned = drop_initial_greeting(cleaned)
        if exclude_last_user and cleaned and cleaned[-1].get("role") == "user":
            cleaned = cleaned[:-1]
        session.messages = cleaned
        return bool(cleaned)

    def has_user_message_ordinal(self, session: ConversationSession, user_message_ordinal: int) -> bool:
        ordinal = int(user_message_ordinal)
        if ordinal < 0:
            raise ValueError("rewind_user_message_ordinal must be non-negative")
        seen_users = 0
        for message in session.messages:
            if message.get("role") != "user":
                continue
            if seen_users == ordinal:
                return True
            seen_users += 1
        return False

    def rebuild_from_display_messages(
        self,
        session: ConversationSession,
        display_messages: list[dict[str, Any]],
        *,
        exclude_last_user: bool = True,
    ) -> bool:
        session.messages = []
        session.summary = ""
        session.summary_message_count = 0
        session.recall_episodes = []
        session.compaction_events = []
        return self.bootstrap_from_display_messages(
            session,
            display_messages,
            exclude_last_user=exclude_last_user,
        )

    def append_user_message(self, session: ConversationSession, content: str) -> None:
        text = str(content or "").strip()
        if not text:
            raise ValueError("user message content is required")
        session.messages.append({"role": "user", "content": text})

    def rewind_before_user_message(
        self,
        session: ConversationSession,
        user_message_ordinal: int,
    ) -> None:
        """Drop the selected user turn and everything after it.

        Display history does not contain hidden assistant tool calls/results, so
        rewinding by a raw message index would corrupt the runtime protocol.
        Counting user turns lets us keep the exact legal runtime prefix.
        """

        ordinal = int(user_message_ordinal)
        if ordinal < 0:
            raise ValueError("rewind_user_message_ordinal must be non-negative")
        seen_users = 0
        cut_index: int | None = None
        for index, message in enumerate(session.messages):
            if message.get("role") != "user":
                continue
            if seen_users == ordinal:
                cut_index = index
                break
            seen_users += 1
        if cut_index is None:
            raise ValueError("The selected user message is not present in the backend session.")
        session.messages = session.messages[:cut_index]
        # A summary may cover turns at or after the cut. Rebuilding it lazily is
        # safer than leaking discarded context into the new branch.
        session.summary = ""
        session.summary_message_count = 0
        session.recall_episodes = [
            episode
            for episode in session.recall_episodes
            if int(episode.get("end_message_index") or 0) <= cut_index
        ]
        session.compaction_events = []

    def _path_for(self, conversation_id: str) -> Path:
        return self.session_dir / f"{conversation_id}.json"


def sanitize_conversation_id(raw_value: Any) -> str:
    text = str(raw_value or "").strip()
    if not text:
        return ""
    text = re.sub(r"[^A-Za-z0-9_.:-]+", "-", text).strip(".-")
    return text[:120]


def sanitize_runtime_message(raw_message: Any) -> dict[str, Any]:
    if not isinstance(raw_message, dict):
        return {}
    role = str(raw_message.get("role") or "").strip()
    if role == "user":
        content = str(raw_message.get("content") or "").strip()
        return {"role": "user", "content": content} if content else {}
    if role == "assistant":
        content = str(raw_message.get("content") or "")
        clean: dict[str, Any] = {"role": "assistant", "content": content}
        tool_calls = sanitize_tool_calls(raw_message.get("tool_calls"))
        if tool_calls:
            clean["tool_calls"] = tool_calls
            reasoning_content = str(raw_message.get("reasoning_content") or "")
            if reasoning_content:
                clean["reasoning_content"] = reasoning_content
        if content or tool_calls:
            return clean
        return {}
    if role == "tool":
        content = str(raw_message.get("content") or "")
        tool_call_id = str(raw_message.get("tool_call_id") or "").strip()
        name = str(raw_message.get("name") or "").strip()
        if not tool_call_id:
            return {}
        clean = {"role": "tool", "tool_call_id": tool_call_id, "content": content}
        if name:
            clean["name"] = name
        return clean
    return {}


def sanitize_tool_calls(raw_tool_calls: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_tool_calls, list):
        return []
    cleaned: list[dict[str, Any]] = []
    for index, raw_call in enumerate(raw_tool_calls):
        if not isinstance(raw_call, dict):
            continue
        function = raw_call.get("function") if isinstance(raw_call.get("function"), dict) else {}
        name = str(function.get("name") or raw_call.get("name") or "").strip()
        if not name:
            continue
        arguments = function.get("arguments", raw_call.get("arguments", "{}"))
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments if arguments is not None else {}, ensure_ascii=False)
        cleaned.append(
            {
                "id": str(raw_call.get("id") or f"call_{index}"),
                "type": str(raw_call.get("type") or "function"),
                "function": {
                    "name": name,
                    "arguments": arguments,
                },
            }
        )
    return cleaned


def repair_runtime_message_sequence(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Make stored OpenAI messages legal after interrupted/paused tool calls.

    OpenAI-compatible APIs require an assistant message with tool_calls to be
    immediately followed by one tool message for every tool_call_id. Older
    approval-paused turns could leave only the first tool result written. This
    repair keeps the history usable by inserting explicit TOOL_MISSING results
    for missing calls and converting orphan tool messages into assistant notes
    so the content is preserved without violating the OpenAI message schema.
    """

    repaired: list[dict[str, Any]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        role = message.get("role")
        if role == "tool":
            # A tool message is only valid as part of the immediate block after
            # an assistant tool_calls message. If we see it here it is orphaned;
            # preserve the content as a normal assistant history note.
            repaired.append(orphan_tool_message_to_assistant(message))
            index += 1
            continue

        repaired.append(message)
        if role != "assistant":
            index += 1
            continue

        tool_calls = message.get("tool_calls") if isinstance(message.get("tool_calls"), list) else []
        expected = [tool_call_id(call, fallback_index=i) for i, call in enumerate(tool_calls)]
        expected = [item for item in expected if item]
        if not expected:
            index += 1
            continue

        expected_set = set(expected)
        seen: set[str] = set()
        orphan_tool_notes: list[dict[str, str]] = []
        next_index = index + 1
        while next_index < len(messages) and messages[next_index].get("role") == "tool":
            tool_message = messages[next_index]
            result_call_id = str(tool_message.get("tool_call_id") or "").strip()
            if result_call_id and result_call_id in expected_set and result_call_id not in seen:
                seen.add(result_call_id)
                repaired.append(tool_message)
            else:
                # A mismatched or duplicate tool message would still be illegal
                # here because it does not answer one of this assistant
                # message's pending tool_call IDs. Preserve it as normal text
                # after the legal tool-result block.
                orphan_tool_notes.append(orphan_tool_message_to_assistant(tool_message))
            next_index += 1

        for call_index, call in enumerate(tool_calls):
            call_id = tool_call_id(call, fallback_index=call_index)
            if not call_id or call_id in seen:
                continue
            repaired.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": tool_call_name(call),
                    "content": (
                        "TOOL_MISSING: 此工具调用在上一轮被审批暂停或中断，"
                        "运行时已补齐占位结果以恢复合法消息结构；该工具没有实际执行。"
                    ),
                }
            )
        repaired.extend(orphan_tool_notes)
        index = next_index
    return repaired


def tool_call_id(call: Any, *, fallback_index: int) -> str:
    if not isinstance(call, dict):
        return f"call_{fallback_index}"
    return str(call.get("id") or f"call_{fallback_index}").strip()


def tool_call_name(call: Any) -> str:
    if not isinstance(call, dict):
        return ""
    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    return str(function.get("name") or call.get("name") or "").strip()


def orphan_tool_message_to_assistant(message: dict[str, Any]) -> dict[str, str]:
    name = str(message.get("name") or "").strip()
    call_id = str(message.get("tool_call_id") or "").strip()
    content = str(message.get("content") or "")
    label_parts = [part for part in (name, call_id) if part]
    label = " / ".join(label_parts) or "unknown"
    return {
        "role": "assistant",
        "content": (
            "历史工具结果（原消息结构中缺少对应 assistant tool_calls，"
            f"已转为普通上下文以避免非法 tool message；工具={label}）：\n"
            f"{content}"
        ),
    }


def display_message_to_runtime(raw_message: Any) -> dict[str, Any]:
    if not isinstance(raw_message, dict):
        return {}
    role = str(raw_message.get("role") or "").strip()
    if role not in {"user", "assistant"}:
        return {}
    content = str(raw_message.get("content") or "").strip()
    if not content:
        return {}
    return {"role": role, "content": content}


def drop_initial_greeting(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not messages:
        return messages
    first = messages[0]
    if first.get("role") != "assistant":
        return messages
    content = str(first.get("content") or "")
    greeting_fragments = (
        "你好，我是本地工作智能体",
        "新聊天已开始",
        "可以直接和我对话",
        "拖入录音或材料",
    )
    if any(fragment in content for fragment in greeting_fragments):
        return messages[1:]
    return messages
