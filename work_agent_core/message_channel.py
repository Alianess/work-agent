from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class ChannelAttachment:
    kind: str
    name: str = ""
    local_path: Path | None = None
    remote_url: str = ""
    mime_type: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChannelMessage:
    channel: str
    account_id: str
    conversation_id: str
    sender_id: str
    message_id: str
    timestamp_ms: int
    text: str
    context_token: str = ""
    reply_to_message_id: str = ""
    attachments: tuple[ChannelAttachment, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChannelReply:
    text: str
    attachments: tuple[ChannelAttachment, ...] = ()


class MessageChannelAdapter(Protocol):
    """Platform-neutral boundary used by the assistant runtime."""

    channel_id: str

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def status(self) -> dict[str, Any]: ...
