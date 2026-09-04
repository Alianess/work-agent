"""Two-phase adapter from a turn inbox to the conversation event log."""

from __future__ import annotations

from .session_runtime import ConversationRuntime
from .turn_runtime import TurnRuntime


class ExternalEventInboxAdapter:
    """Claim follow-ups at a safe point and acknowledge after log flush."""

    def __init__(self, turn: TurnRuntime, conversation: ConversationRuntime) -> None:
        self.turn = turn
        self.conversation = conversation
        self._claimed_ids: list[str] = []

    def take_messages(self) -> list[str]:
        events = self.turn.peek_message_events()
        self._claimed_ids = [
            str(item.get("event_id") or "")
            for item in events
            if str(item.get("event_id") or "")
        ]
        texts: list[str] = []
        for item in events:
            payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
            content = str(payload.get("content") or "").strip()
            if not content:
                continue
            event_id = str(item.get("event_id") or "")
            self.conversation.queue_external_event(
                event_id,
                kind=str(item.get("kind") or "user_followup"),
                payload=payload,
                source=str(item.get("source") or "chat"),
            )
            self.conversation.consume_external_event(
                event_id,
                turn_id=self.turn.turn_id,
            )
            texts.append(content)
        return texts

    def acknowledge(self, _texts: list[str]) -> None:
        self.turn.ack_message_events(list(self._claimed_ids))
        self._claimed_ids.clear()
