from __future__ import annotations

import tempfile
import unittest

from work_agent_core.external_event_inbox import ExternalEventInboxAdapter
from work_agent_core.session_log import SessionHeader, SessionLog
from work_agent_core.session_runtime import ConversationRuntime
from work_agent_core.turn_runtime import TurnRuntime
from work_agent_core.turn_store import TurnStore


class ExternalEventInboxAdapterTests(unittest.TestCase):
    def test_followup_is_logged_as_structured_event_before_ack(self) -> None:
        store = TurnStore(tempfile.mkdtemp())
        turn = TurnRuntime.start(store, conversation_id="chat-1")
        turn.enqueue_message("补充：使用领导汇报口径")
        conversation = ConversationRuntime(SessionLog(SessionHeader(session_id="chat-1")))
        conversation.begin_turn(turn.turn_id)
        adapter = ExternalEventInboxAdapter(turn, conversation)

        texts = adapter.take_messages()
        for text in texts:
            conversation.record_user(text, source="queued_followup")

        self.assertEqual(texts, ["补充：使用领导汇报口径"])
        self.assertEqual(len(conversation.log.pending_external_events()), 0)
        self.assertEqual(len(store.load(turn.turn_id).queued_messages), 1)

        conversation.flush()
        adapter.acknowledge(texts)
        self.assertEqual(store.load(turn.turn_id).queued_messages, [])


if __name__ == "__main__":
    unittest.main()
