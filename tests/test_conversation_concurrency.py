from __future__ import annotations

import threading
import unittest
from unittest.mock import patch

from work_agent_core import web_server


class ConversationConcurrencyTests(unittest.TestCase):
    def setUp(self) -> None:
        with web_server.ACTIVE_CHAT_CONVERSATIONS_LOCK:
            web_server.ACTIVE_CHAT_CONVERSATIONS.clear()

    def test_same_conversation_is_rejected_while_different_conversations_can_run(self) -> None:
        first_started = threading.Event()
        release_first = threading.Event()
        results: list[dict] = []

        def fake_events(payload: dict):
            conversation_id = str(payload["conversation_id"])
            if conversation_id == "conversation-a":
                first_started.set()
                release_first.wait(timeout=2)
            yield {"event": "final", "content": conversation_id}

        first_payload = {
            "conversation_id": "conversation-a",
            "messages": [{"role": "user", "content": "first"}],
        }
        second_payload = {
            "conversation_id": "conversation-b",
            "messages": [{"role": "user", "content": "second"}],
        }

        with patch.object(web_server, "_run_agent_chat_events", side_effect=fake_events):
            first_thread = threading.Thread(
                target=lambda: results.extend(web_server.run_agent_chat_events(first_payload))
            )
            first_thread.start()
            self.assertTrue(first_started.wait(timeout=1))

            duplicate = list(web_server.run_agent_chat_events(first_payload))
            different = list(web_server.run_agent_chat_events(second_payload))
            release_first.set()
            first_thread.join(timeout=2)

        self.assertEqual(duplicate[0]["type"], "ConversationBusy")
        self.assertEqual(different[-1]["content"], "conversation-b")
        self.assertEqual(results[-1]["content"], "conversation-a")
        self.assertFalse(first_thread.is_alive())


if __name__ == "__main__":
    unittest.main()
