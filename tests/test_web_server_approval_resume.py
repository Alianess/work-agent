from __future__ import annotations

import time
import unittest
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import patch

from work_agent_core import web_server


class _FakeTurnStore:
    def __init__(self) -> None:
        self.pending = {
            "conversation_id": "conversation-1",
            "profile_name": "test-profile",
            "runtime_messages_before_batch": [],
            "context_summary": "summary-before-approval",
            "context_summary_message_count": 7,
            "context_compacted": True,
            "context_estimated_tokens": 1234,
        }
        self.cleared = False
        self.saved_pending: dict | None = None

    def pending_approval(self, _turn_id: str) -> dict:
        return self.pending

    def load(self, _turn_id: str) -> SimpleNamespace:
        return SimpleNamespace(conversation_id="conversation-1", profile="test-profile")

    def clear_pending_approval(self, _turn_id: str) -> None:
        self.cleared = True

    def set_pending_approval(self, _turn_id: str, pending: dict) -> None:
        self.saved_pending = pending


class _FakeSessionStore:
    def __init__(self) -> None:
        self.session = SimpleNamespace(
            messages=[],
            summary="",
            summary_message_count=0,
        )

    def load(self, _conversation_id: str) -> SimpleNamespace:
        return self.session

    def save(self, _session: SimpleNamespace) -> None:
        return None


class _FakeTurnRuntime:
    def __init__(self) -> None:
        self.turn_id = "turn-1"
        self.started_at = time.monotonic()

    @classmethod
    def resume(cls, _turn_store: _FakeTurnStore, _turn_id: str) -> "_FakeTurnRuntime":
        return cls()

    def initial_event(self) -> dict:
        return {"event": "turn"}

    def emit(self, event: dict) -> dict:
        return event

    def cancelled(self) -> bool:
        return False

    def raise_if_cancelled(self) -> None:
        return None


class _FakeAgent:
    def __init__(self, **_kwargs: object) -> None:
        return None

    def iter_approved_tool_batch_events(
        self,
        runtime_messages: list[dict],
        _pending_approval: dict,
        *,
        system_context: str = "",
    ):
        del system_context
        runtime_messages.append({"role": "assistant", "content": "done"})
        yield {
            "event": "final",
            "content": "done",
            "steps_used": 1,
            "used_tools": True,
        }


class _FakeWaitingAgent(_FakeAgent):
    def iter_approved_tool_batch_events(
        self,
        runtime_messages: list[dict],
        _pending_approval: dict,
        *,
        system_context: str = "",
    ):
        del runtime_messages, system_context
        yield {
            "event": "final",
            "content": "approval required",
            "steps_used": 2,
            "used_tools": True,
            "waiting_approval": True,
            "pending_approval": {"runtime_messages_before_batch": []},
        }


class _FakeMalformedHistoryAgent(_FakeAgent):
    def iter_approved_tool_batch_events(
        self,
        runtime_messages: list[dict],
        _pending_approval: dict,
        *,
        system_context: str = "",
    ):
        del system_context
        runtime_messages.append(None)
        yield {
            "event": "final",
            "content": '<tool_call name="shell_exec">{"command":"pwd"}</tool_call>',
            "steps_used": 1,
            "used_tools": True,
        }


class _FakeDebugTrace:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        return None

    def emit(self, *_args: object, **_kwargs: object) -> None:
        return None

    def context_payload(self) -> dict:
        return {}


class ApprovalResumeTests(unittest.TestCase):
    @contextmanager
    def _patched_runtime(self, turn_store, session_store, agent_class):
        profile = SimpleNamespace(name="test-profile", model="test-model")
        registry = SimpleNamespace(default_profile="test-profile", get=lambda _name: profile)
        patches = (
            patch.object(web_server, "get_turn_store", return_value=turn_store),
            patch.object(web_server, "get_session_store", return_value=session_store),
            patch.object(web_server, "load_registry", return_value=registry),
            patch.object(web_server, "OpenAICompatibleClient", return_value=object()),
            patch.object(web_server, "DebugTrace", _FakeDebugTrace),
            patch.object(web_server, "TurnRuntime", _FakeTurnRuntime),
            patch.object(web_server, "build_default_tools", return_value=object()),
            patch.object(web_server, "ReActAgent", agent_class),
            patch.object(web_server, "agent_system_context", return_value=""),
        )
        with ExitStack() as stack:
            for runtime_patch in patches:
                stack.enter_context(runtime_patch)
            yield

    def test_final_event_without_another_approval_keeps_resume_context(self) -> None:
        turn_store = _FakeTurnStore()
        session_store = _FakeSessionStore()

        with self._patched_runtime(turn_store, session_store, _FakeAgent):
            events = list(web_server.approve_turn_events("turn-1", {}))

        final = next(event for event in events if event.get("event") == "final")
        self.assertEqual(final["content"], "done")
        self.assertTrue(final["context_compacted"])
        self.assertEqual(final["context_estimated_tokens"], 1234)
        self.assertEqual(final["context_summary"], "summary-before-approval")
        self.assertEqual(final["context_summary_message_count"], 7)
        self.assertTrue(turn_store.cleared)

    def test_a_later_approval_inherits_the_original_context(self) -> None:
        turn_store = _FakeTurnStore()
        session_store = _FakeSessionStore()

        with self._patched_runtime(turn_store, session_store, _FakeWaitingAgent):
            events = list(web_server.approve_turn_events("turn-1", {}))

        final = next(event for event in events if event.get("event") == "final")
        self.assertTrue(final["waiting_approval"])
        self.assertIsNotNone(turn_store.saved_pending)
        self.assertTrue(turn_store.saved_pending["context_compacted"])
        self.assertEqual(turn_store.saved_pending["context_estimated_tokens"], 1234)
        self.assertEqual(turn_store.saved_pending["context_summary"], "summary-before-approval")
        self.assertFalse(turn_store.cleared)

    def test_malformed_runtime_history_does_not_break_final_tool_markup_guard(self) -> None:
        turn_store = _FakeTurnStore()
        session_store = _FakeSessionStore()

        with self._patched_runtime(turn_store, session_store, _FakeMalformedHistoryAgent):
            events = list(web_server.approve_turn_events("turn-1", {}))

        final = next(event for event in events if event.get("event") == "final")
        self.assertIn("后端已拦截", final["content"])
        self.assertEqual(session_store.session.messages, [])
        self.assertTrue(turn_store.cleared)


if __name__ == "__main__":
    unittest.main()
