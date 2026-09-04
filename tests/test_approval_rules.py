from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from work_agent_core import web_server
from work_agent_core.approval_rules import (
    find_remembered_approval,
    forget_command_approval,
    load_remembered_approvals,
    remember_command_approval,
    remembered_approvals_path,
)
from work_agent_core.runtime_profiles import reset_registry
from tests.execution_test_support import trusted_shell_tools_for_test


class ApprovalRulesStoreTests(unittest.TestCase):
    def test_remember_find_and_forget_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "remembered_approvals.json"
            rule = remember_command_approval(path, command="soffice --convert-to pdf a.docx", risk_category="MODIFY")

            self.assertIsNotNone(rule)
            found = find_remembered_approval(path, "soffice --convert-to pdf a.docx")
            self.assertIsNotNone(found)
            self.assertEqual(found["risk_category"], "MODIFY")

            self.assertTrue(forget_command_approval(path, "soffice --convert-to pdf a.docx"))
            self.assertIsNone(find_remembered_approval(path, "soffice --convert-to pdf a.docx"))
            self.assertFalse(forget_command_approval(path, "soffice --convert-to pdf a.docx"))

    def test_lookup_is_exact_command_only(self) -> None:
        """记住的是命令，不是程序名：批准过 a.docx 不等于批准 b.docx。"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "remembered_approvals.json"
            remember_command_approval(path, command="soffice --convert-to pdf a.docx", risk_category="MODIFY")

            self.assertIsNone(find_remembered_approval(path, "soffice --convert-to pdf b.docx"))
            self.assertIsNone(find_remembered_approval(path, "soffice --convert-to pdf a.docx --outdir out"))

    def test_delete_and_system_categories_are_never_remembered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "remembered_approvals.json"
            self.assertIsNone(
                remember_command_approval(path, command="rm a.txt", risk_category="DELETE")
            )
            self.assertIsNone(
                remember_command_approval(path, command="chmod +x run.sh", risk_category="SYSTEM")
            )
            self.assertEqual(load_remembered_approvals(path), [])

    def test_ree_members_same_command_only_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "remembered_approvals.json"
            remember_command_approval(path, command="npm run build", risk_category="EXECUTE")
            remember_command_approval(path, command="npm run build", risk_category="EXECUTE")

            items = load_remembered_approvals(path)
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0]["command"], "npm run build")

    def test_corrupt_file_reads_as_empty(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "remembered_approvals.json"
            path.write_text("not json", encoding="utf-8")
            self.assertEqual(load_remembered_approvals(path), [])
            self.assertIsNone(find_remembered_approval(path, "anything"))


class RememberedApprovalShellTests(unittest.TestCase):
    def _tools(self, workspace: str, rules_path: Path):
        tools = trusted_shell_tools_for_test(workspace)
        tools.approval_rules_path = rules_path
        return tools

    def test_remembered_command_runs_without_asking(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            rules = Path(workspace) / "rules.json"
            remember_command_approval(rules, command="mkdir generated", risk_category="MODIFY")
            tools = self._tools(workspace, rules)
            payload = json.loads(tools.execute({"command": "mkdir generated"}))

        self.assertEqual(payload["status"], "executed")
        self.assertIn("记住", payload.get("policy_reason") or "")

    def test_untremembered_variant_still_asks(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            rules = Path(workspace) / "rules.json"
            remember_command_approval(rules, command="mkdir generated", risk_category="MODIFY")
            tools = self._tools(workspace, rules)
            payload = json.loads(tools.execute({"command": "mkdir other"}))

        self.assertEqual(payload["status"], "approval_required")

    def test_delete_risk_still_asks_even_after_remember_attempt(self) -> None:
        """DELETE 在存储层就进不去名单，命令侧自然永远现场确认。"""
        with tempfile.TemporaryDirectory() as workspace:
            target = Path(workspace) / "generated.txt"
            target.write_text("temporary", encoding="utf-8")
            rules = Path(workspace) / "rules.json"
            remember_command_approval(rules, command="rm generated.txt", risk_category="DELETE")
            tools = self._tools(workspace, rules)
            payload = json.loads(tools.execute({"command": "rm generated.txt"}))

        self.assertEqual(payload["status"], "approval_required")
        self.assertEqual(payload["risk_category"], "DELETE")

    def test_no_rules_path_keeps_asking(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            tools = trusted_shell_tools_for_test(workspace)
            self.assertIsNone(tools.approval_rules_path)
            payload = json.loads(tools.execute({"command": "mkdir generated"}))

        self.assertEqual(payload["status"], "approval_required")


class _FakeTurnStore:
    def __init__(self, pending: dict) -> None:
        self.pending = pending

    def pending_approval(self, _turn_id: str) -> dict:
        return self.pending

    def load(self, _turn_id: str) -> SimpleNamespace:
        return SimpleNamespace(conversation_id="conversation-1", profile="test-profile")

    def clear_pending_approval(self, _turn_id: str) -> None:
        return None

    def set_pending_approval(self, _turn_id: str, _pending: dict) -> None:
        return None


class _FakeSessionStore:
    def __init__(self) -> None:
        self.session = SimpleNamespace(
            messages=[],
            summary="",
            summary_message_count=0,
            metadata={},
        )

    def load(self, _conversation_id: str) -> SimpleNamespace:
        return self.session

    def save(self, _session: SimpleNamespace) -> None:
        return None


class _FakeTurnRuntime:
    def __init__(self) -> None:
        self.turn_id = "turn-1"
        self.started_at = 0.0

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

    def drain_messages(self) -> list[str]:
        return []


class _FakeAgent:
    def __init__(self, **_kwargs: object) -> None:
        return None

    def iter_approved_tool_batch_events(self, runtime_messages, _pending: dict, *, system_context: str = ""):
        del system_context
        runtime_messages.append_message({"role": "assistant", "content": "done"})
        yield {"event": "final", "content": "done", "steps_used": 1, "used_tools": True}


class _FakeDebugTrace:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        return None

    def emit(self, *_args: object, **_kwargs: object) -> None:
        return None

    def context_payload(self) -> dict:
        return {}


class ApproveTurnRememberTests(unittest.TestCase):
    def setUp(self) -> None:
        # approve_turn_events 内部会解析 runtime profile；registry 是进程级
        # 单例，先复位再用后复原，避免把 "Friday" 名字泄漏进其他测试。
        reset_registry()

    def tearDown(self) -> None:
        reset_registry()

    @contextmanager
    def _patched_runtime(self, turn_store: _FakeTurnStore):
        profile = SimpleNamespace(name="test-profile", model="test-model")
        registry = SimpleNamespace(default_profile="test-profile", get=lambda _name: profile)
        patches = (
            patch.object(web_server, "get_turn_store", return_value=turn_store),
            patch.object(web_server, "get_session_store", return_value=_FakeSessionStore()),
            patch.object(web_server, "load_registry", return_value=registry),
            patch.object(web_server, "OpenAICompatibleClient", return_value=object()),
            patch.object(web_server, "DebugTrace", _FakeDebugTrace),
            patch.object(web_server, "TurnRuntime", _FakeTurnRuntime),
            patch.object(web_server, "build_default_tools", return_value=object()),
            patch.object(web_server, "ReActAgent", _FakeAgent),
            patch.object(web_server, "agent_system_context", return_value=""),
        )
        with ExitStack() as stack:
            for runtime_patch in patches:
                stack.enter_context(runtime_patch)
            yield

    def test_approve_with_remember_persists_rule_and_announces_it(self) -> None:
        pending = {
            "conversation_id": "conversation-1",
            "profile_name": "test-profile",
            "runtime_messages_before_batch": [],
            "approval_payload": {
                "command": "npm run build",
                "risk_category": "EXECUTE",
            },
        }
        turn_store = _FakeTurnStore(pending)
        with tempfile.TemporaryDirectory() as directory:
            with self._patched_runtime(turn_store):
                with patch.object(web_server, "user_data_dir", return_value=Path(directory)):
                    events = list(
                        web_server.approve_turn_events("turn-1", {"remember": True})
                    )
            rules_path = remembered_approvals_path(Path(directory))
            found = find_remembered_approval(rules_path, "npm run build")

        self.assertIsNotNone(found)
        self.assertEqual(found["risk_category"], "EXECUTE")
        activity = next(
            event
            for event in events
            if event.get("event") == "activity" and event.get("title") == "已记住这条命令"
        )
        self.assertEqual(activity["content"], "npm run build")

    def test_approve_without_remember_leaves_no_rule(self) -> None:
        pending = {
            "conversation_id": "conversation-1",
            "profile_name": "test-profile",
            "runtime_messages_before_batch": [],
            "approval_payload": {
                "command": "npm run build",
                "risk_category": "EXECUTE",
            },
        }
        turn_store = _FakeTurnStore(pending)
        with tempfile.TemporaryDirectory() as directory:
            with self._patched_runtime(turn_store):
                with patch.object(web_server, "user_data_dir", return_value=Path(directory)):
                    list(web_server.approve_turn_events("turn-1", {}))
            rules_path = remembered_approvals_path(Path(directory))

        self.assertIsNone(find_remembered_approval(rules_path, "npm run build"))

    def test_remember_refuses_delete_category_at_store_layer(self) -> None:
        pending = {
            "conversation_id": "conversation-1",
            "profile_name": "test-profile",
            "runtime_messages_before_batch": [],
            "approval_payload": {
                "command": "rm a.txt",
                "risk_category": "DELETE",
            },
        }
        turn_store = _FakeTurnStore(pending)
        with tempfile.TemporaryDirectory() as directory:
            with self._patched_runtime(turn_store):
                with patch.object(web_server, "user_data_dir", return_value=Path(directory)):
                    events = list(
                        web_server.approve_turn_events("turn-1", {"remember": True})
                    )
            rules_path = remembered_approvals_path(Path(directory))
            found = find_remembered_approval(rules_path, "rm a.txt")

        self.assertIsNone(found)
        self.assertFalse(
            any(event.get("title") == "已记住这条命令" for event in events if event.get("event") == "activity")
        )


if __name__ == "__main__":
    unittest.main()
