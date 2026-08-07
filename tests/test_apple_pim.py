from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from work_agent_core import web_server
from work_agent_core.host_services.apple_pim import (
    ApplePimService,
    ApplePimServiceError,
    register_apple_pim_tools,
)
from work_agent_core.tools import ToolRegistry


class _StubApplePimService(ApplePimService):
    def __init__(self) -> None:
        super().__init__(Path.cwd())
        self.calls: list[dict[str, object]] = []

    def _call(self, payload: dict[str, object], *, timeout_seconds: int = 20) -> dict[str, object]:
        self.calls.append({**payload, "timeout_seconds": timeout_seconds})
        return {
            "ok": True,
            "platform": "macos_eventkit",
            "events_authorization": "full_access",
            "reminders_authorization": "full_access",
        }


class _FakeApplePim:
    def __init__(self, *, reminders_access: str = "full_access") -> None:
        self.reminders_access = reminders_access
        self.calls: list[tuple[str, dict[str, object]]] = []

    def status(self) -> dict[str, object]:
        self.calls.append(("status", {}))
        return {
            "ok": True,
            "available": True,
            "events_authorization": "full_access",
            "reminders_authorization": self.reminders_access,
        }

    def request_access(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(("request_access", kwargs))
        return {"ok": True, "events_granted": True, "reminders_granted": True}

    def items(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(("items", kwargs))
        return {"ok": True, "events": [], "reminders": []}

    def create_reminder(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(("create_reminder", kwargs))
        return {"ok": True, "reminder": {"id": "reminder-1", "title": kwargs["title"]}}


class ApplePimServiceTests(unittest.TestCase):
    def test_non_macos_status_is_explicitly_unavailable(self) -> None:
        service = _StubApplePimService()
        with patch("work_agent_core.host_services.apple_pim.platform.system", return_value="Linux"):
            payload = service.status()
        self.assertTrue(payload["ok"])
        self.assertFalse(payload["available"])
        self.assertEqual(payload["events_authorization"], "unavailable")
        self.assertEqual(service.calls, [])

    def test_status_converts_host_errors_into_safe_unavailable_state(self) -> None:
        service = _StubApplePimService()
        with patch("work_agent_core.host_services.apple_pim.platform.system", return_value="Darwin"):
            with patch.object(service, "_call", side_effect=ApplePimServiceError("HOST_SERVICE_UNAVAILABLE", "编译失败")):
                payload = service.status()
        self.assertFalse(payload["available"])
        self.assertEqual(payload["error_code"], "HOST_SERVICE_UNAVAILABLE")
        self.assertEqual(payload["reason"], "编译失败")

    def test_service_creates_reminder_using_only_an_optional_list_name(self) -> None:
        service = _StubApplePimService()
        service.create_reminder(title="发送纪要", calendar_name="提醒事项", due_at="", notes="", priority=5)
        self.assertEqual(
            service.calls[-1],
            {
                "action": "create_reminder",
                "title": "发送纪要",
                "calendar_name": "提醒事项",
                "due_at": "",
                "notes": "",
                "priority": 5,
                "timeout_seconds": 20,
            },
        )
        self.assertFalse(hasattr(service, "create_event"))


class ApplePimSkillToolTests(unittest.TestCase):
    def _registry(self, service: _FakeApplePim | _StubApplePimService) -> ToolRegistry:
        registry = ToolRegistry()
        register_apple_pim_tools(registry, service)
        return registry

    def test_skill_surface_reads_and_adds_only_reminders(self) -> None:
        service = _FakeApplePim()
        registry = self._registry(service)
        names = sorted(tool.name for tool in registry.list())
        self.assertEqual(names, ["create_apple_reminder", "get_apple_schedule_status", "list_apple_schedule"])

        listed = json.loads(
            registry.get("list_apple_schedule").handler({"include_events": False, "include_reminders": True})
        )
        self.assertTrue(listed["ok"])
        self.assertEqual(service.calls[-1], ("items", {
            "start_at": "", "end_at": "", "include_events": False, "include_reminders": True,
        }))

        blocked = json.loads(registry.get("create_apple_reminder").handler({"title": "发送纪要", "user_confirmed": False}))
        self.assertFalse(blocked["ok"])
        self.assertEqual(blocked["error"]["code"], "USER_CONFIRMATION_REQUIRED")

        created = json.loads(registry.get("create_apple_reminder").handler({
            "title": "发送纪要",
            "list_name": "提醒事项",
            "priority": 5,
            "user_confirmed": True,
        }))
        self.assertTrue(created["ok"])
        self.assertEqual(created["reminder"]["id"], "reminder-1")
        self.assertEqual(service.calls[-1], ("create_reminder", {
            "title": "发送纪要", "calendar_name": "提醒事项", "due_at": "", "notes": "", "priority": 5,
        }))

    def test_reminder_write_requires_eventkit_permission(self) -> None:
        service = _FakeApplePim(reminders_access="denied")
        registry = self._registry(service)
        payload = json.loads(registry.get("create_apple_reminder").handler({"title": "发送纪要", "user_confirmed": True}))
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "PERMISSION_REQUIRED")
        self.assertEqual([call[0] for call in service.calls], ["status"])


class ApplePimWebPayloadTests(unittest.TestCase):
    def test_browser_has_no_apple_write_payloads(self) -> None:
        self.assertFalse(hasattr(web_server, "create_apple_calendar_event_payload"))
        self.assertFalse(hasattr(web_server, "create_apple_reminder_payload"))

    def test_request_access_reuses_one_host_service_instance(self) -> None:
        service = _FakeApplePim()
        with patch.object(web_server, "apple_pim_service", return_value=service) as factory:
            payload = web_server.apple_pim_request_access_payload(events=True, reminders=False)
        self.assertTrue(payload["ok"])
        factory.assert_called_once_with()
        self.assertEqual(service.calls, [
            ("request_access", {"events": True, "reminders": False}),
            ("status", {}),
        ])


if __name__ == "__main__":
    unittest.main()
