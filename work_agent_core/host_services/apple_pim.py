from __future__ import annotations

"""Typed EventKit bridge for the Mac user's iCloud Calendar and Reminders.

EventKit is intentionally reached through a small Swift helper rather than
AppleScript or a generic shell tool.  The helper accepts a fixed JSON protocol,
never sees an arbitrary command, and macOS remains the source of truth for
permission prompts and data synchronization with iPhone/iCloud.
"""

from hashlib import sha256
from pathlib import Path
from typing import Any
import json
import os
import platform
import subprocess
import tempfile

from ..tools import Tool, ToolRegistry


HELPER_SOURCE_NAME = "apple_pim_helper.swift"
HELPER_BINARY_NAME = "work-agent-apple-pim"
MAX_HELPER_OUTPUT_BYTES = 2 * 1024 * 1024


class ApplePimServiceError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class ApplePimService:
    """Minimal host service; access is mediated by macOS EventKit/TCC.

    The source workspace contains only the reviewed helper.  A content-hashed
    binary is built in the OS temporary directory so it neither pollutes a user
    account workspace nor grants the untrusted execution plane access to host
    Calendar data.
    """

    def __init__(self, runtime_workspace_root: str | Path) -> None:
        self.runtime_workspace_root = Path(runtime_workspace_root).resolve()
        self.helper_source = Path(__file__).with_name(HELPER_SOURCE_NAME)

    def status(self) -> dict[str, Any]:
        if platform.system() != "Darwin":
            return {
                "ok": True,
                "available": False,
                "platform": platform.system().lower() or "unknown",
                "reason": "Apple 日历和提醒事项只能在 macOS 上通过 EventKit 使用。",
                "events_authorization": "unavailable",
                "reminders_authorization": "unavailable",
            }
        try:
            payload = self._call({"action": "status"})
        except ApplePimServiceError as error:
            return {
                "ok": True,
                "available": False,
                "platform": "macos_eventkit",
                "reason": error.message,
                "error_code": error.code,
                "events_authorization": "unavailable",
                "reminders_authorization": "unavailable",
            }
        return {
            "ok": True,
            "available": True,
            "platform": str(payload.get("platform") or "macos_eventkit"),
            "events_authorization": str(payload.get("events_authorization") or "unknown"),
            "reminders_authorization": str(payload.get("reminders_authorization") or "unknown"),
        }

    def request_access(self, *, events: bool, reminders: bool) -> dict[str, Any]:
        if not events and not reminders:
            raise ApplePimServiceError("INVALID_INPUT", "请至少选择日历或提醒事项之一。")
        return self._call(
            {
                "action": "request_access",
                "events": bool(events),
                "reminders": bool(reminders),
            },
            timeout_seconds=90,
        )

    def items(
        self,
        *,
        start_at: str = "",
        end_at: str = "",
        include_events: bool = True,
        include_reminders: bool = True,
    ) -> dict[str, Any]:
        if not include_events and not include_reminders:
            raise ApplePimServiceError("INVALID_INPUT", "请至少选择日历事项或提醒事项之一。")
        return self._call(
            {
                "action": "list_items",
                "start_at": start_at.strip(),
                "end_at": end_at.strip(),
                "include_events": bool(include_events),
                "include_reminders": bool(include_reminders),
            }
        )

    def create_reminder(
        self,
        *,
        title: str,
        calendar_name: str = "",
        due_at: str = "",
        notes: str = "",
        priority: int = 0,
    ) -> dict[str, Any]:
        normalized_priority = max(0, min(int(priority or 0), 9))
        return self._call(
            {
                "action": "create_reminder",
                "title": _required(title, "标题"),
                "calendar_name": calendar_name.strip(),
                "due_at": due_at.strip(),
                "notes": notes.strip(),
                "priority": normalized_priority,
            }
        )

    def _call(self, payload: dict[str, Any], *, timeout_seconds: int = 20) -> dict[str, Any]:
        helper = self._helper_binary()
        try:
            completed = subprocess.run(
                [str(helper)],
                input=json.dumps(payload, ensure_ascii=False),
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
                env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"},
            )
        except subprocess.TimeoutExpired as error:
            raise ApplePimServiceError("HOST_SERVICE_TIMEOUT", "Apple 日历服务响应超时，请稍后重试。") from error
        except OSError as error:
            raise ApplePimServiceError("HOST_SERVICE_UNAVAILABLE", f"无法启动 Apple 日历服务：{error}") from error
        raw_output = (completed.stdout or "").strip()
        if len(raw_output.encode("utf-8", errors="replace")) > MAX_HELPER_OUTPUT_BYTES:
            raise ApplePimServiceError("HOST_SERVICE_FAILED", "Apple 日历服务返回内容过大，已拒绝读取。")
        if completed.returncode != 0:
            detail = (completed.stderr or raw_output or f"退出码 {completed.returncode}").strip()
            raise ApplePimServiceError("HOST_SERVICE_FAILED", f"Apple 日历服务执行失败：{detail[:1000]}")
        try:
            response = json.loads(raw_output)
        except json.JSONDecodeError as error:
            raise ApplePimServiceError("HOST_SERVICE_FAILED", "Apple 日历服务没有返回有效 JSON。") from error
        if not isinstance(response, dict):
            raise ApplePimServiceError("HOST_SERVICE_FAILED", "Apple 日历服务返回格式无效。")
        if response.get("ok") is not True:
            raw_error = response.get("error") if isinstance(response.get("error"), dict) else {}
            code = str(raw_error.get("code") or "HOST_SERVICE_FAILED")
            message = str(raw_error.get("message") or "Apple 日历服务请求失败。")
            raise ApplePimServiceError(code, message)
        return response

    def _helper_binary(self) -> Path:
        if platform.system() != "Darwin":
            raise ApplePimServiceError("PLATFORM_UNSUPPORTED", "Apple 日历和提醒事项只能在 macOS 上使用。")
        if not self.helper_source.is_file():
            raise ApplePimServiceError("HOST_SERVICE_UNAVAILABLE", "缺少 Apple EventKit helper 源文件。")
        source_hash = sha256(self.helper_source.read_bytes()).hexdigest()[:20]
        helper_root = Path(tempfile.gettempdir()) / "work-agent-host-services" / "apple-pim" / source_hash
        binary = helper_root / HELPER_BINARY_NAME
        if binary.is_file() and os.access(binary, os.X_OK):
            return binary
        helper_root.mkdir(parents=True, exist_ok=True)
        temporary_binary = helper_root / f"{HELPER_BINARY_NAME}.building"
        swiftc = "/usr/bin/xcrun"
        try:
            completed = subprocess.run(
                [
                    swiftc,
                    "swiftc",
                    str(self.helper_source),
                    "-framework",
                    "EventKit",
                    "-framework",
                    "Foundation",
                    "-o",
                    str(temporary_binary),
                ],
                text=True,
                capture_output=True,
                timeout=120,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise ApplePimServiceError("HOST_SERVICE_UNAVAILABLE", f"无法编译 Apple EventKit helper：{error}") from error
        if completed.returncode != 0 or not temporary_binary.is_file():
            detail = (completed.stderr or completed.stdout or "未知编译错误").strip()
            raise ApplePimServiceError("HOST_SERVICE_UNAVAILABLE", f"Apple EventKit helper 编译失败：{detail[:1200]}")
        temporary_binary.chmod(0o700)
        temporary_binary.replace(binary)
        return binary


def _required(value: str, label: str) -> str:
    clean = str(value or "").strip()
    if not clean:
        raise ApplePimServiceError("INVALID_INPUT", f"{label}不能为空。")
    return clean


def register_apple_pim_tools(registry: ToolRegistry, service: ApplePimService) -> None:
    """Expose reads and explicitly requested Reminder creation to ReAct.

    The Schedule page remains read-only. A model may create one Reminder only
    after a direct user instruction in the current conversation; it can never
    create Calendar events or infer a reminder from work context.
    """

    def list_schedule_handler(args: dict[str, Any]) -> str:
        requested_events = bool(args.get("include_events", True))
        requested_reminders = bool(args.get("include_reminders", True))
        status = service.status()
        can_read_events = status.get("events_authorization") == "full_access"
        can_read_reminders = status.get("reminders_authorization") == "full_access"
        include_events = requested_events and can_read_events
        include_reminders = requested_reminders and can_read_reminders
        if not include_events and not include_reminders:
            return json.dumps(
                {
                    "ok": False,
                    "error": {
                        "code": "PERMISSION_REQUIRED",
                        "message": "所请求的 Apple 日历或提醒事项尚未获得 macOS 完整读取权限。请在日程与提醒页面授权。",
                    },
                    "status": status,
                },
                ensure_ascii=False,
                indent=2,
            )
        return json.dumps(
            service.items(
                start_at=str(args.get("start_at") or ""),
                end_at=str(args.get("end_at") or ""),
                include_events=include_events,
                include_reminders=include_reminders,
            ),
            ensure_ascii=False,
            indent=2,
        )

    registry.register(
        Tool(
            name="get_apple_schedule_status",
            description=(
                "Read the local Mac EventKit permission state for Apple Calendar and Reminders. "
                "The data is the iCloud/Apple-account data already synchronized to this Mac; "
                "never claim it is available until macOS full access is granted."
            ),
            parameters={"type": "object", "properties": {}},
            handler=lambda _args: json.dumps(service.status(), ensure_ascii=False, indent=2),
        )
    )
    registry.register(
        Tool(
            name="list_apple_schedule",
            description=(
                "Read Apple Calendar events and incomplete Apple Reminders in a bounded ISO-8601 time range. "
                "For a user's 待办/待办事项, call this with include_events=false and include_reminders=true. "
                "If macOS permission is missing, tell the user to authorize it in 日程与提醒; "
                "do not invent events or reminders."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "start_at": {"type": "string", "description": "Optional ISO 8601 start; default is yesterday."},
                    "end_at": {"type": "string", "description": "Optional ISO 8601 end; default is 30 days ahead."},
                    "include_events": {"type": "boolean", "default": True},
                    "include_reminders": {"type": "boolean", "default": True},
                },
            },
            handler=list_schedule_handler,
        )
    )

    def create_reminder_handler(args: dict[str, Any]) -> str:
        if args.get("user_confirmed") is not True:
            return json.dumps(
                {
                    "ok": False,
                    "error": {
                        "code": "USER_CONFIRMATION_REQUIRED",
                        "message": "只有用户在当前对话中明确要求新增待办或提醒时，才能写入 Apple 提醒事项。",
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        status = service.status()
        if status.get("reminders_authorization") != "full_access":
            return json.dumps(
                {
                    "ok": False,
                    "error": {
                        "code": "PERMISSION_REQUIRED",
                        "message": "Apple 提醒事项尚未获得 macOS 完整读取权限。请在日程与提醒页面授权后再试。",
                    },
                    "status": status,
                },
                ensure_ascii=False,
                indent=2,
            )
        result = service.create_reminder(
            title=str(args.get("title") or ""),
            calendar_name=str(args.get("list_name") or ""),
            due_at=str(args.get("due_at") or ""),
            notes=str(args.get("notes") or ""),
            priority=int(args.get("priority") or 0),
        )
        return json.dumps({**result, "message": "已新增到 Apple 提醒事项。"}, ensure_ascii=False, indent=2)

    registry.register(
        Tool(
            name="create_apple_reminder",
            description=(
                "Create exactly one Apple Reminder. Use only when the user explicitly asks in the current "
                "conversation to 新增/添加/创建 a 待办 or 提醒; never infer one from a meeting, project, or work context. "
                "Set user_confirmed=true only for that direct request. Omit due_at unless the user supplied a time. "
                "list_name is optional and must exactly match an Apple Reminders list; omit it to use the system default list."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Reminder title stated or approved by the user."},
                    "due_at": {"type": "string", "description": "Optional ISO 8601 due time. Omit when the user did not provide one."},
                    "notes": {"type": "string", "description": "Optional notes explicitly provided by the user."},
                    "priority": {"type": "integer", "minimum": 0, "maximum": 9, "default": 0},
                    "list_name": {"type": "string", "description": "Optional exact Apple Reminders list title. Omit for the system default list."},
                    "user_confirmed": {"type": "boolean", "description": "True only when the user explicitly requested this reminder in the current conversation."},
                },
                "required": ["title", "user_confirmed"],
            },
            handler=create_reminder_handler,
        )
    )
