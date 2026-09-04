from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import threading
import time
import uuid

REMEMBERED_APPROVAL_SCHEMA_VERSION = 1
REMEMBERED_APPROVALS_FILENAME = "remembered_approvals.json"
MAX_REMEMBERED_APPROVALS = 200
MAX_COMMAND_LENGTH = 2000
# 删除与系统级动作永远现场确认：记不住的审批才是这两类仅有的审批形态。
NEVER_REMEMBERED_RISK_CATEGORIES = frozenset({"DELETE", "SYSTEM"})

_RULES_LOCK = threading.RLock()


def remembered_approvals_path(user_data_dir: str | Path) -> Path:
    return Path(user_data_dir) / REMEMBERED_APPROVALS_FILENAME


def load_remembered_approvals(path: str | Path) -> list[dict[str, Any]]:
    rules_path = Path(path)
    with _RULES_LOCK:
        if not rules_path.is_file():
            return []
        try:
            payload = json.loads(rules_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return []
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        return []
    return [item for item in payload["items"] if isinstance(item, dict) and item.get("command")]


def find_remembered_approval(path: str | Path, command: str) -> dict[str, Any] | None:
    """Exact-command lookup only; an approved ``rm a.txt`` never covers ``rm b.txt``."""

    clean_command = str(command or "").strip()
    if not clean_command:
        return None
    for item in load_remembered_approvals(path):
        if str(item.get("command") or "").strip() != clean_command:
            continue
        risk_category = str(item.get("risk_category") or "")
        if risk_category in NEVER_REMEMBERED_RISK_CATEGORIES:
            return None
        return item
    return None


def remember_command_approval(
    path: str | Path,
    *,
    command: str,
    risk_category: str,
) -> dict[str, Any] | None:
    """Persist one exact command the user explicitly approved and chose to remember.

    Returns the stored rule, or ``None`` when the category is excluded or the
    command is empty: the caller surfaces nothing instead of silently keeping a
    rule that would never fire.
    """

    clean_command = str(command or "").strip()[:MAX_COMMAND_LENGTH]
    clean_risk = str(risk_category or "").strip().upper()
    if not clean_command or clean_risk in NEVER_REMEMBERED_RISK_CATEGORIES:
        return None
    rule = {
        "id": uuid.uuid4().hex[:12],
        "command": clean_command,
        "risk_category": clean_risk,
        "created_at": int(time.time()),
    }
    rules_path = Path(path)
    with _RULES_LOCK:
        items = [
            item
            for item in load_remembered_approvals(rules_path)
            if str(item.get("command") or "").strip() != clean_command
        ]
        items.append(rule)
        # 最老的先出局：记住的审批是便利，不是义务。
        items = items[-MAX_REMEMBERED_APPROVALS:]
        payload = {
            "schema_version": REMEMBERED_APPROVAL_SCHEMA_VERSION,
            "items": items,
            "updated_at": int(time.time()),
        }
        rules_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = rules_path.with_name(f".{rules_path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2)
                stream.flush()
            temporary.replace(rules_path)
        finally:
            if temporary.exists():
                temporary.unlink(missing_ok=True)
    return rule


def forget_command_approval(path: str | Path, command: str) -> bool:
    clean_command = str(command or "").strip()
    if not clean_command:
        return False
    rules_path = Path(path)
    with _RULES_LOCK:
        items = load_remembered_approvals(rules_path)
        remaining = [
            item
            for item in items
            if str(item.get("command") or "").strip() != clean_command
        ]
        if len(remaining) == len(items):
            return False
        payload = {
            "schema_version": REMEMBERED_APPROVAL_SCHEMA_VERSION,
            "items": remaining,
            "updated_at": int(time.time()),
        }
        rules_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = rules_path.with_name(f".{rules_path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2)
                stream.flush()
            temporary.replace(rules_path)
        finally:
            if temporary.exists():
                temporary.unlink(missing_ok=True)
    return True
