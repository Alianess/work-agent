from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import threading
import time
import uuid


_NOTIFICATION_LOCK = threading.RLock()


class NotificationStore:
    """Small account-local inbox for Friday's one-way reminders."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def list(self, *, limit: int = 100) -> list[dict[str, Any]]:
        with _NOTIFICATION_LOCK:
            items = [
                item
                for item in self._read()
                if item.get("kind") == "reminder" and int(item.get("delivered_at") or 0) > 0
            ]
        return sorted(
            items,
            key=lambda item: int(item.get("created_at") or 0),
            reverse=True,
        )[: max(1, min(int(limit), 500))]

    def add(
        self,
        *,
        kind: str = "reminder",
        title: str,
        body: str,
        source: str = "friday",
        deliver_at: int = 0,
    ) -> dict[str, Any]:
        normalized_kind = str(kind or "").strip()
        if normalized_kind not in {"reminder", "conversation"}:
            raise ValueError("通知类型必须是 reminder 或 conversation。")
        now = int(time.time())
        scheduled_at = max(now, int(deliver_at or now))
        item = {
            "id": f"notice-{uuid.uuid4().hex}",
            "kind": normalized_kind,
            "title": str(title or "").strip() or "Friday 提醒",
            "body": str(body or "").strip(),
            "source": str(source or "friday").strip() or "friday",
            "created_at": now,
            "deliver_at": scheduled_at,
            "delivered_at": now if normalized_kind == "reminder" and scheduled_at <= now else 0,
            "read_at": 0,
        }
        if not item["body"]:
            raise ValueError("提醒内容不能为空。")
        with _NOTIFICATION_LOCK:
            items = self._read()
            items.append(item)
            self._write(items[-500:])
        return item

    def claim_due(self, *, now: int | None = None, kind: str | None = None) -> list[dict[str, Any]]:
        current = int(now or time.time())
        due: list[dict[str, Any]] = []
        with _NOTIFICATION_LOCK:
            items = self._read()
            for item in items:
                if kind is not None and str(item.get("kind") or "") != kind:
                    continue
                if int(item.get("delivered_at") or 0) > 0:
                    continue
                if int(item.get("deliver_at") or 0) > current:
                    continue
                item["delivered_at"] = current
                due.append(dict(item))
            if due:
                self._write(items)
        return due

    def due_conversations(self, *, now: int | None = None) -> list[dict[str, Any]]:
        """Return scheduled conversation prompts without marking them delivered.

        A conversation prompt is only complete once it has actually been
        appended to Friday's durable chat.  Marking it first could lose it when
        the archive write fails.
        """
        current = int(now or time.time())
        with _NOTIFICATION_LOCK:
            return [
                dict(item)
                for item in self._read()
                if item.get("kind") == "conversation"
                and int(item.get("delivered_at") or 0) == 0
                and int(item.get("deliver_at") or 0) <= current
            ]

    def mark_delivered(self, notification_id: str, *, now: int | None = None) -> bool:
        target_id = str(notification_id or "").strip()
        if not target_id:
            return False
        delivered_at = int(now or time.time())
        with _NOTIFICATION_LOCK:
            items = self._read()
            for item in items:
                if str(item.get("id") or "") != target_id:
                    continue
                if int(item.get("delivered_at") or 0) > 0:
                    return True
                item["delivered_at"] = delivered_at
                self._write(items)
                return True
        return False

    def mark_read(self, notification_id: str = "", *, all_items: bool = False) -> int:
        now = int(time.time())
        changed = 0
        with _NOTIFICATION_LOCK:
            items = self._read()
            for item in items:
                if item.get("read_at"):
                    continue
                if all_items or str(item.get("id") or "") == notification_id:
                    item["read_at"] = now
                    changed += 1
            if changed:
                self._write(items)
        return changed

    def delete(self, notification_id: str) -> bool:
        target_id = str(notification_id or "").strip()
        if not target_id:
            return False
        with _NOTIFICATION_LOCK:
            items = self._read()
            remaining = [item for item in items if str(item.get("id") or "") != target_id]
            if len(remaining) == len(items):
                return False
            self._write(remaining)
        return True

    def payload(self) -> dict[str, Any]:
        items = self.list()
        return {
            "items": items,
            "unread_count": sum(1 for item in items if not item.get("read_at")),
        }

    def _read(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        raw_items = payload.get("items") if isinstance(payload, dict) else []
        return [dict(item) for item in raw_items if isinstance(item, dict)]

    def _write(self, items: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"items": items}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.path)
