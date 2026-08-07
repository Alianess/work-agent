from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import sqlite3
import threading
import time

from .events import ExecutionEvent
from .models import (
    BackendKind,
    DeliveryStatus,
    ExecutionContract,
    ExecutionError,
    ExecutionResult,
    ExecutionStatus,
    PermissionDecision,
    PermissionRequest,
    jsonable,
)


@dataclass(frozen=True)
class ExecutionRecord:
    execution_id: str
    account_id: str
    turn_id: str
    conversation_id: str
    project_id: str
    tool_call_id: str
    idempotency_key: str
    mode: str
    backend: str
    status: str
    phase: str
    delivery_status: str
    request: dict[str, Any]
    contract: dict[str, Any] | None
    result: dict[str, Any] | None
    error: dict[str, Any] | None
    created_at_ms: int
    updated_at_ms: int


class ExecutionStore:
    """Account-local durable store for execution state, events and receipts."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.database_path = self.root / "execution.sqlite3"
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS execution_jobs (
                    execution_id TEXT PRIMARY KEY,
                    account_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    tool_call_id TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    backend TEXT NOT NULL,
                    status TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    delivery_status TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    contract_json TEXT,
                    result_json TEXT,
                    error_json TEXT,
                    created_at_ms INTEGER NOT NULL,
                    updated_at_ms INTEGER NOT NULL,
                    UNIQUE(account_id, idempotency_key)
                );
                CREATE TABLE IF NOT EXISTS execution_events (
                    execution_id TEXT NOT NULL,
                    event_index INTEGER NOT NULL,
                    event_json TEXT NOT NULL,
                    created_at_ms INTEGER NOT NULL,
                    PRIMARY KEY (execution_id, event_index),
                    FOREIGN KEY (execution_id) REFERENCES execution_jobs(execution_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS execution_permissions (
                    permission_request_id TEXT PRIMARY KEY,
                    execution_id TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    decision_json TEXT,
                    status TEXT NOT NULL,
                    created_at_ms INTEGER NOT NULL,
                    updated_at_ms INTEGER NOT NULL,
                    FOREIGN KEY (execution_id) REFERENCES execution_jobs(execution_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS execution_receipts (
                    receipt_id TEXT PRIMARY KEY,
                    execution_id TEXT NOT NULL,
                    receipt_json TEXT NOT NULL,
                    created_at_ms INTEGER NOT NULL,
                    FOREIGN KEY (execution_id) REFERENCES execution_jobs(execution_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS execution_jobs_turn_idx ON execution_jobs(account_id, turn_id, updated_at_ms);
                CREATE INDEX IF NOT EXISTS execution_events_idx ON execution_events(execution_id, event_index);
                """
            )

    def create(
        self,
        *,
        execution_id: str,
        account_id: str,
        turn_id: str,
        conversation_id: str,
        project_id: str,
        tool_call_id: str,
        idempotency_key: str,
        mode: str,
        backend: BackendKind,
        request: dict[str, Any],
        status: ExecutionStatus = ExecutionStatus.QUEUED,
    ) -> ExecutionRecord:
        now = int(time.time() * 1000)
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO execution_jobs (
                  execution_id, account_id, turn_id, conversation_id, project_id, tool_call_id,
                  idempotency_key, mode, backend, status, phase, delivery_status, request_json,
                  created_at_ms, updated_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    execution_id,
                    account_id,
                    turn_id,
                    conversation_id,
                    project_id,
                    tool_call_id,
                    idempotency_key,
                    mode,
                    backend.value,
                    status.value,
                    status.value,
                    DeliveryStatus.NONE.value,
                    _dump(request),
                    now,
                    now,
                ),
            )
        return self.get(execution_id)

    def get(self, execution_id: str) -> ExecutionRecord:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM execution_jobs WHERE execution_id = ?", (execution_id,)
            ).fetchone()
        if row is None:
            raise FileNotFoundError(f"Execution not found: {execution_id}")
        return _record_from_row(row)

    def find_idempotent(self, account_id: str, idempotency_key: str) -> ExecutionRecord | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM execution_jobs WHERE account_id = ? AND idempotency_key = ?",
                (account_id, idempotency_key),
            ).fetchone()
        return _record_from_row(row) if row is not None else None

    def update_status(
        self,
        execution_id: str,
        status: ExecutionStatus,
        *,
        phase: str | None = None,
        delivery_status: DeliveryStatus | None = None,
        error: ExecutionError | None = None,
    ) -> ExecutionRecord:
        now = int(time.time() * 1000)
        values: list[Any] = [status.value, phase or status.value]
        query = "UPDATE execution_jobs SET status = ?, phase = ?"
        if delivery_status is not None:
            query += ", delivery_status = ?"
            values.append(delivery_status.value)
        if error is not None:
            query += ", error_json = ?"
            values.append(_dump(jsonable(error)))
        query += ", updated_at_ms = ? WHERE execution_id = ?"
        values.extend([now, execution_id])
        with self._lock, self._connect() as connection:
            connection.execute(query, values)
        return self.get(execution_id)

    def update_request(self, execution_id: str, request: dict[str, Any]) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE execution_jobs SET request_json = ?, updated_at_ms = ? WHERE execution_id = ?",
                (_dump(request), int(time.time() * 1000), execution_id),
            )

    def store_contract(self, contract: ExecutionContract) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE execution_jobs SET contract_json = ?, updated_at_ms = ? WHERE execution_id = ?",
                (_dump(jsonable(contract)), int(time.time() * 1000), contract.execution_id),
            )

    def append_event(self, event: ExecutionEvent) -> ExecutionEvent:
        with self._lock, self._connect() as connection:
            existing = connection.execute(
                "SELECT COALESCE(MAX(event_index), -1) AS last_index FROM execution_events WHERE execution_id = ?",
                (event.execution_id,),
            ).fetchone()
            expected = int(existing["last_index"]) + 1 if existing is not None else 0
            stored = event
            if event.index != expected:
                stored = ExecutionEvent(
                    execution_id=event.execution_id,
                    index=expected,
                    type=event.type,
                    ts_ms=event.ts_ms,
                    phase=event.phase,
                    summary=event.summary,
                    payload=event.payload,
                    visibility=event.visibility,
                )
            connection.execute(
                "INSERT INTO execution_events (execution_id, event_index, event_json, created_at_ms) VALUES (?, ?, ?, ?)",
                (stored.execution_id, stored.index, _dump(stored.as_dict()), stored.ts_ms),
            )
        return stored

    def events(self, execution_id: str, *, after: int = -1) -> list[dict[str, Any]]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT event_json FROM execution_events WHERE execution_id = ? AND event_index > ? ORDER BY event_index",
                (execution_id, after),
            ).fetchall()
        return [_load(row["event_json"], {}) for row in rows]

    def store_permission(self, request: PermissionRequest) -> None:
        now = int(time.time() * 1000)
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO execution_permissions (
                  permission_request_id, execution_id, request_json, status, created_at_ms, updated_at_ms
                ) VALUES (?, ?, ?, 'waiting', ?, ?)
                """,
                (request.permission_request_id, request.execution_id, _dump(jsonable(request)), now, now),
            )

    def permission(self, permission_request_id: str) -> dict[str, Any]:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM execution_permissions WHERE permission_request_id = ?",
                (permission_request_id,),
            ).fetchone()
        if row is None:
            raise FileNotFoundError(f"Permission request not found: {permission_request_id}")
        return {
            "request": _load(row["request_json"], {}),
            "decision": _load(row["decision_json"], None),
            "status": str(row["status"]),
        }

    def permissions_for_execution(self, execution_id: str) -> list[dict[str, Any]]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT permission_request_id, request_json, decision_json, status, created_at_ms, updated_at_ms
                FROM execution_permissions WHERE execution_id = ? ORDER BY created_at_ms
                """,
                (execution_id,),
            ).fetchall()
        return [
            {
                "permission_request_id": str(row["permission_request_id"]),
                "request": _load(row["request_json"], {}),
                "decision": _load(row["decision_json"], None),
                "status": str(row["status"]),
                "created_at_ms": int(row["created_at_ms"]),
                "updated_at_ms": int(row["updated_at_ms"]),
            }
            for row in rows
        ]

    def decide_permission(self, decision: PermissionDecision) -> None:
        now = int(time.time() * 1000)
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE execution_permissions
                SET decision_json = ?, status = ?, updated_at_ms = ?
                WHERE permission_request_id = ? AND status = 'waiting'
                """,
                (
                    _dump(jsonable(decision)),
                    "allowed" if decision.decision.value == "allow_once" else "denied",
                    now,
                    decision.permission_request_id,
                ),
            )
        if cursor.rowcount != 1:
            raise ValueError("权限请求已处理、过期或不存在。")

    def store_result(self, result: ExecutionResult, receipt: dict[str, Any]) -> None:
        now = int(time.time() * 1000)
        result_payload = jsonable(result)
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE execution_jobs
                SET status = ?, phase = ?, delivery_status = ?, result_json = ?,
                    error_json = ?, updated_at_ms = ?
                WHERE execution_id = ?
                """,
                (
                    result.status.value,
                    result.status.value,
                    result.delivery_status.value,
                    _dump(result_payload),
                    _dump(jsonable(result.error)) if result.error is not None else None,
                    now,
                    result.execution_id,
                ),
            )
            connection.execute(
                "INSERT OR REPLACE INTO execution_receipts (receipt_id, execution_id, receipt_json, created_at_ms) VALUES (?, ?, ?, ?)",
                (result.receipt_id, result.execution_id, _dump(receipt), now),
            )

    def receipt(self, execution_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT receipt_json FROM execution_receipts WHERE execution_id = ? ORDER BY created_at_ms DESC LIMIT 1",
                (execution_id,),
            ).fetchone()
        return _load(row["receipt_json"], None) if row is not None else None


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _load(raw: str | None, fallback: Any) -> Any:
    if not raw:
        return fallback
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return fallback


def _record_from_row(row: sqlite3.Row) -> ExecutionRecord:
    return ExecutionRecord(
        execution_id=str(row["execution_id"]),
        account_id=str(row["account_id"]),
        turn_id=str(row["turn_id"]),
        conversation_id=str(row["conversation_id"]),
        project_id=str(row["project_id"]),
        tool_call_id=str(row["tool_call_id"]),
        idempotency_key=str(row["idempotency_key"]),
        mode=str(row["mode"]),
        backend=str(row["backend"]),
        status=str(row["status"]),
        phase=str(row["phase"]),
        delivery_status=str(row["delivery_status"]),
        request=_load(row["request_json"], {}),
        contract=_load(row["contract_json"], None),
        result=_load(row["result_json"], None),
        error=_load(row["error_json"], None),
        created_at_ms=int(row["created_at_ms"]),
        updated_at_ms=int(row["updated_at_ms"]),
    )
