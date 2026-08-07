from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Callable
import json
import os
import time
import uuid

from .backends import ExecutionBackend, SeatbeltBackend, TrustedHostBackend
from .errors import ExecutionFailure, failure
from .events import EventEmitter, ExecutionEvent
from .models import (
    BackendKind,
    ChangeSet,
    DeliveryStatus,
    ExecutionContract,
    ExecutionError,
    ExecutionMode,
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
    PermissionDecision,
    PermissionDecisionValue,
    PermissionRequest,
    ProcessOutcome,
    ValidationOutcome,
    execution_request_from_dict,
    jsonable,
)
from .policy import PolicyEngine, contract_digest
from .store import ExecutionRecord, ExecutionStore
from .validation import ValidationService
from .workspace import WorkspaceManager, WorkspaceSnapshot


EventCallback = Callable[[ExecutionEvent], None]
CancelCheck = Callable[[], bool]


class ExecutionOrchestrator:
    """Single entry point for untrusted command execution.

    The orchestrator is deliberately synchronous because the existing ToolBus
    handler contract is synchronous.  It still streams durable events through
    ``EventEmitter`` and can be called from threaded HTTP/SSE routes.
    """

    def __init__(
        self,
        *,
        workspace_root: str | Path,
        runtime_workspace_root: str | Path | None = None,
        execution_root: str | Path | None = None,
        store: ExecutionStore | None = None,
        policy: PolicyEngine | None = None,
        backends: dict[BackendKind, ExecutionBackend] | None = None,
        validator: ValidationService | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.runtime_workspace_root = Path(runtime_workspace_root or self.workspace_root).resolve()
        self.execution_root = Path(execution_root or (self.workspace_root / "meet_files" / "execution")).resolve()
        self.execution_root.mkdir(parents=True, exist_ok=True)
        self.store = store or ExecutionStore(self.execution_root)
        self.policy = policy or PolicyEngine()
        self.workspace = WorkspaceManager(self.execution_root)
        self.validator = validator or ValidationService()
        self.backends: dict[BackendKind, ExecutionBackend] = backends or {
            BackendKind.MACOS_SEATBELT: SeatbeltBackend(runtime_workspace_root=self.runtime_workspace_root),
            BackendKind.TRUSTED_HOST: TrustedHostBackend(),
        }

    def preflight(self, request: ExecutionRequest) -> dict[str, Any]:
        decision = self.policy.evaluate(request)
        payload: dict[str, Any] = {
            "ok": decision.allowed,
            "policy_version": self.policy.policy_version,
            "effective_mode": request.mode.value,
            "backend": decision.backend.value if decision.backend else "",
            "capabilities": jsonable(decision.capabilities),
            "resources": jsonable(decision.capabilities.resources),
            "can_start": decision.allowed and decision.permission_request is None,
            "permission_request": jsonable(decision.permission_request) if decision.permission_request else None,
        }
        if not decision.allowed:
            payload["error"] = {"code": decision.denied_code, "message": decision.denied_reason}
            return payload
        assert decision.backend is not None
        health = self.backends[decision.backend].health()
        payload["backend_health"] = {"available": health.available, "detail": health.detail, "version": health.version}
        if not health.available:
            payload["ok"] = False
            payload["can_start"] = False
            payload["error"] = {
                "code": "BACKEND_UNAVAILABLE",
                "message": f"{decision.backend.value} 不可用：{health.detail}。系统不会退回宿主执行。",
            }
        return payload

    def submit(
        self,
        request: ExecutionRequest,
        *,
        source_root: str | Path | None = None,
        on_event: EventCallback | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> ExecutionResult:
        source = Path(source_root or self.workspace_root).resolve()
        existing = self.store.find_idempotent(request.account_id, request.idempotency_key)
        if existing is not None:
            return self._result_for_existing(existing)
        execution_id = f"exe_{uuid.uuid4().hex}"
        decision = self.policy.evaluate(request)
        backend = decision.backend or BackendKind.MACOS_SEATBELT
        record = self.store.create(
            execution_id=execution_id,
            account_id=request.account_id,
            turn_id=request.turn_id,
            conversation_id=request.conversation_id,
            project_id=request.project_id,
            tool_call_id=request.tool_call_id,
            idempotency_key=request.idempotency_key,
            mode=request.mode.value,
            backend=backend,
            request=jsonable(request),
        )
        if not decision.allowed:
            error = ExecutionError(
                code=decision.denied_code or "POLICY_DENIED",
                message=decision.denied_reason or "当前执行请求被安全策略拒绝。",
                phase="preflight",
                user_action="change_scope",
            )
            return self._finish_error(record, error, on_event=on_event)
        contract = self._contract(
            execution_id=execution_id,
            request=request,
            backend=backend,
            capabilities=decision.capabilities,
            snapshot_id="",
        )
        self.store.store_contract(contract)
        if decision.permission_request is not None:
            pending = replace(
                decision.permission_request,
                execution_id=execution_id,
                contract_id=contract.contract_id,
            )
            self.store.store_permission(pending)
            emitter = self._emitter(execution_id, on_event)
            emitter.emit(
                "permission.requested",
                phase=ExecutionStatus.WAITING_PERMISSION.value,
                summary="执行需要新的权限确认。",
                payload={"permission_request": jsonable(pending), "contract_digest": contract.digest},
            )
            self.store.update_status(execution_id, ExecutionStatus.WAITING_PERMISSION)
            return ExecutionResult(
                execution_id=execution_id,
                status=ExecutionStatus.WAITING_PERMISSION,
                process=None,
                delivery_status=DeliveryStatus.NONE,
                receipt_id="",
            )
        return self._run(record, request, source, contract, on_event=on_event, cancel_check=cancel_check)

    def resume_after_permission(
        self,
        execution_id: str,
        decision: PermissionDecision,
        *,
        source_root: str | Path | None = None,
        on_event: EventCallback | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> ExecutionResult:
        record = self.store.get(execution_id)
        permission = self.store.permission(decision.permission_request_id)
        request_payload = permission["request"]
        if str(request_payload.get("execution_id") or "") != execution_id:
            return self._finish_error(
                record,
                ExecutionError("POLICY_DENIED", "权限请求与执行任务不匹配。", phase="waiting_permission"),
                on_event=on_event,
            )
        if permission["status"] != "waiting":
            return self._result_for_existing(record)
        if int(request_payload.get("expires_at_ms") or 0) < int(time.time() * 1000):
            return self._finish_error(
                record,
                ExecutionError("PERMISSION_EXPIRED", "权限请求已过期，请重新发起任务。", phase="waiting_permission"),
                on_event=on_event,
            )
        contract_payload = record.contract or {}
        if decision.expected_contract_digest != str(contract_payload.get("digest") or ""):
            return self._finish_error(
                record,
                ExecutionError("POLICY_DENIED", "权限确认时执行契约已变化，请重新发起任务。", phase="waiting_permission"),
                on_event=on_event,
            )
        self.store.decide_permission(decision)
        if decision.decision is PermissionDecisionValue.DENY:
            return self._finish_error(
                record,
                ExecutionError("PERMISSION_DENIED", "用户拒绝了本次执行所需权限。", phase="waiting_permission"),
                on_event=on_event,
            )
        request = execution_request_from_dict(record.request)
        resumed_request = replace(
            request,
            granted_permission_ids=tuple(sorted(set(request.granted_permission_ids + (decision.permission_request_id,)))),
        )
        self.store.update_request(execution_id, jsonable(resumed_request))
        policy_decision = self.policy.evaluate(resumed_request)
        if not policy_decision.allowed or policy_decision.backend is None:
            return self._finish_error(
                record,
                ExecutionError(
                    policy_decision.denied_code or "POLICY_DENIED",
                    policy_decision.denied_reason or "权限恢复后策略拒绝执行。",
                    phase="preparing",
                ),
                on_event=on_event,
            )
        contract = self._contract(
            execution_id=execution_id,
            request=resumed_request,
            backend=policy_decision.backend,
            capabilities=policy_decision.capabilities,
            snapshot_id="",
        )
        self.store.store_contract(contract)
        return self._run(
            record,
            resumed_request,
            Path(source_root or self.workspace_root).resolve(),
            contract,
            on_event=on_event,
            cancel_check=cancel_check,
        )

    def apply_changes(
        self,
        execution_id: str,
        *,
        change_set_id: str,
        expected_digest: str,
        selected_paths: tuple[str, ...] | None = None,
    ) -> ExecutionResult:
        record = self.store.get(execution_id)
        change_set = self.workspace.load_change_set(change_set_id)
        if change_set.execution_id != execution_id or change_set.digest != expected_digest:
            return self._finish_error(
                record,
                ExecutionError("WORKSPACE_CONFLICT", "变更集与当前执行记录不匹配。", phase="applying"),
            )
        snapshot = self.workspace.load_snapshot(change_set.snapshot_id)
        self.store.update_status(execution_id, ExecutionStatus.APPLYING, delivery_status=DeliveryStatus.CHANGES_READY)
        try:
            applied = self.workspace.apply_changes(snapshot=snapshot, change_set=change_set, selected_paths=selected_paths)
        except ExecutionFailure as error:
            return self._finish_error(record, error.error)
        process, validations = self._stored_process_and_validations(record)
        result = ExecutionResult(
            execution_id=execution_id,
            status=ExecutionStatus.SUCCEEDED,
            process=process,
            validations=validations,
            change_set_id=change_set_id,
            delivery_status=DeliveryStatus.APPLIED,
            receipt_id=f"rcpt_{uuid.uuid4().hex}",
        )
        self._store_receipt(result, applied_paths=applied)
        return result

    def _run(
        self,
        record: ExecutionRecord,
        request: ExecutionRequest,
        source_root: Path,
        initial_contract: ExecutionContract,
        *,
        on_event: EventCallback | None,
        cancel_check: CancelCheck | None,
    ) -> ExecutionResult:
        execution_id = record.execution_id
        emitter = self._emitter(execution_id, on_event)
        backend = self.backends.get(initial_contract.backend)
        if backend is None:
            return self._finish_error(
                record,
                ExecutionError("BACKEND_UNAVAILABLE", f"未注册执行后端：{initial_contract.backend.value}", phase="preparing"),
                on_event=on_event,
            )
        health = backend.health()
        if not health.available:
            return self._finish_error(
                record,
                ExecutionError(
                    "BACKEND_UNAVAILABLE",
                    f"{initial_contract.backend.value} 不可用：{health.detail}。系统不会退回宿主执行。",
                    retryable=True,
                    phase="preparing",
                    user_action="repair_backend",
                ),
                on_event=on_event,
            )
        snapshot: WorkspaceSnapshot | None = None
        environment = None
        self.store.update_status(execution_id, ExecutionStatus.PREPARING)
        emitter.emit("execution.preparing", phase="preparing", summary="正在创建私有工作区和执行环境。")
        try:
            snapshot = self.workspace.create_snapshot(
                source_root=source_root,
                account_id=request.account_id,
                project_id=request.project_id,
            )
            contract = self._contract(
                execution_id=execution_id,
                request=request,
                backend=initial_contract.backend,
                capabilities=initial_contract.capabilities,
                snapshot_id=snapshot.snapshot_id,
            )
            self.store.store_contract(contract)
            log_dir = self.execution_root / "logs" / execution_id
            environment = backend.prepare(contract, workspace_path=snapshot.workspace_path, log_dir=log_dir)
            self.store.update_status(execution_id, ExecutionStatus.RUNNING)
            emitter.emit(
                "environment.ready",
                phase="running",
                summary="隔离执行环境已就绪。",
                payload={"backend": contract.backend.value, "environment_id": environment.environment_id, "snapshot_id": snapshot.snapshot_id},
            )
            if request.command is None:
                raise failure("POLICY_DENIED", "当前执行请求没有可运行的命令。", phase="preparing")
            process = backend.run(
                environment,
                request.command,
                contract.capabilities,
                emitter,
                cancel_check=cancel_check,
            )
            return self._settle_process(
                record,
                snapshot,
                process,
                request,
                emitter,
                on_event=on_event,
            )
        except ExecutionFailure as error:
            return self._finish_error(record, error.error, on_event=on_event)
        except Exception as error:
            return self._finish_error(
                record,
                ExecutionError("EXECUTION_FAILED", str(error) or type(error).__name__, phase="running"),
                on_event=on_event,
            )
        finally:
            if environment is not None:
                try:
                    backend.destroy(environment)
                except Exception as error:
                    emitter.emit(
                        "environment.cleanup_failed",
                        phase="complete",
                        summary="执行环境清理出现问题，已记录供恢复处理。",
                        payload={"detail": str(error)},
                        visibility="debug",
                    )

    def _settle_process(
        self,
        record: ExecutionRecord,
        snapshot: WorkspaceSnapshot,
        process: ProcessOutcome,
        request: ExecutionRequest,
        emitter: EventEmitter,
        *,
        on_event: EventCallback | None,
    ) -> ExecutionResult:
        if process.cancelled:
            return self._finish_error(
                record,
                ExecutionError("PROCESS_CANCELLED", "执行已取消，隔离进程树已终止。", phase="cancelled"),
                status=ExecutionStatus.CANCELLED,
                process=process,
                on_event=on_event,
            )
        if process.timed_out:
            return self._finish_error(
                record,
                ExecutionError("PROCESS_TIMEOUT", "执行超时，隔离进程树已终止。", retryable=True, phase="running"),
                process=process,
                on_event=on_event,
            )
        change_set = self.workspace.capture_changes(execution_id=record.execution_id, snapshot=snapshot)
        if process.exit_code != 0:
            receipt_status = ExecutionStatus.PARTIAL if change_set.changes else ExecutionStatus.FAILED
            return self._finish_error(
                record,
                ExecutionError("PROCESS_FAILED", f"命令退出码为 {process.exit_code}。", phase="running"),
                status=receipt_status,
                process=process,
                change_set=change_set,
                on_event=on_event,
            )
        self.store.update_status(record.execution_id, ExecutionStatus.VALIDATING)
        emitter.emit("validation.started", phase="validating", summary="正在核验执行产物。")
        validations = self.validator.validate(snapshot.workspace_path, request.validations)
        required_failures = [item for item in validations if item.status == "failed"]
        for item in validations:
            emitter.emit(
                "validation.completed",
                phase="validating",
                summary=f"验证 {item.kind}：{item.status}",
                payload=jsonable(item),
                visibility="debug",
            )
        if required_failures:
            return self._finish_error(
                record,
                ExecutionError("VALIDATION_FAILED", "执行产物未通过所需验证。", phase="validating"),
                process=process,
                validations=validations,
                change_set=change_set,
                on_event=on_event,
            )
        if change_set.changes and request.delivery_mode == "apply_after_validation":
            self.store.update_status(record.execution_id, ExecutionStatus.APPLYING, delivery_status=DeliveryStatus.VALIDATED)
            emitter.emit("delivery.applying", phase="applying", summary="验证通过，正在将变更原子写回工作区。")
            try:
                applied = self.workspace.apply_changes(snapshot=snapshot, change_set=change_set)
            except ExecutionFailure as error:
                return self._finish_error(
                    record,
                    error.error,
                    process=process,
                    validations=validations,
                    change_set=change_set,
                    on_event=on_event,
                )
            result = ExecutionResult(
                execution_id=record.execution_id,
                status=ExecutionStatus.SUCCEEDED,
                process=process,
                validations=validations,
                change_set_id=change_set.change_set_id,
                delivery_status=DeliveryStatus.APPLIED,
                receipt_id=f"rcpt_{uuid.uuid4().hex}",
            )
            emitter.emit("delivery.applied", phase="complete", summary="变更已写回并复核。", payload={"paths": applied})
            self._store_receipt(result, applied_paths=applied)
            return result
        if change_set.changes:
            result = ExecutionResult(
                execution_id=record.execution_id,
                status=ExecutionStatus.AWAITING_APPLY,
                process=process,
                validations=validations,
                change_set_id=change_set.change_set_id,
                delivery_status=DeliveryStatus.CHANGES_READY,
                receipt_id=f"rcpt_{uuid.uuid4().hex}",
            )
            emitter.emit("delivery.ready", phase="awaiting_apply", summary="变更已验证，等待写回确认。", payload={"change_set_id": change_set.change_set_id})
            self._store_receipt(result, applied_paths=[])
            return result
        result = ExecutionResult(
            execution_id=record.execution_id,
            status=ExecutionStatus.SUCCEEDED,
            process=process,
            validations=validations,
            delivery_status=DeliveryStatus.VALIDATED,
            receipt_id=f"rcpt_{uuid.uuid4().hex}",
        )
        emitter.emit("execution.completed", phase="complete", summary="执行与验证已完成。")
        self._store_receipt(result, applied_paths=[])
        return result

    def _finish_error(
        self,
        record: ExecutionRecord,
        error: ExecutionError,
        *,
        status: ExecutionStatus = ExecutionStatus.FAILED,
        process: ProcessOutcome | None = None,
        validations: tuple[ValidationOutcome, ...] = (),
        change_set: ChangeSet | None = None,
        on_event: EventCallback | None = None,
    ) -> ExecutionResult:
        result = ExecutionResult(
            execution_id=record.execution_id,
            status=status,
            process=process,
            validations=validations,
            change_set_id=change_set.change_set_id if change_set else None,
            delivery_status=DeliveryStatus.CHANGES_READY if change_set and change_set.changes else DeliveryStatus.NONE,
            error=error,
            receipt_id=f"rcpt_{uuid.uuid4().hex}",
        )
        self.store.update_status(record.execution_id, status, error=error, delivery_status=result.delivery_status)
        emitter = self._emitter(record.execution_id, on_event)
        emitter.emit(
            "execution.cancelled" if status is ExecutionStatus.CANCELLED else "execution.failed",
            phase=status.value,
            summary=error.message,
            payload={"error": jsonable(error), "change_set_id": result.change_set_id},
        )
        self._store_receipt(result, applied_paths=[])
        return result

    def _contract(
        self,
        *,
        execution_id: str,
        request: ExecutionRequest,
        backend: BackendKind,
        capabilities: Any,
        snapshot_id: str,
    ) -> ExecutionContract:
        issued = int(time.time() * 1000)
        payload = {
            "execution_id": execution_id,
            "policy_version": self.policy.policy_version,
            "account_id": request.account_id,
            "project_id": request.project_id,
            "mode": request.mode.value,
            "backend": backend.value,
            "workspace_snapshot_id": snapshot_id,
            "capabilities": jsonable(capabilities),
            "granted_permission_ids": list(request.granted_permission_ids),
            "delivery_mode": request.delivery_mode,
            "issued_at_ms": issued,
            "expires_at_ms": issued + capabilities.resources.wall_timeout_seconds * 1000 + 5 * 60 * 1000,
        }
        digest = contract_digest(payload)
        return ExecutionContract(
            contract_id=f"ctr_{uuid.uuid4().hex}",
            execution_id=execution_id,
            policy_version=self.policy.policy_version,
            account_id=request.account_id,
            project_id=request.project_id,
            mode=request.mode,
            backend=backend,
            workspace_snapshot_id=snapshot_id,
            capabilities=capabilities,
            granted_permission_ids=request.granted_permission_ids,
            delivery_mode=request.delivery_mode,
            issued_at_ms=issued,
            expires_at_ms=int(payload["expires_at_ms"]),
            digest=digest,
        )

    def _emitter(self, execution_id: str, on_event: EventCallback | None) -> EventEmitter:
        existing = self.store.events(execution_id)
        initial_index = int(existing[-1].get("event_index") or -1) if existing else -1
        return EventEmitter(execution_id, self.store.append_event, on_event, initial_index=initial_index)

    def _store_receipt(self, result: ExecutionResult, *, applied_paths: list[str]) -> None:
        receipt = {
            "receipt_id": result.receipt_id,
            "execution_id": result.execution_id,
            "status": result.status.value,
            "delivery_status": result.delivery_status.value,
            "process": jsonable(result.process),
            "validations": jsonable(result.validations),
            "change_set_id": result.change_set_id,
            "applied_paths": applied_paths,
            "error": jsonable(result.error),
            "created_at_ms": int(time.time() * 1000),
        }
        receipt_dir = self.execution_root / "receipts"
        receipt_dir.mkdir(parents=True, exist_ok=True)
        receipt_path = receipt_dir / f"{result.receipt_id}.json"
        temporary_path = receipt_path.with_suffix(".json.tmp")
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(receipt, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, receipt_path)
        self.store.store_result(result, receipt)

    def _result_for_existing(self, record: ExecutionRecord) -> ExecutionResult:
        if record.result:
            return _result_from_payload(record.execution_id, record.result)
        status = ExecutionStatus(record.status)
        if status is ExecutionStatus.WAITING_PERMISSION:
            return ExecutionResult(execution_id=record.execution_id, status=status, process=None)
        error = None
        if record.error:
            error = ExecutionError(
                code=str(record.error.get("code") or "EXECUTION_FAILED"),
                message=str(record.error.get("message") or "执行失败。"),
                retryable=bool(record.error.get("retryable")),
                phase=str(record.error.get("phase") or ""),
                user_action=str(record.error.get("user_action") or ""),
                detail_ref=str(record.error.get("detail_ref") or ""),
            )
        return ExecutionResult(execution_id=record.execution_id, status=status, process=None, error=error)

    def _stored_process_and_validations(self, record: ExecutionRecord) -> tuple[ProcessOutcome | None, tuple[ValidationOutcome, ...]]:
        if not record.result:
            return None, ()
        result = _result_from_payload(record.execution_id, record.result)
        return result.process, result.validations


def _result_from_payload(execution_id: str, payload: dict[str, Any]) -> ExecutionResult:
    process_payload = payload.get("process") if isinstance(payload.get("process"), dict) else None
    process = None
    if process_payload:
        process = ProcessOutcome(
            exit_code=process_payload.get("exit_code"),
            signal=process_payload.get("signal"),
            timed_out=bool(process_payload.get("timed_out")),
            cancelled=bool(process_payload.get("cancelled")),
            stdout_ref=str(process_payload.get("stdout_ref") or ""),
            stderr_ref=str(process_payload.get("stderr_ref") or ""),
            started_at_ms=int(process_payload.get("started_at_ms") or 0),
            finished_at_ms=int(process_payload.get("finished_at_ms") or 0),
        )
    validations = tuple(
        ValidationOutcome(
            validation_id=str(item.get("validation_id") or ""),
            kind=str(item.get("kind") or ""),
            target=str(item.get("target") or ""),
            status=str(item.get("status") or ""),
            detail=str(item.get("detail") or ""),
            evidence_refs=tuple(str(value) for value in item.get("evidence_refs") or ()),
        )
        for item in payload.get("validations") or ()
        if isinstance(item, dict)
    )
    error_payload = payload.get("error") if isinstance(payload.get("error"), dict) else None
    error = None
    if error_payload:
        error = ExecutionError(
            code=str(error_payload.get("code") or "EXECUTION_FAILED"),
            message=str(error_payload.get("message") or "执行失败。"),
            retryable=bool(error_payload.get("retryable")),
            phase=str(error_payload.get("phase") or ""),
            user_action=str(error_payload.get("user_action") or ""),
            detail_ref=str(error_payload.get("detail_ref") or ""),
        )
    return ExecutionResult(
        execution_id=execution_id,
        status=ExecutionStatus(str(payload.get("status") or ExecutionStatus.FAILED.value)),
        process=process,
        validations=validations,
        artifact_ids=tuple(str(item) for item in payload.get("artifact_ids") or ()),
        change_set_id=str(payload["change_set_id"]) if payload.get("change_set_id") else None,
        delivery_status=DeliveryStatus(str(payload.get("delivery_status") or DeliveryStatus.NONE.value)),
        error=error,
        receipt_id=str(payload.get("receipt_id") or ""),
    )
