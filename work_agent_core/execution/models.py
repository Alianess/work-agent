from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any


class _StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class ExecutionMode(_StringEnum):
    ISOLATED = "isolated"
    TRUSTED_HOST = "trusted_host"


class BackendKind(_StringEnum):
    MACOS_SEATBELT = "macos_seatbelt"
    TRUSTED_HOST = "trusted_host"


class ExecutionClass(_StringEnum):
    IN_PROCESS = "in_process"
    ISOLATED_PROCESS = "isolated_process"
    HOST_SERVICE = "host_service"
    EXTERNAL_ACTION = "external_action"


class ExecutionStatus(_StringEnum):
    QUEUED = "queued"
    PREPARING = "preparing"
    READY = "ready"
    RUNNING = "running"
    WAITING_PERMISSION = "waiting_permission"
    VALIDATING = "validating"
    AWAITING_APPLY = "awaiting_apply"
    APPLYING = "applying"
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EnvironmentStatus(_StringEnum):
    CREATING = "creating"
    READY = "ready"
    BUSY = "busy"
    STOPPING = "stopping"
    STOPPED = "stopped"
    BROKEN = "broken"


class DeliveryStatus(_StringEnum):
    NONE = "none"
    CHANGES_READY = "changes_ready"
    VALIDATED = "validated"
    CONFLICTED = "conflicted"
    APPLIED = "applied"
    REJECTED = "rejected"
    APPLY_FAILED = "apply_failed"


class PermissionDecisionValue(_StringEnum):
    ALLOW_ONCE = "allow_once"
    DENY = "deny"


@dataclass(frozen=True)
class FilesystemScope:
    read_roots: tuple[str, ...] = ("project",)
    write_roots: tuple[str, ...] = ("workspace", "artifacts")
    deny_roots: tuple[str, ...] = ("host_home", "other_accounts", "docker_socket")
    allow_symlinks_within_roots: bool = True
    max_written_bytes: int = 2 * 1024 * 1024 * 1024


@dataclass(frozen=True)
class NetworkScope:
    mode: str = "deny"
    allowed_domains: tuple[str, ...] = ()
    allowed_methods: tuple[str, ...] = ("GET", "HEAD", "POST", "PUT", "PATCH", "DELETE")
    deny_private_networks: bool = True
    deny_loopback: bool = True
    deny_link_local: bool = True
    max_bytes_in: int = 512 * 1024 * 1024
    max_bytes_out: int = 64 * 1024 * 1024


@dataclass(frozen=True)
class ResourceScope:
    wall_timeout_seconds: int = 900
    cpu_seconds: int = 600
    memory_bytes: int = 4 * 1024 * 1024 * 1024
    pids: int = 64
    open_files: int = 1024
    stdout_bytes: int = 20 * 1024 * 1024
    stderr_bytes: int = 20 * 1024 * 1024


@dataclass(frozen=True)
class SecretRef:
    name: str
    purpose: str
    delivery: str = "proxy"
    required: bool = True


@dataclass(frozen=True)
class CapabilitySet:
    filesystem: FilesystemScope = field(default_factory=FilesystemScope)
    network: NetworkScope = field(default_factory=NetworkScope)
    resources: ResourceScope = field(default_factory=ResourceScope)
    secrets: tuple[SecretRef, ...] = ()
    host_services: tuple[str, ...] = ()
    allow_background_services: bool = False


@dataclass(frozen=True)
class CommandSpec:
    argv: tuple[str, ...]
    cwd: str = "."
    env: dict[str, str] = field(default_factory=dict)
    stdin_text: str | None = None
    shell: bool = False


@dataclass(frozen=True)
class ValidationSpec:
    kind: str
    target: str
    options: dict[str, Any] = field(default_factory=dict)
    required: bool = True


@dataclass(frozen=True)
class HostServiceCall:
    service: str
    operation: str
    input_refs: tuple[str, ...]
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionRequest:
    request_id: str
    idempotency_key: str
    account_id: str
    turn_id: str
    conversation_id: str
    project_id: str
    tool_call_id: str
    tool_name: str
    execution_class: ExecutionClass
    mode: ExecutionMode
    command: CommandSpec | None
    requested_capabilities: CapabilitySet = field(default_factory=CapabilitySet)
    granted_permission_ids: tuple[str, ...] = ()
    host_service: HostServiceCall | None = None
    validations: tuple[ValidationSpec, ...] = ()
    delivery_mode: str = "apply_after_validation"
    reason: str = ""
    created_at_ms: int = 0


@dataclass(frozen=True)
class ExecutionContract:
    contract_id: str
    execution_id: str
    policy_version: str
    account_id: str
    project_id: str
    mode: ExecutionMode
    backend: BackendKind
    workspace_snapshot_id: str
    capabilities: CapabilitySet
    granted_permission_ids: tuple[str, ...]
    delivery_mode: str
    issued_at_ms: int
    expires_at_ms: int
    digest: str


@dataclass(frozen=True)
class ProcessOutcome:
    exit_code: int | None
    signal: int | None
    timed_out: bool
    cancelled: bool
    stdout_ref: str
    stderr_ref: str
    started_at_ms: int
    finished_at_ms: int


@dataclass(frozen=True)
class ValidationOutcome:
    validation_id: str
    kind: str
    target: str
    status: str
    detail: str
    evidence_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class FileChange:
    path: str
    change_type: str
    base_hash: str | None
    result_hash: str | None
    size_before: int | None
    size_after: int | None
    binary: bool
    diff_ref: str | None = None


@dataclass(frozen=True)
class ChangeSet:
    change_set_id: str
    execution_id: str
    snapshot_id: str
    changes: tuple[FileChange, ...]
    generated_at_ms: int
    digest: str


@dataclass(frozen=True)
class PermissionRequest:
    permission_request_id: str
    execution_id: str
    contract_id: str
    capability: str
    requested_scope: dict[str, Any]
    reason: str
    user_impact: str
    alternatives: tuple[str, ...]
    risk_level: str
    policy_code: str
    requested_by_tool: str
    requested_at_ms: int
    expires_at_ms: int
    remember_allowed: bool = False


@dataclass(frozen=True)
class PermissionDecision:
    permission_request_id: str
    decision: PermissionDecisionValue
    decided_by: str
    decided_at_ms: int
    expected_contract_digest: str
    client_nonce: str


@dataclass(frozen=True)
class ExecutionError:
    code: str
    message: str
    retryable: bool = False
    phase: str = ""
    user_action: str = ""
    detail_ref: str = ""


@dataclass(frozen=True)
class ExecutionResult:
    execution_id: str
    status: ExecutionStatus
    process: ProcessOutcome | None
    validations: tuple[ValidationOutcome, ...] = ()
    artifact_ids: tuple[str, ...] = ()
    change_set_id: str | None = None
    delivery_status: DeliveryStatus = DeliveryStatus.NONE
    error: ExecutionError | None = None
    receipt_id: str = ""


def jsonable(value: Any) -> Any:
    """Convert execution dataclasses into JSON-safe deterministic data."""
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    return value


def capability_set_from_dict(data: dict[str, Any]) -> CapabilitySet:
    filesystem_data = data.get("filesystem") if isinstance(data.get("filesystem"), dict) else {}
    network_data = data.get("network") if isinstance(data.get("network"), dict) else {}
    resources_data = data.get("resources") if isinstance(data.get("resources"), dict) else {}
    secrets_data = data.get("secrets") if isinstance(data.get("secrets"), list) else []
    return CapabilitySet(
        filesystem=FilesystemScope(
            read_roots=tuple(str(item) for item in filesystem_data.get("read_roots") or ("project",)),
            write_roots=tuple(str(item) for item in filesystem_data.get("write_roots") or ("workspace", "artifacts")),
            deny_roots=tuple(str(item) for item in filesystem_data.get("deny_roots") or ("host_home", "other_accounts", "docker_socket")),
            allow_symlinks_within_roots=bool(filesystem_data.get("allow_symlinks_within_roots", True)),
            max_written_bytes=int(filesystem_data.get("max_written_bytes") or 2 * 1024 * 1024 * 1024),
        ),
        network=NetworkScope(
            mode=str(network_data.get("mode") or "deny"),
            allowed_domains=tuple(str(item) for item in network_data.get("allowed_domains") or ()),
            allowed_methods=tuple(str(item) for item in network_data.get("allowed_methods") or ("GET", "HEAD", "POST", "PUT", "PATCH", "DELETE")),
            deny_private_networks=bool(network_data.get("deny_private_networks", True)),
            deny_loopback=bool(network_data.get("deny_loopback", True)),
            deny_link_local=bool(network_data.get("deny_link_local", True)),
            max_bytes_in=int(network_data.get("max_bytes_in") or 512 * 1024 * 1024),
            max_bytes_out=int(network_data.get("max_bytes_out") or 64 * 1024 * 1024),
        ),
        resources=ResourceScope(
            wall_timeout_seconds=int(resources_data.get("wall_timeout_seconds") or 900),
            cpu_seconds=int(resources_data.get("cpu_seconds") or 600),
            memory_bytes=int(resources_data.get("memory_bytes") or 4 * 1024 * 1024 * 1024),
            pids=int(resources_data.get("pids") or 64),
            open_files=int(resources_data.get("open_files") or 1024),
            stdout_bytes=int(resources_data.get("stdout_bytes") or 20 * 1024 * 1024),
            stderr_bytes=int(resources_data.get("stderr_bytes") or 20 * 1024 * 1024),
        ),
        secrets=tuple(
            SecretRef(
                name=str(item.get("name") or ""),
                purpose=str(item.get("purpose") or ""),
                delivery=str(item.get("delivery") or "proxy"),
                required=bool(item.get("required", True)),
            )
            for item in secrets_data
            if isinstance(item, dict) and str(item.get("name") or "")
        ),
        host_services=tuple(str(item) for item in data.get("host_services") or ()),
        allow_background_services=bool(data.get("allow_background_services")),
    )


def execution_request_from_dict(data: dict[str, Any]) -> ExecutionRequest:
    command_data = data.get("command") if isinstance(data.get("command"), dict) else None
    command = None
    if command_data is not None:
        command = CommandSpec(
            argv=tuple(str(item) for item in command_data.get("argv") or ()),
            cwd=str(command_data.get("cwd") or "."),
            env={str(key): str(value) for key, value in (command_data.get("env") or {}).items()},
            stdin_text=str(command_data["stdin_text"]) if command_data.get("stdin_text") is not None else None,
            shell=bool(command_data.get("shell")),
        )
    host_data = data.get("host_service") if isinstance(data.get("host_service"), dict) else None
    host_service = None
    if host_data is not None:
        host_service = HostServiceCall(
            service=str(host_data.get("service") or ""),
            operation=str(host_data.get("operation") or ""),
            input_refs=tuple(str(item) for item in host_data.get("input_refs") or ()),
            parameters=dict(host_data.get("parameters") or {}),
        )
    validations = tuple(
        ValidationSpec(
            kind=str(item.get("kind") or ""),
            target=str(item.get("target") or ""),
            options=dict(item.get("options") or {}),
            required=bool(item.get("required", True)),
        )
        for item in data.get("validations") or ()
        if isinstance(item, dict)
    )
    return ExecutionRequest(
        request_id=str(data.get("request_id") or ""),
        idempotency_key=str(data.get("idempotency_key") or ""),
        account_id=str(data.get("account_id") or ""),
        turn_id=str(data.get("turn_id") or ""),
        conversation_id=str(data.get("conversation_id") or ""),
        project_id=str(data.get("project_id") or ""),
        tool_call_id=str(data.get("tool_call_id") or ""),
        tool_name=str(data.get("tool_name") or ""),
        execution_class=ExecutionClass(str(data.get("execution_class") or ExecutionClass.ISOLATED_PROCESS.value)),
        mode=ExecutionMode(str(data.get("mode") or ExecutionMode.ISOLATED.value)),
        command=command,
        requested_capabilities=capability_set_from_dict(
            data.get("requested_capabilities") if isinstance(data.get("requested_capabilities"), dict) else {}
        ),
        granted_permission_ids=tuple(str(item) for item in data.get("granted_permission_ids") or ()),
        host_service=host_service,
        validations=validations,
        delivery_mode=str(data.get("delivery_mode") or "apply_after_validation"),
        reason=str(data.get("reason") or ""),
        created_at_ms=int(data.get("created_at_ms") or 0),
    )
