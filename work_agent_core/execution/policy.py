from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import hashlib
import json
import time
import uuid

from .models import (
    BackendKind,
    CapabilitySet,
    ExecutionMode,
    ExecutionRequest,
    FilesystemScope,
    NetworkScope,
    PermissionRequest,
    ResourceScope,
)


POLICY_VERSION = "exec-policy-2026-08-v1"


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    backend: BackendKind | None
    capabilities: CapabilitySet
    denied_reason: str = ""
    denied_code: str = ""
    permission_request: PermissionRequest | None = None


def capability_template(name: str) -> CapabilitySet:
    standard = CapabilitySet()
    if name == "project_readonly":
        return CapabilitySet(
            filesystem=FilesystemScope(read_roots=("project",), write_roots=()),
            network=NetworkScope(mode="deny"),
            resources=standard.resources,
        )
    if name == "project_dependencies":
        return CapabilitySet(
            filesystem=standard.filesystem,
            network=NetworkScope(
                mode="domain_allowlist",
                allowed_domains=("pypi.org", "files.pythonhosted.org", "registry.npmjs.org"),
            ),
            resources=standard.resources,
        )
    if name == "document_render":
        return CapabilitySet(
            filesystem=standard.filesystem,
            network=NetworkScope(mode="deny"),
            resources=standard.resources,
            host_services=("office",),
        )
    if name == "meeting_processing":
        return CapabilitySet(
            filesystem=standard.filesystem,
            network=NetworkScope(mode="deny"),
            resources=ResourceScope(wall_timeout_seconds=3600, cpu_seconds=2400),
            host_services=("meeting_asr",),
        )
    return standard


class PolicyEngine:
    """Fixed-deny policy and capability contract compiler.

    This engine intentionally evaluates semantic capability requests rather
    than command-name allowlists.  The shell layer may still request user
    confirmation for UX, but cannot turn a denied capability into an allowed
    one.
    """

    def __init__(
        self,
        *,
        policy_version: str = POLICY_VERSION,
        default_backend: BackendKind = BackendKind.MACOS_SEATBELT,
        allow_trusted_host: bool = True,
    ) -> None:
        self.policy_version = policy_version
        self.default_backend = default_backend
        self.allow_trusted_host = allow_trusted_host

    def evaluate(self, request: ExecutionRequest) -> PolicyDecision:
        caps = self._normalize(request.requested_capabilities)
        denied = self._fixed_denial(request, caps)
        if denied is not None:
            return PolicyDecision(
                allowed=False,
                backend=None,
                capabilities=caps,
                denied_reason=denied[1],
                denied_code=denied[0],
            )

        backend = self._backend_for(request)
        permission = self._permission_if_needed(request, caps, backend)
        if permission is None and backend is BackendKind.MACOS_SEATBELT and caps.network.mode != "deny":
            return PolicyDecision(
                allowed=False,
                backend=backend,
                capabilities=caps,
                denied_code="NETWORK_BROKER_UNAVAILABLE",
                denied_reason=(
                    "原生隔离执行尚未配置受控域名代理；为避免放宽为任意网络访问，本次请求不会启动。"
                ),
            )
        return PolicyDecision(
            allowed=True,
            backend=backend,
            capabilities=caps,
            permission_request=permission,
        )

    def _normalize(self, caps: CapabilitySet) -> CapabilitySet:
        network = caps.network
        normalized_domains = tuple(sorted({domain.strip().lower() for domain in network.allowed_domains if domain.strip()}))
        normalized_network = NetworkScope(
            mode=network.mode if network.mode in {"deny", "domain_allowlist", "unrestricted"} else "deny",
            allowed_domains=normalized_domains,
            allowed_methods=tuple(sorted(set(network.allowed_methods))),
            deny_private_networks=True,
            deny_loopback=True,
            deny_link_local=True,
            max_bytes_in=max(0, network.max_bytes_in),
            max_bytes_out=max(0, network.max_bytes_out),
        )
        resources = caps.resources
        normalized_resources = ResourceScope(
            wall_timeout_seconds=min(max(1, resources.wall_timeout_seconds), 4 * 3600),
            cpu_seconds=min(max(1, resources.cpu_seconds), 3 * 3600),
            memory_bytes=min(max(128 * 1024 * 1024, resources.memory_bytes), 32 * 1024 * 1024 * 1024),
            pids=min(max(1, resources.pids), 512),
            open_files=min(max(32, resources.open_files), 4096),
            stdout_bytes=min(max(1024, resources.stdout_bytes), 64 * 1024 * 1024),
            stderr_bytes=min(max(1024, resources.stderr_bytes), 64 * 1024 * 1024),
        )
        return CapabilitySet(
            filesystem=caps.filesystem,
            network=normalized_network,
            resources=normalized_resources,
            secrets=caps.secrets,
            host_services=tuple(sorted(set(caps.host_services))),
            allow_background_services=bool(caps.allow_background_services),
        )

    def _fixed_denial(self, request: ExecutionRequest, caps: CapabilitySet) -> tuple[str, str] | None:
        if request.command is not None and request.command.shell:
            return "POLICY_DENIED", "隔离执行不支持由模型请求 Shell 解释器。请使用 argv 形式的命令。"
        if caps.network.mode == "unrestricted":
            return "POLICY_DENIED", "隔离执行不允许不受限制的网络访问。"
        if any(item.delivery == "env" for item in caps.secrets):
            return "POLICY_DENIED", "隔离执行不允许通过环境变量注入原始凭据。"
        if caps.allow_background_services:
            return "POLICY_DENIED", "普通任务不能创建未受 Lease 管理的后台服务。"
        sensitive_roots = {root.lower() for root in caps.filesystem.read_roots + caps.filesystem.write_roots}
        if sensitive_roots & {"host_home", "other_accounts", "docker_socket", "system"}:
            return "POLICY_DENIED", "请求的文件范围包含固定拒绝的宿主敏感区域。"
        if request.mode is ExecutionMode.TRUSTED_HOST and not self.allow_trusted_host:
            return "POLICY_DENIED", "当前账户策略不允许宿主执行。"
        return None

    def _backend_for(self, request: ExecutionRequest) -> BackendKind:
        if request.mode is ExecutionMode.TRUSTED_HOST:
            return BackendKind.TRUSTED_HOST
        return self.default_backend

    def _permission_if_needed(
        self,
        request: ExecutionRequest,
        caps: CapabilitySet,
        backend: BackendKind,
    ) -> PermissionRequest | None:
        if request.granted_permission_ids:
            return None
        capability = ""
        requested_scope: dict[str, Any] = {}
        reason = ""
        user_impact = ""
        alternatives: tuple[str, ...] = ()
        risk = "medium"
        code = ""
        if request.mode is ExecutionMode.TRUSTED_HOST:
            capability = "trusted_host_execution"
            requested_scope = {"command": list(request.command.argv) if request.command else [], "backend": backend.value}
            reason = request.reason or "该任务需要直接使用本机进程环境。"
            user_impact = "命令将直接在本机运行，不具备隔离环境的文件和进程边界。"
            alternatives = ("保持隔离执行",)
            risk = "high"
            code = "TRUSTED_HOST_REQUIRED"
        elif caps.network.mode == "domain_allowlist" and caps.network.allowed_domains:
            capability = "network_domains"
            requested_scope = {"domains": list(caps.network.allowed_domains), "methods": list(caps.network.allowed_methods)}
            reason = request.reason or "当前任务需要连接指定服务。"
            user_impact = "隔离环境将仅能访问列出的域名，私网和本机地址仍被阻止。"
            alternatives = ("离线完成", "由用户预先提供所需材料")
            risk = "medium"
            code = "NETWORK_SCOPE_REQUIRED"
        elif caps.secrets:
            capability = "secret_broker"
            requested_scope = {"secrets": [{"name": item.name, "purpose": item.purpose} for item in caps.secrets]}
            reason = request.reason or "当前任务需要通过受控凭据代理访问服务。"
            user_impact = "环境不会得到原始密钥；仅在约定用途和时长内使用受控凭据。"
            alternatives = ("不使用凭据执行",)
            risk = "high"
            code = "SECRET_SCOPE_REQUIRED"
        if not capability:
            return None
        now = int(time.time() * 1000)
        return PermissionRequest(
            permission_request_id=f"perm_{uuid.uuid4().hex}",
            execution_id="",
            contract_id="",
            capability=capability,
            requested_scope=requested_scope,
            reason=reason,
            user_impact=user_impact,
            alternatives=alternatives,
            risk_level=risk,
            policy_code=code,
            requested_by_tool=request.tool_name,
            requested_at_ms=now,
            expires_at_ms=now + 5 * 60 * 1000,
        )


def contract_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()
