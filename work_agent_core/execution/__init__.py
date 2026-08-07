"""Durable, policy-driven execution plane for Work Agent.

The package deliberately owns every untrusted process boundary.  Callers submit
an :class:`ExecutionRequest`; they never invoke a backend subprocess directly.
"""

from .models import (
    BackendKind,
    CapabilitySet,
    CommandSpec,
    DeliveryStatus,
    ExecutionClass,
    ExecutionMode,
    ExecutionRequest,
    ExecutionStatus,
)
from .orchestrator import ExecutionOrchestrator
from .store import ExecutionStore

__all__ = [
    "BackendKind",
    "CapabilitySet",
    "CommandSpec",
    "DeliveryStatus",
    "ExecutionClass",
    "ExecutionMode",
    "ExecutionOrchestrator",
    "ExecutionRequest",
    "ExecutionStatus",
    "ExecutionStore",
]
