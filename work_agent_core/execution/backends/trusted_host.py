from __future__ import annotations

from ..models import BackendKind
from .base import ProcessExecutionBackend


class TrustedHostBackend(ProcessExecutionBackend):
    """Explicit opt-in host runner used only after a scoped user approval."""

    kind = BackendKind.TRUSTED_HOST
