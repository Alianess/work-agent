from __future__ import annotations

from .models import ExecutionError


class ExecutionFailure(RuntimeError):
    def __init__(self, error: ExecutionError) -> None:
        super().__init__(error.message)
        self.error = error


def failure(
    code: str,
    message: str,
    *,
    retryable: bool = False,
    phase: str = "",
    user_action: str = "",
    detail_ref: str = "",
) -> ExecutionFailure:
    return ExecutionFailure(
        ExecutionError(
            code=code,
            message=message,
            retryable=retryable,
            phase=phase,
            user_action=user_action,
            detail_ref=detail_ref,
        )
    )
