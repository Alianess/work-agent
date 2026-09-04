"""Small outcome-plus-trajectory evaluator for historical production faults."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class RegressionExpectation:
    id: str
    symptom: str
    required_trace: tuple[str, ...]
    forbidden_trace: tuple[str, ...] = ()


HISTORICAL_REGRESSIONS: dict[str, RegressionExpectation] = {
    item.id: item
    for item in (
        RegressionExpectation(
            "missing_chat_after_restart",
            "事件已存在但浏览器归档缺失，刷新后对话消失",
            ("session/created", "user/message", "assistant/message", "session/checkpoint"),
        ),
        RegressionExpectation(
            "stale_error_delivery_state",
            "旧失败状态在每次打开或重启后重新显示为交付红点",
            ("user/message", "agent/error", "turn/end"),
        ),
        RegressionExpectation(
            "invalid_tool_json",
            "模型工具参数含控制字符或坏 JSON，整轮 HTTP 400",
            ("tool/arguments_invalid", "tool/call", "tool/result"),
        ),
        RegressionExpectation(
            "disconnect_after_tool_result",
            "连接中断后已经完成的工具结果丢失或被重复执行",
            ("tool/call", "tool/result", "turn/end"),
        ),
        RegressionExpectation(
            "unstructured_delivery",
            "文件只写在最终 Markdown 中，前端无法形成交付卡",
            ("tool/result", "artifact/created", "turn/end"),
        ),
        RegressionExpectation(
            "lost_mid_turn_followup",
            "用户在运行中补充的信息被提前清空，重启后丢失",
            ("external/queued", "external/consumed", "user/message"),
        ),
        RegressionExpectation(
            "unbounded_retrieval_context",
            "召回与历史全量塞入模型，长时间停在模型尚未开始返回",
            ("session/checkpoint", "request/header"),
        ),
    )
}


def evaluate_regression(
    regression_id: str,
    *,
    outcome_ok: bool,
    trajectory: Iterable[str | dict[str, Any]],
) -> list[str]:
    """Return human-readable failures for both result and execution path."""

    expectation = HISTORICAL_REGRESSIONS[regression_id]
    names = [
        str(item.get("type") or item.get("event") or "") if isinstance(item, dict) else str(item)
        for item in trajectory
    ]
    failures: list[str] = []
    if not outcome_ok:
        failures.append(f"{regression_id}: final outcome failed")
    if not is_ordered_subsequence(expectation.required_trace, names):
        failures.append(
            f"{regression_id}: required trajectory {expectation.required_trace!r} not in {names!r}"
        )
    forbidden = [name for name in expectation.forbidden_trace if name in names]
    if forbidden:
        failures.append(f"{regression_id}: forbidden trajectory present {forbidden!r}")
    return failures


def is_ordered_subsequence(required: Sequence[str], actual: Sequence[str]) -> bool:
    position = 0
    for item in actual:
        if position < len(required) and item == required[position]:
            position += 1
    return position == len(required)
