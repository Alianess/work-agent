"""Facts the assistant extracts from its own turns, and how they are kept.

Extraction runs per turn rather than as a periodic sweep, so a commitment made
at 10am is known at 10am. Running a model on every turn would cost more than
the sweep it replaces, so a deterministic gate decides which turns are worth
reading: most turns carry nothing durable, and that is cheap to establish.

Extracted facts are appended to the session log like everything else. A fact
that only lived in a side table could not be traced back to the turn that
produced it, and replaying the log would not reproduce it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Protocol
import re

from .session_log import (
    ASSISTANT_MESSAGE,
    SessionEvent,
    SessionLog,
    TOOL_CALL,
    USER_MESSAGE,
)


FACT_RECORDED = "memory/fact"

COMMITMENT = "commitment"
ENTITY = "entity"
PREFERENCE = "preference"
DECISION = "decision"
FACT_KINDS = frozenset({COMMITMENT, ENTITY, PREFERENCE, DECISION})

# Reconciliation outcomes, following mem0's vocabulary.
ADD = "add"
UPDATE = "update"
NOOP = "noop"


@dataclass(frozen=True)
class Fact:
    kind: str
    subject: str
    """What the fact is about — the anchor used to find it again."""

    statement: str
    due_at: str = ""
    confidence: float = 0.6
    source_seqs: tuple[int, ...] = ()

    def to_data(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "subject": self.subject,
            "statement": self.statement,
            "due_at": self.due_at,
            "confidence": self.confidence,
        }

    @classmethod
    def from_data(cls, data: dict[str, Any], source_seqs: tuple[int, ...] = ()) -> "Fact":
        return cls(
            kind=str(data.get("kind") or ENTITY),
            subject=str(data.get("subject") or "").strip(),
            statement=str(data.get("statement") or "").strip(),
            due_at=str(data.get("due_at") or ""),
            confidence=float(data.get("confidence") or 0.6),
            source_seqs=source_seqs,
        )


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------

# Markers that a turn contains something worth keeping. Deliberately generous:
# a false positive costs one cheap model call, a false negative loses a fact.
_TIME_MARKERS = re.compile(
    r"(今天|明天|后天|下周|本周|周[一二三四五六日天]|下个?月|月底|季度末|"
    r"\d{1,2}\s*月\s*\d{1,2}\s*[日号]|\d{4}-\d{2}-\d{2}|"
    r"截止|之前|前提交|前交|deadline)"
)
_COMMITMENT_MARKERS = re.compile(
    r"(交|提交|给|出|写|做|发|报|定|办|安排|计划|负责|跟进|盯|约定|承诺|"
    r"记一下|记住|别忘|提醒我|完成|开会|见面|汇报)"
)
_CORRECTION_MARKERS = re.compile(r"(不是|应该是|错了|其实是|叫做|更正|改成|就是)")
# A standing obligation has no date but still needs following up: "定期报告
# 进展" is exactly the kind of promise that quietly lapses.
_RECURRING_MARKERS = re.compile(
    r"(定期|每天|每周|每月|每两周|双周|按期|周期性|常态化|持续(报告|反馈|汇报|跟进))"
)


@dataclass(frozen=True)
class GateDecision:
    worth_reading: bool
    reason: str


def gate_turn(log: SessionLog, *, from_seq: int = 0) -> GateDecision:
    """Decide whether this turn is worth spending a model call on.

    Reads only what the user said and what the assistant produced. Most turns
    are questions and answers that leave nothing durable behind.
    """

    user_text: list[str] = []
    produced = False
    for event in log.events:
        if event.seq < from_seq:
            continue
        data = dict(event.data)
        if event.type == USER_MESSAGE:
            user_text.append(str(data.get("content") or ""))
        elif event.type == TOOL_CALL:
            if str(data.get("name") or "") in {
                "write_text_file",
                "create_docx_from_markdown",
                "save_work_report",
                "update_plan",
            }:
                produced = True
    text = "\n".join(user_text)
    if _TIME_MARKERS.search(text):
        if _COMMITMENT_MARKERS.search(text):
            return GateDecision(True, "用户提到了带时间的约定")
        return GateDecision(True, "用户提到了具体时间点")
    if _RECURRING_MARKERS.search(text):
        return GateDecision(True, "用户提到了周期性义务")
    if _CORRECTION_MARKERS.search(text) and len(text) < 400:
        return GateDecision(True, "用户像是在更正一个说法")
    if produced:
        return GateDecision(True, "本轮产出了工作成果")
    return GateDecision(False, "本轮没有需要长期记住的内容")


# --------------------------------------------------------------------------
# Extraction and reconciliation
# --------------------------------------------------------------------------

class FactExtractor(Protocol):
    def extract(self, transcript: str, *, now: datetime) -> list[dict[str, Any]]: ...


def turn_transcript(log: SessionLog, *, from_seq: int = 0, limit_chars: int = 6000) -> str:
    """Render the turn for the extractor: what was asked and what was answered."""
    lines: list[str] = []
    for event in log.events:
        if event.seq < from_seq:
            continue
        data = dict(event.data)
        if event.type == USER_MESSAGE:
            lines.append(f"用户：{str(data.get('content') or '').strip()}")
        elif event.type == ASSISTANT_MESSAGE:
            content = str(data.get("content") or "").strip()
            if content:
                lines.append(f"助手：{content}")
    return "\n".join(lines)[-limit_chars:]


def existing_facts(log: SessionLog) -> list[Fact]:
    return [
        Fact.from_data(dict(event.data), source_seqs=event.source_event_seqs)
        for event in log.iter_type(FACT_RECORDED)
        if str(dict(event.data).get("operation") or ADD) != NOOP
    ]


def reconcile(candidate: Fact, known: Iterable[Fact]) -> tuple[str, Fact | None]:
    """Decide whether a candidate is new, an update, or already known.

    Without this step every turn appends another copy of the same fact and the
    store fills with near-duplicates — the memory equivalent of writing a new
    file for every revision.
    """

    for fact in known:
        if fact.kind != candidate.kind or fact.subject != candidate.subject:
            continue
        if fact.statement == candidate.statement and fact.due_at == candidate.due_at:
            return NOOP, None
        return UPDATE, candidate
    return ADD, candidate


def record_facts(
    runtime: Any,
    candidates: Iterable[Fact],
    *,
    known: Iterable[Fact] | None = None,
) -> list[tuple[str, Fact]]:
    """Append reconciled facts to the log through the conversation runtime."""
    settled = list(known if known is not None else existing_facts(runtime.log))
    applied: list[tuple[str, Fact]] = []
    for candidate in candidates:
        if not candidate.subject or not candidate.statement:
            continue
        if candidate.kind not in FACT_KINDS:
            continue
        operation, resolved = reconcile(candidate, settled)
        if operation == NOOP or resolved is None:
            continue
        runtime.log.append(
            FACT_RECORDED,
            {**resolved.to_data(), "operation": operation},
            source_event_seqs=resolved.source_seqs,
            surface_op="none",
        )
        settled.append(resolved)
        applied.append((operation, resolved))
    return applied
