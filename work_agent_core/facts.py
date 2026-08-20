"""What the assistant keeps knowing after a turn ends, and how it is kept.

Remembering is a tool the model calls, not a pipeline that runs behind it.
The model has already read the conversation, the transcript or the document;
asking a second pass to decide "was anything worth keeping" pays twice for one
reading, and gating that second pass on a keyword table only guarantees the
table will one day miss the thing that mattered.

Deadlines are deliberately absent from this module. A commitment with a date
belongs in Apple Reminders, where it syncs to the phone, rings on time, and is
forgotten by being ticked off — none of which a fact store does well, and all
of which it would have to reinvent. This module keeps only what has no better
home: names, preferences, settings and decisions.

Facts are appended to the session log like everything else. A fact that only
lived in a side table could not be traced back to the turn that produced it,
and replaying the log would not reproduce it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .session_log import SessionEvent, SessionLog


FACT_RECORDED = "memory/fact"

ENTITY = "entity"
PREFERENCE = "preference"
SETTING = "setting"
DECISION = "decision"
# Commitments are not here on purpose: a dated promise goes to Apple Reminders.
FACT_KINDS = frozenset({ENTITY, PREFERENCE, SETTING, DECISION})

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
    confidence: float = 0.6
    source_seqs: tuple[int, ...] = ()

    def to_data(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "subject": self.subject,
            "statement": self.statement,
            "confidence": self.confidence,
        }

    @classmethod
    def from_data(cls, data: dict[str, Any], source_seqs: tuple[int, ...] = ()) -> "Fact":
        return cls(
            kind=str(data.get("kind") or ENTITY),
            subject=str(data.get("subject") or "").strip(),
            statement=str(data.get("statement") or "").strip(),
            confidence=float(data.get("confidence") or 0.6),
            source_seqs=source_seqs,
        )


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
        if fact.statement == candidate.statement:
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
