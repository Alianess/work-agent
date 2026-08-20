"""What the assistant did, derived from its own record rather than the disk.

Scanning the filesystem to work out what happened is archaeology: it can only
see the residue, it cannot tell a revision from an unrelated file, and it knows
nothing about who asked for what. Everything the assistant handled is already
in the session log — every write it made, every prompt it was given — so the
ledger is a projection of that log, not a second survey of the outcome.

Rule-based scanning still belongs to input the assistant has *not* handled:
a recording that just arrived, a calendar item, a feed it subscribes to. Those
are things the world changed without telling it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import PurePosixPath
from typing import Any, Iterable
import json

from .session_log import SessionLog, TOOL_CALL, TOOL_RESULT, USER_MESSAGE


# Tools whose call produces a durable work product, and the argument naming the
# file they produce.
ARTIFACT_TOOLS: dict[str, tuple[str, ...]] = {
    "write_text_file": ("path",),
    "edit_text_file": ("path",),
    "apply_unified_patch": ("path",),
    "create_docx_from_markdown": ("output_path", "path"),
    "save_work_report": ("path", "output_path"),
}

# Internal scratch that is not a work product the user would recognise.
IGNORED_PATH_PARTS = {
    "_unpack_",
    ".work_agent_tmp",
    "execution",
    "debug_traces",
    "office_extracts",
    "file_previews",
    "asr_full",
}


@dataclass
class Revision:
    """One recorded write to an artifact."""

    seq: int
    path: str
    tool: str
    ts_ms: int
    turn_id: str = ""

    @property
    def when(self) -> datetime:
        return datetime.fromtimestamp(self.ts_ms / 1000)


@dataclass
class Artifact:
    """One work product and every revision the assistant made to it.

    Identity is the containing folder plus the shared stem, which is how a
    person refers to "the material" regardless of how many filenames it grew.
    """

    key: str
    title: str
    folder: str
    revisions: list[Revision] = field(default_factory=list)

    @property
    def latest(self) -> Revision | None:
        return max(self.revisions, key=lambda item: item.seq, default=None)

    @property
    def distinct_paths(self) -> list[str]:
        seen: list[str] = []
        for revision in sorted(self.revisions, key=lambda item: item.seq):
            if revision.path not in seen:
                seen.append(revision.path)
        return seen

    @property
    def touched_at(self) -> datetime | None:
        latest = self.latest
        return latest.when if latest else None


@dataclass
class WorkLedger:
    """The projection: what was produced, and what was asked."""

    artifacts: dict[str, Artifact] = field(default_factory=dict)
    prompts: list[str] = field(default_factory=list)

    def sorted_artifacts(self) -> list[Artifact]:
        return sorted(
            self.artifacts.values(),
            key=lambda item: -(item.latest.seq if item.latest else 0),
        )


def _is_ignored(path: str) -> bool:
    return any(part in path for part in IGNORED_PATH_PARTS)


def _artifact_identity(path: str) -> tuple[str, str, str]:
    """Return (key, title, folder) for a produced file.

    A folder under 材料/ or 会议项目/ names one deliverable, so every file in it
    is a revision of the same thing. Elsewhere the filename stem is the identity.
    """

    pure = PurePosixPath(path)
    folder = str(pure.parent)
    stem = pure.stem
    parts = pure.parts
    for anchor in ("材料", "会议项目", "projects"):
        if anchor in parts:
            index = parts.index(anchor)
            if index + 1 < len(parts) - 1:
                title = parts[index + 1]
                return f"{anchor}/{title}", title, "/".join(parts[: index + 2])
    return f"{folder}/{stem}", stem, folder


def build_work_ledger(log: SessionLog) -> WorkLedger:
    """Project one conversation's log into what it produced and was asked."""
    ledger = WorkLedger()
    failed_calls = {
        str(dict(event.data).get("call_id") or "")
        for event in log.iter_type(TOOL_RESULT)
        if str(dict(event.data).get("error") or "")
    }
    for event in log.events:
        data = dict(event.data)
        if event.type == USER_MESSAGE:
            content = str(data.get("content") or "").strip()
            if content:
                ledger.prompts.append(content)
            continue
        if event.type != TOOL_CALL:
            continue
        tool = str(data.get("name") or "")
        argument_names = ARTIFACT_TOOLS.get(tool)
        if not argument_names:
            continue
        if str(data.get("call_id") or "") in failed_calls:
            continue
        raw_arguments = data.get("arguments")
        if isinstance(raw_arguments, str):
            try:
                arguments = json.loads(raw_arguments or "{}")
            except json.JSONDecodeError:
                continue
        else:
            arguments = dict(raw_arguments or {})
        path = ""
        for name in argument_names:
            candidate = str(arguments.get(name) or "").strip()
            if candidate:
                path = candidate
                break
        if not path or _is_ignored(path):
            continue
        key, title, folder = _artifact_identity(path)
        artifact = ledger.artifacts.setdefault(key, Artifact(key=key, title=title, folder=folder))
        artifact.revisions.append(
            Revision(seq=event.seq, path=path, tool=tool, ts_ms=event.ts_ms)
        )
    return ledger


def merge_ledgers(ledgers: Iterable[WorkLedger]) -> WorkLedger:
    """Combine per-conversation ledgers into one account-wide view."""
    merged = WorkLedger()
    for ledger in ledgers:
        for key, artifact in ledger.artifacts.items():
            target = merged.artifacts.setdefault(
                key, Artifact(key=key, title=artifact.title, folder=artifact.folder)
            )
            target.revisions.extend(artifact.revisions)
        merged.prompts.extend(ledger.prompts)
    return merged
