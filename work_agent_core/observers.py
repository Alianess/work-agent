"""Built-in observers: the things the assistant should notice on its own.

Each of these corresponds to something that actually went unnoticed while the
user worked. They read workspace state only — no model call — so the attention
pass stays cheap enough to run continuously.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Iterable
import json
import re

from .attention import FunctionObserver, Observation, ObserverContext, ObserverRegistry


MATERIALS_DIRNAME = "材料"
ARCHIVE_DIRNAME = "会议项目"
ATTACHMENTS_DIRNAME = "attachments"
TRANSCRIBABLE_SUFFIXES = {".m4a", ".mp3", ".wav", ".mp4", ".mov"}
DOCUMENT_SUFFIXES = {".md", ".docx", ".pdf"}
MANIFEST_FILENAME = "manifest.json"


def _string_values(value: object) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _string_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _string_values(child)


def _manifest_output_exists(data_root: Path, payload: dict[str, object]) -> bool:
    outputs = payload.get("canonical_outputs")
    if not isinstance(outputs, dict):
        return False
    for key in ("internal", "work_md", "work_docx"):
        raw = str(outputs.get(key) or "").strip()
        if not raw:
            continue
        path = Path(raw)
        if not path.is_absolute():
            path = data_root / path
        if path.is_file():
            return True
    return False


def _handled_recording_stems(data_root: Path, archives: Path) -> set[str]:
    """Return recordings referenced by a manifest with a real minutes output."""

    handled: set[str] = set()
    if not archives.is_dir():
        return handled
    for manifest in archives.rglob(MANIFEST_FILENAME):
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or not _manifest_output_exists(data_root, payload):
            continue
        references = "\n".join(_string_values(payload))
        for match in re.finditer(r"\b(\d{8}-\d{6}-[^/\n]+?)(?:\.m4a|\.mp3|\.wav|\.mp4|\.mov|_|/)", references, re.IGNORECASE):
            handled.add(match.group(1).strip())
    return handled


def _manifest_canonical_stems(data_root: Path, folder: str) -> set[str]:
    """Stems the folder's manifest registers as canonical deliverables.

    The manifest a meeting archive writes is the current-draft registry: a stem
    it covers already has a designated home, not version sprawl.
    """

    base = Path(folder)
    if not base.is_absolute():
        base = data_root / base
    try:
        payload = json.loads((base / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    outputs = payload.get("canonical_outputs") if isinstance(payload, dict) else None
    if not isinstance(outputs, dict):
        return set()
    return {
        PurePosixPath(str(value or "")).stem
        for value in outputs.values()
        if PurePosixPath(str(value or "")).stem
    }


def observe_material_versions(context: ObserverContext) -> Iterable[Observation]:
    """Notice work the assistant itself split across several parallel files.

    Read from the ledger, not the folder: the assistant made every one of these
    writes, so it should recall doing so rather than infer it from what is left
    on disk. Counting files also cannot tell a revision apart from an unrelated
    document that happens to sit in the same place.

    Two things that are not sprawl: renderings of one draft (the md source and
    its docx export share a stem) and files the folder's manifest registers as
    canonical deliverables. Only unregistered stems count.
    """

    ledger = context.ledger
    if ledger is None:
        return []
    observations: list[Observation] = []
    for artifact in ledger.sorted_artifacts():
        names = artifact.distinct_paths
        if len(names) < 3:
            continue
        stems_by_folder: dict[str, set[str]] = defaultdict(set)
        for name in names:
            pure = PurePosixPath(name)
            stems_by_folder[str(pure.parent)].add(pure.stem)
        for folder, stems in stems_by_folder.items():
            stems -= _manifest_canonical_stems(context.data_root, folder)
        untracked = sum(len(stems) for stems in stems_by_folder.values())
        if untracked < 3:
            continue
        latest = artifact.latest
        observations.append(
            Observation(
                key=f"versions:{artifact.key}:{untracked}",
                summary=(
                    f"《{artifact.title}》我先后写成了 {untracked} 份不同的稿子、"
                    f"改过 {len(artifact.revisions)} 次，没有哪一个被标记为当前稿。"
                ),
                detail=(
                    "文件："
                    + "、".join(PurePosixPath(name).name for name in names[:6])
                    + (f"\n最近一次是我在 {latest.when:%m-%d %H:%M} 写的。" if latest else "")
                    + "\n要的话我把它们并成一份带版本记录的材料，只留一个当前稿。"
                ),
                salience=0.55 + min(0.3, 0.05 * (untracked - 3)),
                source="material-versions",
                data={"artifact": artifact.key, "files": names},
            )
        )
    return observations


def observe_unprocessed_recordings(context: ObserverContext) -> Iterable[Observation]:
    """Notice a recording that came in but never became a set of minutes."""
    attachments = context.data_root / "meet_files" / ATTACHMENTS_DIRNAME
    archives = context.data_root / "meet_files" / ARCHIVE_DIRNAME
    if not attachments.is_dir():
        return []
    handled_stems = _handled_recording_stems(context.data_root, archives)
    observations: list[Observation] = []
    cutoff = context.now - timedelta(days=7)
    for path in sorted(attachments.iterdir()):
        if not path.is_file() or path.suffix.lower() not in TRANSCRIBABLE_SUFFIXES:
            continue
        modified = datetime.fromtimestamp(path.stat().st_mtime).astimezone(context.now.tzinfo)
        if modified < cutoff:
            continue
        # Folder names are editorial titles and often do not resemble the
        # source filename.  The manifest is the durable source/output link.
        if path.stem in handled_stems:
            continue
        observations.append(
            Observation(
                key=f"recording:{path.name}",
                summary=f"录音《{path.name}》还没有整理成纪要。",
                detail="需要的话我现在就可以转写并生成纪要。",
                salience=0.7,
                quiet_hours=24.0,
                source="unprocessed-recording",
                data={"path": str(path)},
            )
        )
    return observations


def observe_inconsistent_entity_spellings(
    context: ObserverContext,
    *,
    alias_groups: dict[str, list[str]] | None = None,
) -> Iterable[Observation]:
    """Notice one name written several ways across the archive.

    ASR turns a company name into two or three homophones, they get written into
    minutes, and every later material inherits the error. Nobody checks because
    each document looks internally consistent.
    """

    groups = alias_groups or {}
    if not groups:
        return []
    archive = context.data_root / "meet_files"
    if not archive.is_dir():
        return []
    corpus: list[str] = []
    for path in archive.rglob("*.md"):
        if any(part in {"execution", "debug_traces", "asr_full"} for part in path.parts):
            continue
        try:
            corpus.append(path.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
        if len(corpus) > 400:
            break
    text = "\n".join(corpus)
    observations: list[Observation] = []
    for canonical, variants in groups.items():
        seen = [variant for variant in variants if variant in text]
        if not seen:
            continue
        observations.append(
            Observation(
                key=f"spelling:{canonical}:{','.join(sorted(seen))}",
                summary=f"归档里“{canonical}”还有 {'、'.join(seen)} 这些旧写法，需要统一核对。",
                detail="需要的话我可以修正已有纪要，并把确认后的名称加入语音识别热词。",
                salience=0.6,
                quiet_hours=72.0,
                source="entity-spelling",
                data={"canonical": canonical, "variants": seen},
            )
        )
    return observations


def observe_ambiguous_version_families(context: ObserverContext) -> Iterable[Observation]:
    """Notice files that *look* like versions of one document but don't read like it.

    The version-folding pass pairs a shared name stem with a content check;
    when the names match but the overlap is too thin to fold, the pair stays
    searchable side by side — which is the right behaviour either way.  What
    the system cannot decide alone is which story is true: a rewrite, or two
    unrelated documents that happen to share a name.  That is a one-question,
    once-and-forever fact the user can settle, and the answer belongs in core
    memory as a decision.
    """

    try:
        from .recall.tools import recall_index_for

        index = recall_index_for(context.data_root)
        with index._connect() as connection:  # noqa: SLF001 - same-package read
            rows = connection.execute(
                "SELECT source_id, uri, family_key, superseded_by, occurred_at"
                " FROM recall_sources WHERE family_key <> '' ORDER BY family_key, occurred_at"
            ).fetchall()
    except Exception:
        return []
    families: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        families[str(row["family_key"])].append(
            {
                "uri": str(row["uri"]),
                "superseded": str(row["superseded_by"] or ""),
            }
        )
    observations: list[Observation] = []
    for key, members in families.items():
        if len(members) < 2:
            continue
        folded = any(member["superseded"] for member in members)
        if folded:
            continue  # 内容已确认版本关系，正常折叠，无需打扰
        names = [PurePosixPath(member["uri"]).name for member in members]
        observations.append(
            Observation(
                key=f"family-ambiguous:{key}",
                summary=(
                    f"《{names[0]}》和《{names[1]}》"
                    + (f" 等 {len(names)} 份" if len(names) > 2 else "")
                    + "名字像同一份材料的先后版本，但内容差异较大。"
                ),
                detail=(
                    "它们是同一份的两次大改（以最新为准），还是两份不同的材料？"
                    "说一声我记下来，以后检索和引用按这个口径处理。"
                ),
                salience=0.5,
                quiet_hours=240.0,
                source="version-family",
                data={"family": key, "files": names},
            )
        )
    return observations[:3]


def build_default_registry(*, alias_groups: dict[str, list[str]] | None = None) -> ObserverRegistry:
    return ObserverRegistry(
        [
            FunctionObserver("material-versions", observe_material_versions),
            FunctionObserver("unprocessed-recording", observe_unprocessed_recordings),
            FunctionObserver(
                "entity-spelling",
                lambda context: observe_inconsistent_entity_spellings(
                    context, alias_groups=alias_groups
                ),
            ),
            FunctionObserver("version-family", observe_ambiguous_version_families),
        ]
    )
