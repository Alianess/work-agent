from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEARCH_ROOTS = (
    WORKSPACE_ROOT / "meet_files" / "asr_full",
    WORKSPACE_ROOT / "meet_files" / "会议项目",
    WORKSPACE_ROOT / "meet_files" / "asr_outputs",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect local Qwen3-ASR chunk progress without starting transcription."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Audio path, ASR output directory, chunk_plan.json, summary.json, or transcript path. Omit to show recent runs.",
    )
    parser.add_argument("--limit", type=int, default=12, help="Maximum runs to print when no path is supplied.")
    args = parser.parse_args()

    runs = collect_runs(args.paths)
    if not runs:
        print("No Qwen3-ASR chunk progress found.")
        print("Search roots:")
        for root in DEFAULT_SEARCH_ROOTS:
            print(f"- {display_path(root)}")
        return 0

    for index, run in enumerate(runs[: max(1, args.limit)], start=1):
        if index > 1:
            print()
        print(render_run(run))
    return 0


def collect_runs(raw_paths: list[str]) -> list[dict[str, Any]]:
    if not raw_paths:
        roots = [root for root in DEFAULT_SEARCH_ROOTS if root.exists()]
        return sort_runs([run for root in roots for run in runs_under(root)])

    results: list[dict[str, Any]] = []
    for raw_path in raw_paths:
        path = resolve_path(raw_path)
        if path.is_file() and path.name in {"chunk_plan.json", "summary.json"}:
            results.append(read_run(path.parent))
            continue
        if path.is_file() and path.name in {"transcript.md", "transcript.txt"}:
            results.append(read_run(path.parent))
            continue
        if path.is_dir():
            direct = nearest_run_dir(path)
            results.extend([read_run(direct)] if direct else runs_under(path))
            continue
        if path.is_file():
            matches = find_runs_for_audio(path)
            results.extend(matches)
            continue
        print(f"Path not found: {raw_path}")

    return dedupe_runs(sort_runs(results))


def runs_under(root: Path) -> list[dict[str, Any]]:
    return [read_run(path.parent) for path in root.rglob("chunk_plan.json")]


def nearest_run_dir(path: Path) -> Path | None:
    current = path
    while True:
        if (current / "chunk_plan.json").is_file():
            return current
        if current == current.parent:
            return None
        current = current.parent


def find_runs_for_audio(audio_path: Path) -> list[dict[str, Any]]:
    audio_resolved = audio_path.resolve()
    audio_stem = audio_path.stem
    candidates: list[dict[str, Any]] = []
    for root in DEFAULT_SEARCH_ROOTS:
        if not root.exists():
            continue
        for run in runs_under(root):
            source = Path(str(run.get("source_audio") or "")).expanduser()
            source_text = str(source)
            if source_text:
                try:
                    if source.resolve() == audio_resolved:
                        candidates.append(run)
                        continue
                except OSError:
                    pass
            if audio_stem and audio_stem in source_text:
                candidates.append(run)
                continue
            if audio_stem and audio_stem in str(run.get("output_root") or ""):
                candidates.append(run)
    return candidates


def read_run(output_root: Path) -> dict[str, Any]:
    plan = read_json(output_root / "chunk_plan.json")
    summary = read_json(output_root / "summary.json")
    chunk_count = int(plan.get("chunk_count") or summary.get("chunk_count") or 0)
    chunks = plan.get("chunks") if isinstance(plan.get("chunks"), list) else []
    planned_indexes = [int(item.get("index") or 0) for item in chunks if isinstance(item, dict)]
    if not chunk_count and planned_indexes:
        chunk_count = max(planned_indexes)
    completed = completed_indexes(output_root)
    missing = [index for index in planned_indexes if index not in completed]
    if not missing and chunk_count:
        missing = [index for index in range(1, chunk_count + 1) if index not in completed]

    source_audio = str(plan.get("source_audio") or summary.get("source_audio") or "")
    updated_at = latest_mtime(output_root)
    return {
        "output_root": output_root,
        "source_audio": source_audio,
        "duration": float(plan.get("duration_seconds") or summary.get("duration_seconds") or 0),
        "chunk_count": chunk_count,
        "completed": sorted(completed),
        "missing": missing,
        "complete": bool(summary.get("complete")) if summary else bool(chunk_count and len(completed) >= chunk_count),
        "transcript_md": output_root / "transcript.md",
        "transcript_txt": output_root / "transcript.txt",
        "summary_path": output_root / "summary.json",
        "chunk_plan_path": output_root / "chunk_plan.json",
        "updated_at": updated_at,
    }


def completed_indexes(output_root: Path) -> set[int]:
    completed: set[int] = set()
    item_dir = output_root / "items"
    if item_dir.is_dir():
        for result in item_dir.glob("chunk_*/transcript.txt"):
            if result.is_file() and result.read_text(encoding="utf-8", errors="replace").strip():
                index = index_from_chunk_dir(result.parent.name)
                if index:
                    completed.add(index)
    progress_jsonl = output_root / "progress.jsonl"
    if progress_jsonl.is_file():
        for line in progress_jsonl.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            index = int(row.get("index") or 0)
            if index:
                completed.add(index)
    summary = read_json(output_root / "summary.json")
    for row in summary.get("chunks") or []:
        if not isinstance(row, dict):
            continue
        index = int(row.get("index") or 0)
        if index:
            completed.add(index)
    return completed


def index_from_chunk_dir(name: str) -> int:
    if not name.startswith("chunk_"):
        return 0
    try:
        return int(name.split("_", 1)[1])
    except ValueError:
        return 0


def render_run(run: dict[str, Any]) -> str:
    chunk_count = int(run["chunk_count"])
    completed = run["completed"]
    missing = run["missing"]
    completed_count = len(completed)
    percent = (completed_count / chunk_count * 100) if chunk_count else 0
    status = "complete" if chunk_count and completed_count >= chunk_count else "incomplete"
    if run.get("complete") and status == "complete":
        status = "complete(summary)"
    lines = [
        f"ASR run: {display_path(Path(run['output_root']))}",
        f"- status: {status}",
        f"- chunks: {completed_count}/{chunk_count} ({percent:.1f}%)",
    ]
    if run.get("duration"):
        lines.append(f"- duration: {format_seconds(float(run['duration']))}")
    if run.get("source_audio"):
        lines.append(f"- source: {display_path(Path(str(run['source_audio'])))}")
    if completed:
        lines.append(f"- completed indexes: {range_summary(completed)}")
    if missing:
        lines.append(f"- missing indexes: {range_summary(missing)}")
    transcript_md = Path(run["transcript_md"])
    transcript_txt = Path(run["transcript_txt"])
    if transcript_md.exists():
        lines.append(f"- transcript.md: {display_path(transcript_md)}")
    if transcript_txt.exists():
        lines.append(f"- transcript.txt: {display_path(transcript_txt)}")
    lines.append(f"- chunk plan: {display_path(Path(run['chunk_plan_path']))}")
    if Path(run["summary_path"]).exists():
        lines.append(f"- summary: {display_path(Path(run['summary_path']))}")
    lines.append("- resume: rerun transcribe_meeting_audio or the Qwen3 command with --skip-existing; existing chunk results will be reused.")
    return "\n".join(lines)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def latest_mtime(path: Path) -> float:
    latest = path.stat().st_mtime if path.exists() else 0
    if path.is_dir():
        for child in path.rglob("*"):
            if child.is_file():
                latest = max(latest, child.stat().st_mtime)
    return latest


def sort_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(runs, key=lambda item: float(item.get("updated_at") or 0), reverse=True)


def dedupe_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[Path] = set()
    result: list[dict[str, Any]] = []
    for run in runs:
        root = Path(run["output_root"]).resolve()
        if root in seen:
            continue
        seen.add(root)
        result.append(run)
    return result


def resolve_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return WORKSPACE_ROOT / path


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(WORKSPACE_ROOT))
    except ValueError:
        return str(path)


def range_summary(indexes: list[int] | set[int]) -> str:
    ordered = sorted(set(indexes))
    if not ordered:
        return "-"
    ranges: list[str] = []
    start = previous = ordered[0]
    for index in ordered[1:]:
        if index == previous + 1:
            previous = index
            continue
        ranges.append(f"{start}" if start == previous else f"{start}-{previous}")
        start = previous = index
    ranges.append(f"{start}" if start == previous else f"{start}-{previous}")
    return ", ".join(ranges)


def format_seconds(seconds: float) -> str:
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


if __name__ == "__main__":
    raise SystemExit(main())
