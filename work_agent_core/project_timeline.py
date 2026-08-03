from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any
import json
import re
import shutil
import time

from openpyxl import load_workbook

from .timeline_xlsx import apply_timeline_changes, inspect_timeline_workbook


PROJECT_ID_PATTERN = re.compile(r"^project-[a-f0-9]{12}$")
TIMELINE_EXTENSIONS = {".xlsx", ".xlsm", ".xltx", ".xltm"}
TIMELINE_NAME_MARKERS = ("项目推进", "推进计划", "时间线", "时间节点", "关键节点", "里程碑", "排期")
COMPLETED_STATUSES = {"已完成", "完成", "已办结", "办结", "done", "completed"}
CANCELLED_STATUSES = {"已取消", "取消", "不再推进", "cancelled", "canceled"}
IN_PROGRESS_STATUSES = {"推进中", "进行中", "处理中", "执行中", "in progress", "doing"}
RISK_STATUSES = {"有风险", "风险", "阻塞", "已阻塞", "延期", "逾期", "blocked", "at risk"}
DEFAULT_TIMELINE_TEMPLATE = Path(__file__).with_name("templates") / "project_timeline.xlsx"


def _safe_project_name(value: Any) -> str:
    name = re.sub(r"\s+", " ", str(value or "项目")).strip()
    name = re.sub(r'[/\\:*?"<>|]+', "_", name)
    return name[:60] or "项目"


def _project_manifest_path(project_directory: Path) -> Path:
    return project_directory / "project.json"


def _read_manifest(project_directory: Path) -> dict[str, Any]:
    path = _project_manifest_path(project_directory)
    if not path.is_file():
        raise ValueError("项目配置不存在。")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("项目配置损坏。")
    return payload


def _write_manifest(project_directory: Path, payload: dict[str, Any]) -> None:
    path = _project_manifest_path(project_directory)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _resolve_manifest_timeline(project_directory: Path, manifest: dict[str, Any]) -> Path | None:
    relative = str(manifest.get("timeline_path") or "").strip()
    if not relative:
        return None
    candidate = (project_directory / relative).resolve()
    try:
        candidate.relative_to(project_directory.resolve())
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def find_project_timeline(
    project_directory: str | Path,
    *,
    manifest: dict[str, Any] | None = None,
) -> Path | None:
    directory = Path(project_directory).resolve()
    data = manifest or _read_manifest(directory)
    selected = _resolve_manifest_timeline(directory, data)
    if selected:
        return selected

    sources = directory / "sources"
    if not sources.is_dir():
        return None
    candidates = [
        path
        for path in sources.rglob("*")
        if path.is_file()
        and path.suffix.lower() in TIMELINE_EXTENSIONS
        and ".backup-" not in path.name.lower()
    ]
    candidates.sort(
        key=lambda path: (
            "项目推进" in path.parts,
            sum(marker in path.name for marker in TIMELINE_NAME_MARKERS),
            path.stat().st_mtime_ns,
        ),
        reverse=True,
    )
    for candidate in candidates:
        try:
            inspect_timeline_workbook(candidate, limit=1)
        except (OSError, ValueError):
            continue
        return candidate
    return None


def select_project_timeline(project_directory: str | Path, timeline_path: str | Path) -> Path:
    directory = Path(project_directory).resolve()
    candidate = Path(timeline_path).resolve()
    sources = (directory / "sources").resolve()
    try:
        candidate.relative_to(sources)
    except ValueError as error:
        raise ValueError("时间线文件必须位于当前项目资料目录。") from error
    inspect_timeline_workbook(candidate, limit=1)
    manifest = _read_manifest(directory)
    manifest["timeline_path"] = str(candidate.relative_to(directory))
    manifest["updated_at"] = int(time.time())
    _write_manifest(directory, manifest)
    return candidate


def create_project_timeline(
    project_directory: str | Path,
    *,
    project_name: str,
    template_path: str | Path = DEFAULT_TIMELINE_TEMPLATE,
) -> Path:
    directory = Path(project_directory).resolve()
    existing = find_project_timeline(directory)
    if existing:
        return existing
    template = Path(template_path).resolve()
    if not template.is_file():
        raise FileNotFoundError(f"项目时间线模板不存在：{template}")
    target_directory = directory / "sources" / "项目推进"
    target_directory.mkdir(parents=True, exist_ok=True)
    target = target_directory / f"{_safe_project_name(project_name)}项目推进.xlsx"
    shutil.copy2(template, target)
    workbook = load_workbook(target)
    try:
        sheet = workbook["项目推进"]
        sheet["A1"] = f"{project_name}关键节点推进表"
        workbook.save(target)
    finally:
        workbook.close()
    select_project_timeline(directory, target)
    return target


def _reference_year(row: dict[str, Any], today: date) -> int:
    for field in ("updated_at", "actual_date"):
        raw = row.get(field)
        if isinstance(raw, (date, datetime)):
            return raw.year
        match = re.match(r"^\s*(20\d{2})[-/.年]", str(raw or ""))
        if match:
            return int(match.group(1))
    return today.year


def parse_timeline_date(value: Any, *, reference_year: int) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        pass
    for pattern in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y年%m月%d日"):
        try:
            return datetime.strptime(text, pattern).date()
        except ValueError:
            pass
    match = re.fullmatch(r"(\d{1,2})\s*月\s*(\d{1,2})\s*日?", text)
    if match:
        try:
            return date(reference_year, int(match.group(1)), int(match.group(2)))
        except ValueError:
            return None
    return None


def _normalized_status(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "未开始")).strip() or "未开始"


def _schedule_state(status: str, planned: date | None, today: date) -> tuple[str, int | None]:
    normalized = status.lower()
    if normalized in COMPLETED_STATUSES:
        return "completed", None if planned is None else (planned - today).days
    if normalized in CANCELLED_STATUSES:
        return "cancelled", None if planned is None else (planned - today).days
    days_until = None if planned is None else (planned - today).days
    if normalized in RISK_STATUSES:
        return "risk", days_until
    if days_until is None:
        return "unscheduled", None
    if days_until < 0:
        return "overdue", days_until
    if days_until <= 7:
        return "due_soon", days_until
    return "upcoming", days_until


def project_timeline_payload(
    project_directory: str | Path,
    *,
    today: date | None = None,
) -> dict[str, Any]:
    directory = Path(project_directory).resolve()
    current_day = today or datetime.now().astimezone().date()
    path = find_project_timeline(directory)
    empty_summary = {
        "total": 0,
        "completed": 0,
        "in_progress": 0,
        "risk": 0,
        "overdue": 0,
        "scheduled": 0,
        "unscheduled": 0,
        "completion_rate": 0,
    }
    if not path:
        return {
            "exists": False,
            "path": None,
            "name": None,
            "sheet_name": None,
            "header_row": None,
            "modified": None,
            "nodes": [],
            "summary": empty_summary,
            "error": None,
        }
    try:
        inspected = inspect_timeline_workbook(path, limit=2000)
    except (OSError, ValueError) as error:
        return {
            "exists": True,
            "path": str(path),
            "name": path.name,
            "sheet_name": None,
            "header_row": None,
            "modified": int(path.stat().st_mtime),
            "nodes": [],
            "summary": empty_summary,
            "error": str(error),
        }

    nodes: list[dict[str, Any]] = []
    summary = dict(empty_summary)
    for raw in inspected["rows"]:
        status = _normalized_status(raw.get("status"))
        reference_year = _reference_year(raw, current_day)
        planned = parse_timeline_date(raw.get("planned_date"), reference_year=reference_year)
        actual = parse_timeline_date(raw.get("actual_date"), reference_year=reference_year)
        schedule_state, days_until = _schedule_state(status, planned, current_day)
        item = {
            **raw,
            "status": status,
            "planned_date": planned.isoformat() if planned else (str(raw.get("planned_date") or "") or None),
            "actual_date": actual.isoformat() if actual else (str(raw.get("actual_date") or "") or None),
            "schedule_state": schedule_state,
            "days_until": days_until,
        }
        nodes.append(item)
        summary["total"] += 1
        if schedule_state == "completed":
            summary["completed"] += 1
        elif schedule_state == "risk":
            summary["risk"] += 1
        elif schedule_state == "overdue":
            summary["overdue"] += 1
        if status.lower() in IN_PROGRESS_STATUSES:
            summary["in_progress"] += 1
        if planned:
            summary["scheduled"] += 1
        else:
            summary["unscheduled"] += 1
    summary["completion_rate"] = (
        round(summary["completed"] / summary["total"] * 100)
        if summary["total"]
        else 0
    )
    nodes.sort(
        key=lambda item: (
            item.get("planned_date") is None,
            str(item.get("planned_date") or "9999-12-31"),
            int(item.get("row") or 0),
        )
    )
    return {
        "exists": True,
        "path": str(path),
        "name": path.name,
        "sheet_name": inspected["layout"]["sheet_name"],
        "header_row": inspected["layout"]["header_row"],
        "modified": int(path.stat().st_mtime),
        "nodes": nodes,
        "summary": summary,
        "error": None,
    }


def apply_project_timeline_changes(
    project_directory: str | Path,
    *,
    changes: list[dict[str, Any]],
    change_source: str = "Friday",
) -> dict[str, Any]:
    directory = Path(project_directory).resolve()
    path = find_project_timeline(directory)
    if not path:
        raise ValueError("项目尚未创建或选择时间线 Excel。")
    history_directory = directory / "history" / "timeline"
    history_directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    backup_path = history_directory / f"{path.stem}.{timestamp}{path.suffix}"
    shutil.copy2(path, backup_path)
    try:
        result = apply_timeline_changes(
            path,
            changes=changes,
            dry_run=False,
            backup=False,
            record_history=True,
            create_missing_columns=True,
            change_source=change_source,
        )
    except Exception:
        shutil.copy2(backup_path, path)
        raise
    manifest = _read_manifest(directory)
    manifest["timeline_path"] = str(path.relative_to(directory))
    manifest["updated_at"] = int(time.time())
    _write_manifest(directory, manifest)
    result["history_path"] = str(backup_path)
    return result
