from __future__ import annotations

from datetime import date, datetime, time as datetime_time, timedelta
from calendar import monthrange
from hashlib import sha256
from pathlib import Path
from typing import Any
import json
import re
import time

from .tools import Tool, ToolRegistry
from .recall_archive import extract_completed_turns
from .memory import estimate_context_tokens


REPORT_TYPES = {"daily", "weekly", "biweekly"}
DEFAULT_AUDIT_INTERVAL_SECONDS = 30 * 60
DEFAULT_DAILY_CUTOFF_HOUR = 18
DEFAULT_LOOKBACK_WORKDAYS = 10
MAX_EVIDENCE_ITEMS = 300
MAX_EVIDENCE_PER_DAY = 25
MAX_REPORT_EVIDENCE_TOKENS = 150_000
OFFICIAL_WORKDAY_CALENDAR_DIR = Path(__file__).resolve().parent / "data" / "workday_calendars"


class WorkReportStore:
    """Account-local evidence ledger and report store.

    Conversation display messages do not consistently carry timestamps. The
    durable activity/turn records do, so report evidence is derived from those
    records and linked back to the visible conversation title and messages.
    """

    def __init__(self, account_root: str | Path) -> None:
        self.account_root = Path(account_root).resolve()
        self.root = self.account_root / "work_reports"

    def collect(
        self,
        *,
        report_type: str,
        target_date: str = "",
        start_date: str = "",
        end_date: str = "",
    ) -> dict[str, Any]:
        period_type, start, end = resolve_report_period(
            report_type=report_type,
            target_date=target_date,
            start_date=start_date,
            end_date=end_date,
        )
        raw_evidence = self._merged_evidence(start, end)
        raw_evidence_estimated_tokens = estimate_context_tokens(
            json.dumps(raw_evidence, ensure_ascii=False, separators=(",", ":"))
        )
        all_evidence = (
            raw_evidence
            if raw_evidence_estimated_tokens <= MAX_REPORT_EVIDENCE_TOKENS
            else balanced_evidence(raw_evidence)
        )

        daily_reports = [
            report
            for day in date_range(start, end)
            if (report := self.load_report("daily", day, day)) is not None
        ]
        workdays = [day.isoformat() for day in date_range(start, end) if self.is_workday(day)]
        covered_days = {
            str(report.get("start_date") or "") for report in daily_reports
        }
        merged = (
            [item for item in all_evidence if str(item.get("date") or "") not in covered_days]
            if period_type in {"weekly", "biweekly"} and covered_days
            else all_evidence
        )
        artifacts = dedupe_strings(
            path
            for item in merged
            for path in item.get("artifacts") or []
            if is_reportable_artifact(path)
        )
        evidence_counts: dict[str, int] = {}
        for item in raw_evidence:
            day = datetime.fromtimestamp(int(item["timestamp"])).date().isoformat()
            evidence_counts[day] = evidence_counts.get(day, 0) + 1

        return {
            "ok": True,
            "report_type": period_type,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "workdays": workdays,
            "daily_reports": daily_reports,
            "missing_daily_reports": [day for day in workdays if day not in covered_days],
            "evidence_counts_by_date": evidence_counts,
            "raw_evidence_count": len(raw_evidence),
            "selected_evidence_count": len(all_evidence),
            "raw_evidence_estimated_tokens": raw_evidence_estimated_tokens,
            "evidence_truncated_for_context": len(all_evidence) < len(raw_evidence),
            "raw_evidence_days": sorted({str(item.get("date") or "") for item in merged}),
            "covered_by_daily_report_days": sorted(covered_days),
            "evidence": merged,
            "artifacts": artifacts,
            "style_references": self._style_references(period_type),
            "calendar": self.calendar_status(start, end),
            "guidance": (
                "For weekly/biweekly reports, saved daily reports replace raw evidence for covered dates. "
                "Raw evidence is returned only for uncovered dates; call the daily period for one date if verification is needed. "
                "A missing date may contain external work; ask the user rather than inventing it."
            ),
        }

    def _style_references(self, report_type: str) -> list[dict[str, Any]]:
        if report_type not in {"weekly", "biweekly"}:
            return []
        reference_dir = self.root / "references"
        if not reference_dir.is_dir():
            return []
        references: list[dict[str, Any]] = []
        for path in sorted(reference_dir.iterdir()):
            if not path.is_file() or path.name.startswith("."):
                continue
            item = {
                "name": path.name,
                "path": relative_to_account(path, self.account_root),
                "kind": "style_profile" if path.suffix.lower() == ".md" else "approved_sample",
            }
            if path.suffix.lower() == ".md":
                item["content"] = clip_text(path.read_text(encoding="utf-8", errors="replace"), 12000)
            references.append(item)
        return references

    def calendar_month(self, *, year: int, month: int) -> dict[str, Any]:
        """Return compact, UI-safe month data without exposing full chat text."""
        normalized_year = int(year)
        normalized_month = int(month)
        if normalized_year < 2000 or normalized_year > 2100:
            raise ValueError("year must be between 2000 and 2100")
        if normalized_month < 1 or normalized_month > 12:
            raise ValueError("month must be between 1 and 12")
        start = date(normalized_year, normalized_month, 1)
        end = date(normalized_year, normalized_month, monthrange(normalized_year, normalized_month)[1])
        evidence = self._merged_evidence(start, end)
        counts: dict[str, int] = {}
        artifact_counts: dict[str, set[str]] = {}
        for item in evidence:
            day = str(item.get("date") or "")
            counts[day] = counts.get(day, 0) + 1
            artifact_counts.setdefault(day, set()).update(
                str(path) for path in item.get("artifacts") or [] if is_reportable_artifact(str(path))
            )
        reports = self._reports_overlapping(start, end, include_content=False)
        report_types_by_day: dict[str, list[str]] = {}
        for report in reports:
            report_start = parse_date(str(report["start_date"]))
            report_end = parse_date(str(report["end_date"]))
            for day in date_range(max(start, report_start), min(end, report_end)):
                report_types_by_day.setdefault(day.isoformat(), []).append(str(report["report_type"]))
        days = []
        for day in date_range(start, end):
            day_text = day.isoformat()
            days.append(
                {
                    "date": day_text,
                    "weekday": day.weekday(),
                    "is_workday": self.is_workday(day),
                    "evidence_count": counts.get(day_text, 0),
                    "artifact_count": len(artifact_counts.get(day_text, set())),
                    "report_types": sorted(set(report_types_by_day.get(day_text, []))),
                }
            )
        return {
            "ok": True,
            "year": normalized_year,
            "month": normalized_month,
            "days": days,
            "reports": reports,
            "calendar": self.calendar_status(start, end),
        }

    def day_detail(self, *, target_date: str) -> dict[str, Any]:
        day = parse_date(target_date)
        evidence = self._merged_evidence(day, day)
        visible_evidence = balanced_evidence(evidence)
        return {
            "ok": True,
            "date": day.isoformat(),
            "is_workday": self.is_workday(day),
            "evidence_count": len(evidence),
            "evidence_truncated": len(visible_evidence) < len(evidence),
            "evidence": visible_evidence,
            "daily_report": self.load_report("daily", day, day),
            "covering_reports": self._reports_overlapping(day, day, include_content=True),
            "calendar": self.calendar_status(day, day),
        }

    def save_report(
        self,
        *,
        report_type: str,
        content: str,
        target_date: str = "",
        start_date: str = "",
        end_date: str = "",
        source_coverage: str = "partial",
        needs_user_input: bool = False,
    ) -> dict[str, Any]:
        period_type, start, end = resolve_report_period(
            report_type=report_type,
            target_date=target_date,
            start_date=start_date,
            end_date=end_date,
        )
        text = str(content or "").strip()
        if not text:
            raise ValueError("content is required")
        report_dir = self.root / period_type
        report_dir.mkdir(parents=True, exist_ok=True)
        stem = report_stem(period_type, start, end)
        markdown_path = report_dir / f"{stem}.md"
        metadata_path = report_dir / f"{stem}.json"
        saved_content = text.rstrip() + "\n"
        write_text_atomically(markdown_path, saved_content)
        persisted_content = markdown_path.read_text(encoding="utf-8")
        if persisted_content != saved_content:
            raise RuntimeError("日报文件写入后校验不一致，未确认保存成功。")
        now = int(time.time())
        metadata = {
            "schema_version": 1,
            "report_type": period_type,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "source_coverage": normalize_coverage(source_coverage),
            "needs_user_input": bool(needs_user_input),
            "content_path": relative_to_account(markdown_path, self.account_root),
            "content_sha256": sha256(saved_content.encode("utf-8")).hexdigest(),
            "content_bytes": len(saved_content.encode("utf-8")),
            "verified": True,
            "updated_at": now,
        }
        existing = read_json(metadata_path)
        metadata["created_at"] = int(existing.get("created_at") or now)
        write_json(metadata_path, metadata)
        return {"ok": True, **metadata}

    def read_report(
        self,
        *,
        report_type: str,
        target_date: str = "",
        start_date: str = "",
        end_date: str = "",
    ) -> dict[str, Any]:
        period_type, start, end = resolve_report_period(
            report_type=report_type,
            target_date=target_date,
            start_date=start_date,
            end_date=end_date,
        )
        report = self.load_report(period_type, start, end)
        if report is None:
            raise FileNotFoundError(f"尚未保存 {start.isoformat()} 至 {end.isoformat()} 的{period_type}工作汇报。")
        content = str(report.get("content") or "")
        if not content:
            raise RuntimeError("工作汇报元数据存在，但 Markdown 正文不可读取。")
        expected_hash = str(report.get("content_sha256") or "")
        actual_hash = sha256(content.encode("utf-8")).hexdigest()
        if expected_hash and expected_hash != actual_hash:
            raise RuntimeError("工作汇报正文校验失败，文件可能已被外部修改。")
        return {
            "ok": True,
            **report,
            "verified": True,
            "content_sha256": actual_hash,
            "content_bytes": len(content.encode("utf-8")),
        }

    def load_report(self, report_type: str, start: date, end: date) -> dict[str, Any] | None:
        metadata_path = self.root / report_type / f"{report_stem(report_type, start, end)}.json"
        if not metadata_path.is_file():
            return None
        metadata = read_json(metadata_path)
        content_path = self.account_root / str(metadata.get("content_path") or "")
        content = content_path.read_text(encoding="utf-8", errors="replace") if content_path.is_file() else ""
        return {**metadata, "content": content}

    def status(self, *, now: datetime | None = None, lookback_workdays: int = DEFAULT_LOOKBACK_WORKDAYS) -> dict[str, Any]:
        current = now or datetime.now()
        due_dates = self._recent_due_workdays(current, lookback_workdays)
        missing = [day for day in due_dates if self.load_report("daily", day, day) is None]
        evidence_counts: dict[str, int] = {day.isoformat(): 0 for day in missing}
        if due_dates:
            period = self.collect(
                report_type="biweekly",
                start_date=due_dates[0].isoformat(),
                end_date=due_dates[-1].isoformat(),
            )
            for day in missing:
                items = [item for item in period["evidence"] if item.get("date") == day.isoformat()]
                ledger = self._write_daily_ledger(day, items)
                evidence_counts[day.isoformat()] = int(ledger.get("evidence_count") or 0)
        return {
            "ok": True,
            "checked_at": int(current.timestamp()),
            "missing_daily_reports": [day.isoformat() for day in missing],
            "evidence_counts_by_date": evidence_counts,
            "calendar": self.calendar_status(due_dates[0] if due_dates else current.date(), current.date()),
        }

    def audit_if_due(
        self,
        *,
        now: datetime | None = None,
        interval_seconds: int = DEFAULT_AUDIT_INTERVAL_SECONDS,
    ) -> dict[str, Any]:
        current = now or datetime.now()
        state_path = self.root / "audit_state.json"
        state = read_json(state_path)
        last_checked = int(state.get("last_checked_at") or 0)
        if int(current.timestamp()) - last_checked < max(60, int(interval_seconds)):
            return {"ok": True, "skipped": True, **state}

        status = self.status(now=current)
        missing = status["missing_daily_reports"]
        signature = sha256("\n".join(missing).encode("utf-8")).hexdigest() if missing else ""
        last_notified = int(state.get("last_notified_at") or 0)
        notify = bool(missing) and (
            signature != str(state.get("missing_signature") or "")
            or int(current.timestamp()) - last_notified >= 24 * 60 * 60
        )
        next_state = {
            "last_checked_at": int(current.timestamp()),
            "missing_signature": signature,
            "last_notified_at": int(current.timestamp()) if notify else last_notified,
            "missing_daily_reports": missing,
        }
        self.root.mkdir(parents=True, exist_ok=True)
        write_json(state_path, next_state)
        return {"ok": True, "skipped": False, "notify": notify, **status, **next_state}

    def refresh_daily_ledger(self, day: date) -> dict[str, Any]:
        evidence = self.collect(report_type="daily", target_date=day.isoformat())
        return self._write_daily_ledger(day, evidence["evidence"])

    def _write_daily_ledger(self, day: date, items: list[dict[str, Any]]) -> dict[str, Any]:
        artifacts = dedupe_strings(
            path for item in items for path in item.get("artifacts") or [] if is_reportable_artifact(path)
        )
        payload = {
            "schema_version": 1,
            "date": day.isoformat(),
            "is_workday": self.is_workday(day),
            "evidence_count": len(items),
            "artifact_count": len(artifacts),
            "evidence": items,
            "artifacts": artifacts,
            "updated_at": int(time.time()),
        }
        ledger_path = self.root / "ledgers" / f"{day.isoformat()}.json"
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(ledger_path, payload)
        return payload

    def is_workday(self, day: date) -> bool:
        overrides = self._calendar_overrides()
        explicit = overrides.get(day.isoformat())
        if isinstance(explicit, bool):
            return explicit
        return day.weekday() < 5

    def calendar_status(self, start: date, end: date) -> dict[str, Any]:
        calendar_payload = self._calendar_metadata_for_year(start.year)
        overrides = self._calendar_overrides()
        covered_years = sorted({int(key[:4]) for key in overrides if re.fullmatch(r"\d{4}-\d{2}-\d{2}", key)})
        requested_years = list(range(start.year, end.year + 1))
        return {
            "timezone": "local",
            "rule": "explicit China workday overrides, otherwise Monday-Friday fallback",
            "override_years": covered_years,
            "years_using_weekday_fallback": [year for year in requested_years if year not in covered_years],
            "override_path": "work_reports/calendar_overrides.json",
            "source": str(calendar_payload.get("source") or ""),
            "document_title": str(calendar_payload.get("document_title") or ""),
            "document_number": str(calendar_payload.get("document_number") or ""),
        }

    def update_calendar(self, *, source: str, days: dict[str, Any]) -> dict[str, Any]:
        source_text = str(source or "").strip()
        if not source_text:
            raise ValueError("source is required")
        if not isinstance(days, dict) or not days:
            raise ValueError("days must be a non-empty YYYY-MM-DD to boolean mapping")
        normalized: dict[str, bool] = {}
        for raw_day, raw_value in days.items():
            day_text = str(raw_day or "").strip()
            parse_date(day_text)
            if not isinstance(raw_value, bool):
                raise ValueError(f"calendar value for {day_text} must be boolean")
            normalized[day_text] = raw_value
        path = self.root / "calendar_overrides.json"
        existing = read_json(path)
        merged = existing.get("days") if isinstance(existing.get("days"), dict) else {}
        merged = {**merged, **normalized}
        payload = {
            "schema_version": 1,
            "source": source_text,
            "days": dict(sorted(merged.items())),
            "updated_at": int(time.time()),
        }
        write_json(path, payload)
        return {
            "ok": True,
            "path": relative_to_account(path, self.account_root),
            "updated_count": len(normalized),
            "total_count": len(merged),
            "years": sorted({int(day[:4]) for day in merged}),
        }

    def _calendar_overrides(self) -> dict[str, bool]:
        merged: dict[str, bool] = {}
        if OFFICIAL_WORKDAY_CALENDAR_DIR.is_dir():
            for path in sorted(OFFICIAL_WORKDAY_CALENDAR_DIR.glob("*.json")):
                merged.update(calendar_days(read_json(path)))
        merged.update(calendar_days(read_json(self.root / "calendar_overrides.json")))
        return merged

    def _calendar_metadata_for_year(self, year: int) -> dict[str, Any]:
        local = read_json(self.root / "calendar_overrides.json")
        if any(str(day).startswith(f"{year:04d}-") for day in calendar_days(local)):
            return local
        return read_json(OFFICIAL_WORKDAY_CALENDAR_DIR / f"{year:04d}.json")

    def _recent_due_workdays(self, current: datetime, limit: int) -> list[date]:
        include_today = current.time() >= datetime_time(DEFAULT_DAILY_CUTOFF_HOUR)
        cursor = current.date() if include_today else current.date() - timedelta(days=1)
        days: list[date] = []
        while len(days) < max(1, min(int(limit), 60)):
            if self.is_workday(cursor):
                days.append(cursor)
            cursor -= timedelta(days=1)
        return list(reversed(days))

    def _conversation_evidence(
        self,
        start: date,
        end: date,
    ) -> tuple[list[dict[str, Any]], dict[str, str]]:
        items = load_conversation_archive_items(
            self.account_root / "conversation_history" / "conversations.json"
        )
        results: list[dict[str, Any]] = []
        title_by_id: dict[str, str] = {}
        start_ts, end_ts = date_bounds(start, end)
        for conversation in items:
            if not isinstance(conversation, dict):
                continue
            conversation_id = str(conversation.get("id") or "")
            title = str(conversation.get("title") or conversation_id or "未命名聊天")
            title_by_id[conversation_id] = title
            messages = conversation.get("messages") if isinstance(conversation.get("messages"), list) else []
            activities = conversation.get("activities") if isinstance(conversation.get("activities"), dict) else {}
            for raw_index, record in activities.items():
                if not isinstance(record, dict):
                    continue
                events = [item for item in record.get("events") or [] if isinstance(item, dict)]
                timestamps = [int(item.get("ts_ms") or 0) // 1000 for item in events if item.get("ts_ms")]
                timestamp = min(timestamps) if timestamps else 0
                if timestamp < start_ts or timestamp > end_ts:
                    continue
                index = safe_int(raw_index, -1)
                user_text = ""
                final_text = ""
                if 0 <= index - 1 < len(messages) and messages[index - 1].get("role") == "user":
                    user_text = str(messages[index - 1].get("content") or "").strip()
                if 0 <= index < len(messages) and messages[index].get("role") == "assistant":
                    final_text = str(messages[index].get("content") or "").strip()
                public_path = dedupe_strings(
                    str(event.get("detail") or event.get("content") or "")
                    for event in events
                    if event.get("activity_type") == "work_note"
                )
                artifacts = dedupe_strings(
                    str(event.get("file_path") or "")
                    for event in events
                    if event.get("file_path") and event.get("command_status") != "error"
                )
                turn_id = next((str(event.get("turn_id") or "") for event in events if event.get("turn_id")), "")
                results.append(
                    {
                        "timestamp": timestamp,
                        "date": datetime.fromtimestamp(timestamp).date().isoformat(),
                        "conversation_id": conversation_id,
                        "conversation_title": title,
                        "project_id": str(conversation.get("projectId") or ""),
                        "turn_id": turn_id,
                        "user_request": user_text,
                        "public_path": public_path,
                        "result": final_text,
                        "artifacts": artifacts,
                        "source": "conversation_activity",
                    }
                )
        return results, title_by_id

    def _merged_evidence(self, start: date, end: date) -> list[dict[str, Any]]:
        conversations, title_by_id = self._conversation_evidence(start, end)
        turns = self._turn_evidence(start, end, title_by_id)
        known_turn_ids = {str(item.get("turn_id") or "") for item in turns if item.get("turn_id")}
        merged = [
            *turns,
            *(item for item in conversations if item.get("turn_id") not in known_turn_ids),
        ]
        merged.sort(key=lambda item: int(item.get("timestamp") or 0))
        return merged

    def _reports_overlapping(
        self,
        start: date,
        end: date,
        *,
        include_content: bool,
    ) -> list[dict[str, Any]]:
        reports: list[dict[str, Any]] = []
        for report_type in ("daily", "weekly", "biweekly"):
            report_dir = self.root / report_type
            if not report_dir.is_dir():
                continue
            for metadata_path in sorted(report_dir.glob("*.json")):
                metadata = read_json(metadata_path)
                try:
                    report_start = parse_date(str(metadata.get("start_date") or ""))
                    report_end = parse_date(str(metadata.get("end_date") or ""))
                except ValueError:
                    continue
                if report_end < start or report_start > end:
                    continue
                item = {**metadata, "report_type": report_type}
                if include_content:
                    content_path = self.account_root / str(metadata.get("content_path") or "")
                    item["content"] = (
                        content_path.read_text(encoding="utf-8", errors="replace")
                        if content_path.is_file()
                        else ""
                    )
                reports.append(item)
        reports.sort(key=lambda item: (str(item.get("start_date") or ""), str(item.get("report_type") or "")))
        return reports

    def _turn_evidence(
        self,
        start: date,
        end: date,
        title_by_id: dict[str, str],
    ) -> list[dict[str, Any]]:
        turn_dir = self.account_root / "conversation_history" / "turns"
        start_ts, end_ts = date_bounds(start, end)
        results: list[dict[str, Any]] = []
        if not turn_dir.is_dir():
            return results
        payloads: list[tuple[Path, dict[str, Any]]] = []
        for path in turn_dir.glob("turn-*.json"):
            payload = read_json(path)
            timestamp = int(payload.get("created_at") or 0)
            if timestamp < start_ts or timestamp > end_ts:
                continue
            final_text = str(payload.get("final_message") or "").strip()
            if not is_reportable_turn(str(payload.get("status") or ""), final_text):
                continue
            payloads.append((path, payload))
        payloads.sort(key=lambda item: int(item[1].get("created_at") or 0))
        projections: dict[str, list[dict[str, Any]]] = {}
        for path, payload in payloads:
            timestamp = int(payload.get("created_at") or 0)
            final_text = str(payload.get("final_message") or "").strip()
            events = [item for item in payload.get("events") or [] if isinstance(item, dict)]
            artifacts = dedupe_strings(
                str(event.get("file_path") or "")
                for event in events
                if event.get("file_path") and event.get("command_status") != "error"
            )
            conversation_id = str(payload.get("conversation_id") or "")
            if conversation_id not in projections:
                projections[conversation_id] = self._session_turn_projections(conversation_id)
            projection = match_turn_projection(
                projections[conversation_id],
                final_text,
            )
            results.append(
                {
                    "timestamp": timestamp,
                    "date": datetime.fromtimestamp(timestamp).date().isoformat(),
                    "conversation_id": conversation_id,
                    "conversation_title": title_by_id.get(conversation_id, conversation_id or "未命名聊天"),
                    "project_id": str((payload.get("metadata") or {}).get("project_id") or ""),
                    "turn_id": str(payload.get("id") or path.stem),
                    "user_request": str(projection.get("user_request") or ""),
                    "public_path": list(projection.get("public_path") or []),
                    "result": final_text,
                    "artifacts": artifacts,
                    "status": str(payload.get("status") or ""),
                    "source": "recall_archive_projection" if projection else "turn_runtime",
                }
            )
        return results

    def _session_turn_projections(self, conversation_id: str) -> list[dict[str, Any]]:
        if not conversation_id:
            return []
        session_path = self.account_root / "conversation_history" / "sessions" / f"{conversation_id}.json"
        payload = read_json(session_path)
        messages = [item for item in payload.get("messages") or [] if isinstance(item, dict)]
        projections: list[dict[str, Any]] = []
        for turn in extract_completed_turns(messages):
            final_text = "\n\n".join(str(item) for item in turn.get("final_texts") or [] if str(item).strip()).strip()
            if not final_text:
                continue
            projections.append(
                {
                    "user_request": "\n\n".join(
                        str(item) for item in turn.get("user_texts") or [] if str(item).strip()
                    ).strip(),
                    "public_path": list(turn.get("path_texts") or []),
                    "result": final_text,
                }
            )
        return projections


def resolve_report_period(
    *,
    report_type: str,
    target_date: str = "",
    start_date: str = "",
    end_date: str = "",
) -> tuple[str, date, date]:
    period_type = str(report_type or "daily").strip().lower()
    if period_type not in REPORT_TYPES:
        raise ValueError("report_type must be daily, weekly, or biweekly")
    if start_date or end_date:
        end = parse_date(end_date or start_date)
        start = parse_date(start_date or end_date)
    else:
        end = parse_date(target_date) if target_date else date.today()
        days = 1 if period_type == "daily" else 7 if period_type == "weekly" else 14
        start = end - timedelta(days=days - 1)
    if period_type == "daily":
        start = end
    if start > end:
        start, end = end, start
    if (end - start).days > 62:
        raise ValueError("report period cannot exceed 63 days")
    return period_type, start, end


def report_stem(report_type: str, start: date, end: date) -> str:
    return start.isoformat() if report_type == "daily" else f"{start.isoformat()}_{end.isoformat()}"


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(str(value or "").strip())
    except ValueError as error:
        raise ValueError("date must use YYYY-MM-DD") from error


def date_range(start: date, end: date):
    cursor = start
    while cursor <= end:
        yield cursor
        cursor += timedelta(days=1)


def date_bounds(start: date, end: date) -> tuple[int, int]:
    start_dt = datetime.combine(start, datetime_time.min)
    end_dt = datetime.combine(end, datetime_time.max)
    return int(start_dt.timestamp()), int(end_dt.timestamp())


def normalize_coverage(value: str) -> str:
    text = str(value or "partial").strip().lower()
    return text if text in {"full", "partial", "external_gap"} else "partial"


def relative_to_account(path: Path, account_root: Path) -> str:
    return str(path.resolve().relative_to(account_root.resolve()))


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def load_conversation_archive_items(path: Path) -> list[dict[str, Any]]:
    """Read both legacy archive files and v2 per-conversation archive items."""
    payload = read_json(path)
    raw_items = payload.get("items") if isinstance(payload.get("items"), list) else []
    if payload.get("storage") != "per_item":
        return [dict(item) for item in raw_items if isinstance(item, dict)]
    raw_order = payload.get("order") if isinstance(payload.get("order"), list) else []
    order = [str(value or "").strip() for value in raw_order]
    if not order:
        order = [str(item.get("id") or "").strip() for item in raw_items if isinstance(item, dict)]
    items: list[dict[str, Any]] = []
    for conversation_id in order:
        if not conversation_id:
            continue
        item = read_json(path.parent / "archive_items" / f"{conversation_id}.json")
        if item and str(item.get("id") or "").strip() == conversation_id:
            items.append(item)
    return items


def calendar_days(payload: dict[str, Any]) -> dict[str, bool]:
    raw = payload.get("days") if isinstance(payload.get("days"), dict) else payload
    if not isinstance(raw, dict):
        return {}
    return {
        str(key): bool(value)
        for key, value in raw.items()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(key)) and isinstance(value, bool)
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_text_atomically(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def clip_text(value: Any, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def dedupe_strings(values) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def is_reportable_artifact(path: str) -> bool:
    normalized = str(path or "").replace("\\", "/")
    blocked = ("conversation_history/", "debug_traces/", "/.work_agent_tmp/", "work_reports/ledgers/")
    return bool(normalized) and not any(marker in normalized for marker in blocked)


def balanced_evidence(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Bound prompt size without erasing the first half of a reporting period."""
    by_date: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        by_date.setdefault(str(item.get("date") or "unknown"), []).append(item)
    selected = [
        item
        for day in sorted(by_date)
        for item in by_date[day][-MAX_EVIDENCE_PER_DAY:]
    ]
    if len(selected) <= MAX_EVIDENCE_ITEMS:
        return selected
    # A long explicit range can still exceed the global ceiling. Keep an equal
    # recent slice from every represented date instead of dropping early dates.
    per_day = max(1, MAX_EVIDENCE_ITEMS // max(1, len(by_date)))
    return [
        item
        for day in sorted(by_date)
        for item in by_date[day][-per_day:]
    ][:MAX_EVIDENCE_ITEMS]


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def match_turn_projection(
    projections: list[dict[str, Any]],
    final_text: str,
) -> dict[str, Any]:
    target = normalize_match_text(final_text)
    if not target:
        return {}
    for index, projection in enumerate(projections):
        if normalize_match_text(str(projection.get("result") or "")) == target:
            return projections.pop(index)
    return {}


def normalize_match_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def is_reportable_turn(status: str, final_text: str) -> bool:
    text = str(final_text or "").strip()
    if not text:
        return False
    normalized_status = str(status or "").strip().lower()
    if normalized_status == "succeeded":
        return True
    if normalized_status != "failed":
        return False
    compact = text.casefold()
    mechanical_failures = (
        "模型流式响应中断",
        "llm stream failed",
        "network error",
        "unexpected_eof_while_reading",
        "模型规划失败",
        "reached max react steps",
    )
    return not any(marker in compact for marker in mechanical_failures)


def register_work_report_tools(registry: ToolRegistry, account_root: str | Path) -> None:
    store = WorkReportStore(account_root)

    registry.register(
        Tool(
            name="collect_work_report_evidence",
            description=(
                "Collect account-local, timestamped work evidence for a daily, weekly, or biweekly report. "
                "Returns prior daily reports, missing workdays, full user requests, public implementation-path notes, "
                "full final answers, and successfully edited artifact paths. Detailed tool payloads and observations "
                "are folded using the same deterministic recall-archive projection as context compaction. "
                "Use this instead of scanning every conversation or guessing from a compressed summary."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "report_type": {"type": "string", "enum": ["daily", "weekly", "biweekly"]},
                    "target_date": {"type": "string", "description": "YYYY-MM-DD; defaults to today."},
                    "start_date": {"type": "string", "description": "Optional explicit YYYY-MM-DD start."},
                    "end_date": {"type": "string", "description": "Optional explicit YYYY-MM-DD end."},
                },
                "required": ["report_type"],
            },
            handler=lambda args: json.dumps(store.collect(**args), ensure_ascii=False, indent=2),
        )
    )
    registry.register(
        Tool(
            name="save_work_report",
            description=(
                "Save a completed daily, weekly, or biweekly Markdown report with structured local metadata. "
                "Set needs_user_input=true when external/offline work is still missing."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "report_type": {"type": "string", "enum": ["daily", "weekly", "biweekly"]},
                    "content": {"type": "string"},
                    "target_date": {"type": "string"},
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                    "source_coverage": {
                        "type": "string",
                        "enum": ["full", "partial", "external_gap"],
                        "default": "partial",
                    },
                    "needs_user_input": {"type": "boolean", "default": False},
                },
                "required": ["report_type", "content"],
            },
            handler=lambda args: json.dumps(store.save_report(**args), ensure_ascii=False, indent=2),
        )
    )
    registry.register(
        Tool(
            name="read_saved_work_report",
            description=(
                "Read and verify a saved work report from the account-local report store. "
                "Use this instead of read_text_file: work_reports are intentionally separate from the general file workspace."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "report_type": {"type": "string", "enum": ["daily", "weekly", "biweekly"]},
                    "target_date": {"type": "string"},
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                },
                "required": ["report_type"],
            },
            handler=lambda args: json.dumps(store.read_report(**args), ensure_ascii=False, indent=2),
        )
    )
    registry.register(
        Tool(
            name="check_work_report_status",
            description=(
                "Check recent workdays for missing daily reports and return how much local evidence exists for each gap."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "lookback_workdays": {"type": "integer", "default": DEFAULT_LOOKBACK_WORKDAYS},
                },
            },
            handler=lambda args: json.dumps(
                store.status(lookback_workdays=int(args.get("lookback_workdays") or DEFAULT_LOOKBACK_WORKDAYS)),
                ensure_ascii=False,
                indent=2,
            ),
        )
    )
    registry.register(
        Tool(
            name="update_workday_calendar",
            description=(
                "Merge verified China statutory holiday/workday overrides into the account-local calendar. "
                "Use false for holidays and true for adjusted weekend workdays; preserve the official source URL."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Official annual holiday notice URL or citation."},
                    "days": {
                        "type": "object",
                        "description": "YYYY-MM-DD keys mapped to boolean workday values.",
                        "additionalProperties": {"type": "boolean"},
                    },
                },
                "required": ["source", "days"],
            },
            handler=lambda args: json.dumps(store.update_calendar(**args), ensure_ascii=False, indent=2),
        )
    )
