from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import subprocess

from .runtime_env import find_runtime_executable


AUDIO_EXTENSIONS = {
    ".m4a",
    ".mp3",
    ".wav",
    ".aac",
    ".flac",
    ".ogg",
    ".opus",
    ".wma",
    ".amr",
    ".aiff",
    ".aif",
    ".caf",
    ".webm",
    ".mp4",
}

CREATION_TIME_TAGS = (
    "creation_time",
    "com.apple.quicktime.creationdate",
)


def probe_audio_metadata(path: str | Path, *, timeout_seconds: int = 15) -> dict[str, Any]:
    """Read media timestamps and expose a recording start only when the file timeline is plausible."""
    audio_path = Path(path)
    if not audio_path.is_file() or audio_path.suffix.lower() not in AUDIO_EXTENSIONS:
        return {}

    ffprobe = find_runtime_executable("ffprobe")
    if not ffprobe:
        return {}

    try:
        completed = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration:format_tags=creation_time,com.apple.quicktime.creationdate",
                "-of",
                "json",
                str(audio_path),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=max(1, timeout_seconds),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if completed.returncode != 0:
        return {}

    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError:
        return {}
    format_data = payload.get("format")
    if not isinstance(format_data, dict):
        return {}

    metadata: dict[str, Any] = {}
    duration = parse_duration_seconds(format_data.get("duration"))
    if duration is not None:
        metadata["duration_seconds"] = duration

    tags = format_data.get("tags")
    if not isinstance(tags, dict):
        return metadata
    raw_creation_time = next(
        (str(tags.get(tag) or "").strip() for tag in CREATION_TIME_TAGS if str(tags.get(tag) or "").strip()),
        "",
    )
    if not raw_creation_time:
        return metadata

    metadata["raw_creation_time"] = raw_creation_time
    metadata["media_time_source"] = "embedded_media_creation_time"
    started_at = parse_media_datetime(raw_creation_time)
    if started_at is None:
        return metadata

    if started_at.tzinfo is None:
        metadata["media_created_at"] = started_at.isoformat(timespec="seconds")
        metadata["media_time_timezone_known"] = False
        metadata["recording_time_validation"] = "timezone_unknown"
        return metadata

    local_started_at = started_at.astimezone()
    metadata.update(
        {
            "media_created_at": local_started_at.isoformat(timespec="seconds"),
            "media_created_at_utc": started_at.astimezone(timezone.utc).isoformat(timespec="seconds"),
            "media_creation_time_epoch": int(started_at.timestamp()),
            "media_time_timezone_known": True,
        }
    )
    if duration is None:
        metadata["recording_time_validation"] = "duration_unknown"
        return metadata

    ended_timestamp = started_at.timestamp() + duration
    file_saved_timestamp = audio_path.stat().st_mtime
    metadata["file_saved_at"] = datetime.fromtimestamp(file_saved_timestamp).astimezone().isoformat(timespec="seconds")
    if ended_timestamp > file_saved_timestamp + 300:
        metadata["recording_time_validation"] = "inconsistent_with_file_timeline"
        return metadata

    ended_at = datetime.fromtimestamp(ended_timestamp, tz=timezone.utc).astimezone()
    metadata.update(
        {
            "recording_started_at": local_started_at.isoformat(timespec="seconds"),
            "recording_started_at_utc": started_at.astimezone(timezone.utc).isoformat(timespec="seconds"),
            "recording_started_at_epoch": int(started_at.timestamp()),
            "recording_ended_at": ended_at.isoformat(timespec="seconds"),
            "recording_time_source": "embedded_media_creation_time",
            "recording_time_timezone_known": True,
            "recording_time_validation": "plausible_file_timeline",
        }
    )
    return metadata


def parse_duration_seconds(value: Any) -> float | None:
    try:
        duration = float(value)
    except (TypeError, ValueError):
        return None
    if duration < 0:
        return None
    return round(duration, 3)


def parse_media_datetime(value: str) -> datetime | None:
    normalized = value.strip()
    if not normalized:
        return None
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def recording_metadata_summary(metadata: dict[str, Any]) -> str:
    started_at = str(metadata.get("recording_started_at") or "").strip()
    duration = metadata.get("duration_seconds")
    lines: list[str] = []
    if started_at:
        lines.append(f"录音文件内嵌开始时间：{started_at}（来源：媒体 creation_time）")
    if isinstance(duration, (int, float)):
        lines.append(f"录音时长：{format_duration(float(duration))}")
    if started_at:
        lines.append("注意：这是录音开始时间，不自动等同于会议正式开始时间；若与用户确认信息冲突，以用户确认信息为准。")
    elif metadata.get("recording_time_validation") == "inconsistent_with_file_timeline":
        media_created_at = str(metadata.get("media_created_at") or metadata.get("raw_creation_time") or "").strip()
        lines.append(
            f"媒体 creation_time：{media_created_at}，但与文件保存时间及录音时长不一致，按导出/封装时间处理，不作为录音开始时间。"
        )
    return "\n".join(lines)


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}小时{minutes}分{seconds}秒"
    if minutes:
        return f"{minutes}分{seconds}秒"
    return f"{seconds}秒"
