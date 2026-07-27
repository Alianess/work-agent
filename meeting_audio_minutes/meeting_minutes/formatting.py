from __future__ import annotations

from pathlib import Path
from typing import Any
import json


def get_segments(result: dict[str, Any]) -> list[dict[str, Any]]:
    return list(result.get("segments") or [])


def get_transcript_text(result: dict[str, Any]) -> str:
    text = str(result.get("text") or "").strip()
    if text:
        return text
    return " ".join(str(segment.get("text") or "").strip() for segment in get_segments(result)).strip()


def write_plain_transcript(result: dict[str, Any], destination: str | Path) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    for segment in get_segments(result):
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        start = format_timestamp(float(segment.get("start") or 0.0))
        end = format_timestamp(float(segment.get("end") or 0.0))
        lines.append(f"[{start} - {end}] {text}")

    if not lines:
        text = get_transcript_text(result)
        if text:
            lines.append(text)

    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return path


def write_srt(result: dict[str, Any], destination: str | Path) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)

    blocks: list[str] = []
    for index, segment in enumerate(get_segments(result), start=1):
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        start = format_srt_timestamp(float(segment.get("start") or 0.0))
        end = format_srt_timestamp(float(segment.get("end") or 0.0))
        blocks.append(f"{index}\n{start} --> {end}\n{text}")

    path.write_text("\n\n".join(blocks).strip() + "\n", encoding="utf-8")
    return path


def write_raw_json(result: dict[str, Any], destination: str | Path) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return path


def format_timestamp(seconds: float) -> str:
    seconds = max(0.0, seconds)
    whole_seconds = int(seconds)
    hours, remainder = divmod(whole_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_srt_timestamp(seconds: float) -> str:
    seconds = max(0.0, seconds)
    whole_seconds = int(seconds)
    milliseconds = int(round((seconds - whole_seconds) * 1000))
    if milliseconds == 1000:
        whole_seconds += 1
        milliseconds = 0
    hours, remainder = divmod(whole_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
