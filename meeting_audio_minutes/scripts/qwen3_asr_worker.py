from __future__ import annotations

from pathlib import Path
import argparse
import contextlib
import json
import sys
import time
import traceback
from typing import Any

from transcribe_qwen3_asr_mlx import DEFAULT_MLX_MODEL_ID
from transcribe_qwen3_asr_mlx import filter_pathological_repetitions
from transcribe_qwen3_asr_mlx import load_mlx_model
from transcribe_qwen3_asr_mlx import normalize_language
from transcribe_qwen3_asr_mlx import transcribe_one_mlx


JSON_STDOUT = sys.stdout


def emit(payload: dict[str, Any]) -> None:
    JSON_STDOUT.write(json.dumps(payload, ensure_ascii=False) + "\n")
    JSON_STDOUT.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description="JSONL worker for Qwen3-ASR MLX realtime/chunk transcription.")
    parser.add_argument("--model-id", default=DEFAULT_MLX_MODEL_ID)
    parser.add_argument("--cache-dir", default="meeting_audio_minutes/model_cache")
    parser.add_argument("--device", default="mlx-metal")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--mlx-chunk-duration", type=float, default=30.0)
    parser.add_argument("--worker-id", type=int, default=1)
    args = parser.parse_args()

    try:
        with contextlib.redirect_stdout(sys.stderr):
            started_at = time.perf_counter()
            model = load_mlx_model(args.model_id)
            load_seconds = time.perf_counter() - started_at
        emit(
            {
                "event": "ready",
                "ok": True,
                "worker_id": args.worker_id,
                "backend": "mlx",
                "load_seconds": round(load_seconds, 3),
            }
        )
    except Exception as exc:
        emit(
            {
                "event": "ready",
                "ok": False,
                "worker_id": args.worker_id,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        return 1

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            emit({"event": "error", "ok": False, "worker_id": args.worker_id, "error": str(exc)})
            continue

        event_name = event.get("event")
        if event_name == "shutdown":
            return 0
        if event_name != "transcribe":
            emit(
                {
                    "event": "error",
                    "ok": False,
                    "worker_id": args.worker_id,
                    "error": f"unknown event: {event_name}",
                }
            )
            continue

        job = event.get("job") or {}
        try:
            row = transcribe_job(
                job=job,
                model=model,
                model_id=args.model_id,
                language=args.language,
                device=args.device,
                max_new_tokens=args.max_new_tokens,
                mlx_chunk_duration=args.mlx_chunk_duration,
            )
            emit({"event": "result", "ok": True, "worker_id": args.worker_id, "row": row})
        except Exception as exc:
            emit(
                {
                    "event": "result",
                    "ok": False,
                    "worker_id": args.worker_id,
                    "job": job,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
    return 0


def transcribe_job(
    *,
    job: dict[str, Any],
    model: Any,
    model_id: str,
    language: str,
    device: str,
    max_new_tokens: int,
    mlx_chunk_duration: float,
) -> dict[str, Any]:
    chunk_path = Path(str(job["chunk_path"])).expanduser().resolve()
    infer_started = time.perf_counter()
    with contextlib.redirect_stdout(sys.stderr):
        text = transcribe_one_mlx(
            model=model,
            source=chunk_path,
            language=language,
            max_new_tokens=max_new_tokens,
            chunk_duration=mlx_chunk_duration,
        )
    original_char_count = len(text)
    text, repetition_events = filter_pathological_repetitions(text)
    infer_seconds = time.perf_counter() - infer_started
    row = {
        "index": int(job["index"]),
        "audio": str(chunk_path),
        "source_audio": str(job.get("source_audio") or job.get("source") or ""),
        "start_seconds": job["start_seconds"],
        "end_seconds": job["end_seconds"],
        "start": job["start"],
        "end": job["end"],
        "chunk_mode": job.get("chunk_mode", "fixed"),
        "vad_speech_segment_count": int(job.get("vad_speech_segment_count") or 0),
        "vad_speech_segments_ms": job.get("vad_speech_segments_ms") or [],
        "model_id": model_id,
        "backend": "mlx",
        "language": normalize_language(language),
        "device": device,
        "infer_seconds": round(infer_seconds, 3),
        "char_count": len(text),
        "transcription": text,
    }
    if repetition_events:
        row["repetition_filter"] = {
            "applied": True,
            "original_char_count": original_char_count,
            "removed_chars": sum(int(event["removed_chars"]) for event in repetition_events),
            "runs": repetition_events,
        }
    return row


if __name__ == "__main__":
    raise SystemExit(main())
