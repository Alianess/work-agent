from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import math
import select
import subprocess
import sys
import time
from typing import Any

from transcribe_qwen3_asr_mlx import DEFAULT_MLX_MODEL_ID
from transcribe_qwen3_asr_mlx import configure_project_cache
from transcribe_qwen3_asr_mlx import filter_pathological_repetitions
from transcribe_qwen3_asr_mlx import load_mlx_model
from transcribe_qwen3_asr_mlx import transcribe_one_mlx


SAMPLE_RATE = 16000


@dataclass(frozen=True)
class ChunkSpec:
    index: int
    start_seconds: float
    end_seconds: float
    mode: str
    speech_segments_ms: list[list[int]]

    @property
    def length_seconds(self) -> float:
        return max(0.0, self.end_seconds - self.start_seconds)

    def to_job(self, source: Path, chunk_path: Path) -> dict[str, Any]:
        return {
            "index": self.index,
            "chunk_path": str(chunk_path),
            "source_audio": str(source),
            "start_seconds": round(self.start_seconds, 3),
            "end_seconds": round(self.end_seconds, 3),
            "start": format_time(self.start_seconds),
            "end": format_time(self.end_seconds),
            "chunk_mode": self.mode,
            "vad_speech_segment_count": len(self.speech_segments_ms),
            "vad_speech_segments_ms": self.speech_segments_ms,
        }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Chunk a long meeting recording and transcribe it with Qwen3-ASR."
    )
    parser.add_argument("audio", help="Prepared meeting audio, preferably 16 kHz mono WAV.")
    parser.add_argument("--output-dir", default="meet_files/asr_outputs/qwen3_asr_chunked")
    parser.add_argument("--model-id", default=DEFAULT_MLX_MODEL_ID)
    parser.add_argument("--cache-dir", default="meeting_audio_minutes/model_cache")
    parser.add_argument(
        "--backend",
        choices=["mlx"],
        default="mlx",
        help="ASR inference backend. This project uses MLX 8bit on Apple Silicon.",
    )
    parser.add_argument("--device", default="mlx-metal")
    parser.add_argument("--language", default="zh")
    parser.add_argument(
        "--chunk-seconds",
        type=int,
        default=120,
        help="Maximum chunk span in seconds. In VAD mode, chunks are merged up to this length.",
    )
    parser.add_argument(
        "--chunk-mode",
        choices=["fixed", "vad"],
        default="fixed",
        help="fixed keeps old equal-length chunks; vad splits/merges on speech boundaries for best full-recording ASR.",
    )
    parser.add_argument(
        "--min-chunk-seconds",
        type=int,
        default=20,
        help="In VAD mode, try to avoid tiny chunks shorter than this when nearby speech can be merged.",
    )
    parser.add_argument(
        "--vad-padding-ms",
        type=int,
        default=300,
        help="Audio kept before/after each VAD chunk to reduce boundary word loss.",
    )
    parser.add_argument(
        "--vad-max-gap-ms",
        type=int,
        default=10000,
        help="In VAD mode, do not merge speech regions separated by a longer silence gap.",
    )
    parser.add_argument(
        "--vad-max-single-segment-ms",
        type=int,
        default=30000,
        help="Maximum single segment length passed to the FSMN VAD model.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--mlx-chunk-duration",
        type=float,
        default=30.0,
        help="Internal MLX model chunk duration in seconds inside each VAD/fixed chunk.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-chunks", type=int, default=0, help="Debug limit; 0 means all chunks.")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Only build/write the chunk plan; do not load Qwen3 or run ASR.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Kept for command compatibility; MLX Metal runs as one shared worker.",
    )
    args = parser.parse_args()

    source = Path(args.audio).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if args.chunk_seconds <= 0:
        raise ValueError("--chunk-seconds must be positive")
    if args.min_chunk_seconds < 0:
        raise ValueError("--min-chunk-seconds must be non-negative")
    if args.vad_padding_ms < 0:
        raise ValueError("--vad-padding-ms must be non-negative")
    if args.vad_max_gap_ms < 0:
        raise ValueError("--vad-max-gap-ms must be non-negative")
    args.backend = "mlx"
    args.device = "mlx-metal"
    if args.workers > 1:
        print("MLX backend uses one shared Metal worker; ignoring --workers > 1.", flush=True)
        args.workers = 1

    output_prefix = "qwen3-asr-mlx8"
    output_leaf = f"{output_prefix}-vad-chunked" if args.chunk_mode == "vad" else f"{output_prefix}-chunked"
    output_root = Path(args.output_dir).expanduser().resolve() / source.stem / output_leaf
    chunks_dir = output_root / "chunks"
    item_dir = output_root / "items"
    output_root.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    item_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    configure_project_cache(cache_dir)

    duration = probe_duration(source)
    chunk_specs, chunk_plan_meta = build_chunk_plan(
        source=source,
        duration=duration,
        cache_dir=cache_dir,
        args=args,
    )
    if args.max_chunks:
        chunk_specs = chunk_specs[: args.max_chunks]
        chunk_plan_meta["max_chunks_limit"] = args.max_chunks
        chunk_plan_meta["truncated"] = True
    chunk_count = len(chunk_specs)
    write_chunk_plan(output_root, source, duration, chunk_specs, chunk_plan_meta, args)

    print(f"Source: {source}", flush=True)
    print(
        f"Duration: {format_time(duration)} | chunks: {chunk_count} | mode: {chunk_plan_meta['effective_chunk_mode']}",
        flush=True,
    )
    print(f"Output: {output_root}", flush=True)
    print(f"Backend: {args.backend}", flush=True)
    print(f"Workers: {max(1, args.workers)}", flush=True)

    if args.plan_only:
        print(f"Plan written. Chunk plan: {output_root / 'chunk_plan.json'}", flush=True)
        return 0

    if not chunk_specs:
        raise RuntimeError("No chunks were produced for this audio.")

    if args.workers > 1:
        run_parallel(
            source=source,
            output_root=output_root,
            chunks_dir=chunks_dir,
            item_dir=item_dir,
            duration=duration,
            chunk_specs=chunk_specs,
            chunk_plan_meta=chunk_plan_meta,
            args=args,
        )
        print(f"Done. Transcript: {output_root / 'transcript.md'}", flush=True)
        return 0

    load_started = time.perf_counter()
    model = load_mlx_model(args.model_id)
    load_seconds = time.perf_counter() - load_started
    print(f"Model loaded in {load_seconds:.1f}s", flush=True)

    rows: list[dict[str, Any]] = []
    progress_jsonl = output_root / "progress.jsonl"
    for spec in chunk_specs:
        start = spec.start_seconds
        end = spec.end_seconds
        length = spec.length_seconds
        chunk_path = chunks_dir / chunk_name(spec.index, start, end)
        result_dir = item_dir / f"chunk_{spec.index:04d}"
        result_json = result_dir / "raw_result.json"
        result_txt = result_dir / "transcript.txt"

        if not chunk_path.exists():
            extract_chunk(source, chunk_path, start, length)

        if args.skip_existing and result_json.exists() and result_txt.exists():
            row = json.loads(result_json.read_text(encoding="utf-8"))
            row = sanitize_result_row(row)
            if row.get("repetition_filter", {}).get("applied"):
                write_item_outputs(row, result_dir)
            print(
                f"Qwen3 chunks: {spec.index}/{chunk_count} | {format_time(start)}-{format_time(end)} | skipped",
                flush=True,
            )
        else:
            print(
                f"Qwen3 chunks: {spec.index}/{chunk_count} | {format_time(start)}-{format_time(end)} | ASR start",
                flush=True,
            )
            infer_started = time.perf_counter()
            text = transcribe_one_mlx(
                model=model,
                source=chunk_path,
                language=args.language,
                max_new_tokens=args.max_new_tokens,
                chunk_duration=args.mlx_chunk_duration,
            )
            infer_seconds = time.perf_counter() - infer_started
            row = build_result_row(
                spec=spec,
                chunk_path=chunk_path,
                source=source,
                model_id=args.model_id,
                backend=args.backend,
                language=args.language,
                device=args.device,
                infer_seconds=infer_seconds,
                text=text,
            )
            write_item_outputs(row, result_dir)
            with progress_jsonl.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                f"Qwen3 chunks: {spec.index}/{chunk_count} | ASR {infer_seconds:.1f}s | {len(text)} chars",
                flush=True,
            )
        rows.append(row)
        write_summary(
            output_root=output_root,
            source=source,
            rows=rows,
            model_id=args.model_id,
            backend=args.backend,
            language=args.language,
            device=args.device,
            duration=duration,
            chunk_seconds=args.chunk_seconds,
            chunk_mode=chunk_plan_meta["effective_chunk_mode"],
            chunk_plan_meta=chunk_plan_meta,
            load_seconds=load_seconds,
            complete=len(rows) == chunk_count,
        )

    print(f"Done. Transcript: {output_root / 'transcript.md'}", flush=True)
    return 0


def run_parallel(
    *,
    source: Path,
    output_root: Path,
    chunks_dir: Path,
    item_dir: Path,
    duration: float,
    chunk_specs: list[ChunkSpec],
    chunk_plan_meta: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    started_at = time.perf_counter()
    progress_jsonl = output_root / "progress.jsonl"
    rows_by_index: dict[int, dict[str, Any]] = {}
    jobs: list[dict[str, Any]] = []
    chunk_count = len(chunk_specs)

    for spec in chunk_specs:
        start = spec.start_seconds
        end = spec.end_seconds
        chunk_path = chunks_dir / chunk_name(spec.index, start, end)
        result_dir = item_dir / f"chunk_{spec.index:04d}"
        result_json = result_dir / "raw_result.json"
        result_txt = result_dir / "transcript.txt"

        if not chunk_path.exists():
            extract_chunk(source, chunk_path, start, spec.length_seconds)

        if args.skip_existing and result_json.exists() and result_txt.exists():
            row = json.loads(result_json.read_text(encoding="utf-8"))
            row = sanitize_result_row(row)
            if row.get("repetition_filter", {}).get("applied"):
                write_item_outputs(row, result_dir)
            rows_by_index[spec.index] = row
            print(
                f"Qwen3 chunks: {spec.index}/{chunk_count} | {format_time(start)}-{format_time(end)} | skipped",
                flush=True,
            )
            continue

        jobs.append(spec.to_job(source, chunk_path))

    if not jobs:
        rows = sorted(rows_by_index.values(), key=lambda item: int(item["index"]))
        write_summary(
            output_root=output_root,
            source=source,
            rows=rows,
            model_id=args.model_id,
            backend=args.backend,
            language=args.language,
            device=args.device,
            duration=duration,
            chunk_seconds=args.chunk_seconds,
            chunk_mode=chunk_plan_meta["effective_chunk_mode"],
            chunk_plan_meta=chunk_plan_meta,
            load_seconds=0.0,
            complete=len(rows) == chunk_count,
        )
        return

    worker_count = min(max(1, args.workers), len(jobs))
    print(f"Parallel ASR jobs: {len(jobs)} | workers: {worker_count}", flush=True)
    workers = start_qwen3_workers(worker_count, args)
    pending_jobs = list(jobs)
    active: dict[int, dict[str, Any]] = {}
    try:
        for worker in workers:
            if pending_jobs:
                send_worker_job(worker, pending_jobs.pop(0))
                active[int(worker["id"])] = worker

        while active:
            streams = [worker["process"].stdout for worker in active.values()]
            ready_streams, _, _ = select.select(streams, [], [], 1.0)
            if not ready_streams:
                for worker in active.values():
                    if worker["process"].poll() is not None:
                        raise RuntimeError(f"Qwen3 worker {worker['id']} exited unexpectedly.")
                continue

            for stream in ready_streams:
                worker = next(item for item in active.values() if item["process"].stdout is stream)
                event = read_worker_event(worker)
                if event.get("event") != "result":
                    continue
                if not event.get("ok"):
                    rows = sorted(rows_by_index.values(), key=lambda item: int(item["index"]))
                    write_summary(
                        output_root=output_root,
                        source=source,
                        rows=rows,
                        model_id=args.model_id,
                        backend=args.backend,
                        language=args.language,
                        device=args.device,
                        duration=duration,
                        chunk_seconds=args.chunk_seconds,
                        chunk_mode=chunk_plan_meta["effective_chunk_mode"],
                        chunk_plan_meta=chunk_plan_meta,
                        load_seconds=time.perf_counter() - started_at,
                        complete=False,
                    )
                    raise RuntimeError(str(event.get("error") or "Qwen3 worker failed"))

                row = dict(event["row"])
                index = int(row["index"])
                rows_by_index[index] = row
                result_dir = item_dir / f"chunk_{index:04d}"
                write_item_outputs(row, result_dir)
                with progress_jsonl.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows = sorted(rows_by_index.values(), key=lambda item: int(item["index"]))
                print(
                    f"Qwen3 chunks: {len(rows_by_index)}/{chunk_count} | "
                    f"{row['start']}-{row['end']} | ASR {row['infer_seconds']}s | {row['char_count']} chars",
                    flush=True,
                )
                write_summary(
                    output_root=output_root,
                    source=source,
                    rows=rows,
                    model_id=args.model_id,
                    backend=args.backend,
                    language=args.language,
                    device=args.device,
                    duration=duration,
                    chunk_seconds=args.chunk_seconds,
                    chunk_mode=chunk_plan_meta["effective_chunk_mode"],
                    chunk_plan_meta=chunk_plan_meta,
                    load_seconds=time.perf_counter() - started_at,
                    complete=len(rows_by_index) == chunk_count,
                )

                if pending_jobs:
                    send_worker_job(worker, pending_jobs.pop(0))
                else:
                    active.pop(int(worker["id"]), None)
                    send_worker_shutdown(worker)
    finally:
        stop_qwen3_workers(workers)


def build_chunk_plan(
    *,
    source: Path,
    duration: float,
    cache_dir: Path,
    args: argparse.Namespace,
) -> tuple[list[ChunkSpec], dict[str, Any]]:
    if args.chunk_mode == "fixed":
        chunks = build_fixed_chunk_specs(duration, args.chunk_seconds)
        return chunks, {
            "requested_chunk_mode": "fixed",
            "effective_chunk_mode": "fixed",
            "chunk_seconds": args.chunk_seconds,
        }

    print("Running FSMN VAD for Qwen3 chunk planning...", flush=True)
    started_at = time.perf_counter()
    raw_segments = detect_speech_segments(
        source=source,
        cache_dir=cache_dir,
        device="cpu",
        max_single_segment_ms=args.vad_max_single_segment_ms,
    )
    vad_seconds = time.perf_counter() - started_at
    if not raw_segments:
        print("VAD returned no speech segments; falling back to fixed chunks.", flush=True)
        chunks = build_fixed_chunk_specs(duration, args.chunk_seconds)
        return chunks, {
            "requested_chunk_mode": "vad",
            "effective_chunk_mode": "fixed",
            "fallback_reason": "empty_vad_segments",
            "chunk_seconds": args.chunk_seconds,
            "vad_seconds": round(vad_seconds, 3),
            "raw_vad_segment_count": 0,
        }

    chunks = merge_vad_segments_to_chunk_specs(
        raw_segments=raw_segments,
        duration=duration,
        max_chunk_seconds=args.chunk_seconds,
        min_chunk_seconds=args.min_chunk_seconds,
        padding_ms=args.vad_padding_ms,
        max_gap_ms=args.vad_max_gap_ms,
    )
    if not chunks:
        print("VAD produced no usable chunks; falling back to fixed chunks.", flush=True)
        chunks = build_fixed_chunk_specs(duration, args.chunk_seconds)
        return chunks, {
            "requested_chunk_mode": "vad",
            "effective_chunk_mode": "fixed",
            "fallback_reason": "no_usable_vad_chunks",
            "chunk_seconds": args.chunk_seconds,
            "vad_seconds": round(vad_seconds, 3),
            "raw_vad_segment_count": len(raw_segments),
        }

    print(
        f"VAD speech segments: {len(raw_segments)} | merged Qwen3 chunks: {len(chunks)} | VAD {vad_seconds:.1f}s",
        flush=True,
    )
    return chunks, {
        "requested_chunk_mode": "vad",
        "effective_chunk_mode": "vad",
        "chunk_seconds": args.chunk_seconds,
        "min_chunk_seconds": args.min_chunk_seconds,
        "vad_padding_ms": args.vad_padding_ms,
        "vad_max_gap_ms": args.vad_max_gap_ms,
        "vad_max_single_segment_ms": args.vad_max_single_segment_ms,
        "vad_seconds": round(vad_seconds, 3),
        "raw_vad_segment_count": len(raw_segments),
    }


def build_fixed_chunk_specs(duration: float, chunk_seconds: int) -> list[ChunkSpec]:
    chunk_count = math.ceil(duration / chunk_seconds)
    chunks: list[ChunkSpec] = []
    for index in range(1, chunk_count + 1):
        start = (index - 1) * chunk_seconds
        end = min(duration, start + chunk_seconds)
        if end <= start:
            continue
        chunks.append(
            ChunkSpec(
                index=index,
                start_seconds=round(start, 3),
                end_seconds=round(end, 3),
                mode="fixed",
                speech_segments_ms=[],
            )
        )
    return chunks


def detect_speech_segments(
    *,
    source: Path,
    cache_dir: Path,
    device: str,
    max_single_segment_ms: int,
) -> list[list[int]]:
    from funasr import AutoModel

    configure_funasr_vad_cache(cache_dir)
    vad_model_ref = resolve_vad_model_ref(cache_dir)
    vad_model = AutoModel(
        model=str(vad_model_ref),
        device=device,
        disable_update=True,
        vad_kwargs={"max_single_segment_time": max_single_segment_ms},
    )
    vad_result = vad_model.generate(input=str(source), disable_pbar=True)
    segments: list[list[int]] = []
    for item in vad_result:
        for raw_segment in item.get("value") or []:
            if len(raw_segment) < 2:
                continue
            start_ms = max(0, int(raw_segment[0]))
            end_ms = max(start_ms, int(raw_segment[1]))
            if end_ms > start_ms:
                segments.append([start_ms, end_ms])
    return sorted(segments, key=lambda item: (item[0], item[1]))


def configure_funasr_vad_cache(cache_dir: Path) -> None:
    import os

    os.environ.setdefault("MODELSCOPE_CACHE", str(cache_dir / "modelscope"))
    os.environ.setdefault("HF_HOME", str(cache_dir / "huggingface"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_dir / "huggingface" / "hub"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir / "xdg"))


def resolve_vad_model_ref(cache_dir: Path) -> Path:
    vad_path = cache_dir / "modelscope" / "iic" / "speech_fsmn_vad_zh-cn-16k-common-pytorch"
    if not (vad_path / "model.pt").exists():
        raise FileNotFoundError(
            "Missing local FSMN VAD model used for Qwen3 chunk planning: "
            f"{vad_path}"
        )
    return vad_path


def merge_vad_segments_to_chunk_specs(
    *,
    raw_segments: list[list[int]],
    duration: float,
    max_chunk_seconds: int,
    min_chunk_seconds: int,
    padding_ms: int,
    max_gap_ms: int,
) -> list[ChunkSpec]:
    duration_ms = max(0, int(round(duration * 1000)))
    max_chunk_ms = max(1000, int(max_chunk_seconds * 1000))
    min_chunk_ms = max(0, int(min_chunk_seconds * 1000))

    clipped: list[list[int]] = []
    for start_ms, end_ms in raw_segments:
        start_ms = min(max(0, int(start_ms)), duration_ms)
        end_ms = min(max(start_ms, int(end_ms)), duration_ms)
        if end_ms > start_ms:
            clipped.append([start_ms, end_ms])
    if not clipped:
        return []

    chunks: list[ChunkSpec] = []
    current_segments: list[list[int]] = [clipped[0]]
    current_start = clipped[0][0]
    current_end = clipped[0][1]

    def flush_current() -> None:
        nonlocal current_segments, current_start, current_end
        if not current_segments:
            return
        padded_start = max(0, current_start - padding_ms)
        padded_end = min(duration_ms, current_end + padding_ms)
        if padded_end <= padded_start:
            return
        chunks.append(
            ChunkSpec(
                index=len(chunks) + 1,
                start_seconds=round(padded_start / 1000, 3),
                end_seconds=round(padded_end / 1000, 3),
                mode="vad",
                speech_segments_ms=[list(item) for item in current_segments],
            )
        )
        current_segments = []

    for start_ms, end_ms in clipped[1:]:
        gap_ms = max(0, start_ms - current_end)
        proposed_end = max(current_end, end_ms)
        proposed_span_ms = proposed_end - current_start
        current_span_ms = current_end - current_start
        can_merge_by_length = proposed_span_ms <= max_chunk_ms
        can_merge_to_avoid_tiny_chunk = (
            current_span_ms < min_chunk_ms and proposed_span_ms <= int(max_chunk_ms * 1.25)
        )
        can_merge_by_gap = gap_ms <= max_gap_ms

        if can_merge_by_gap and (can_merge_by_length or can_merge_to_avoid_tiny_chunk):
            current_segments.append([start_ms, end_ms])
            current_end = proposed_end
            continue

        flush_current()
        current_segments = [[start_ms, end_ms]]
        current_start = start_ms
        current_end = end_ms

    flush_current()
    return chunks


def write_chunk_plan(
    output_root: Path,
    source: Path,
    duration: float,
    chunks: list[ChunkSpec],
    chunk_plan_meta: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    plan = {
        "source_audio": str(source),
        "duration_seconds": round(duration, 3),
        "duration": format_time(duration),
        "model_id": args.model_id,
        "backend": args.backend,
        "language": args.language,
        "device": args.device,
        "chunk_plan": chunk_plan_meta,
        "chunk_count": len(chunks),
        "chunks": [chunk_spec_to_dict(item) for item in chunks],
    }
    (output_root / "chunk_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def chunk_spec_to_dict(spec: ChunkSpec) -> dict[str, Any]:
    return {
        "index": spec.index,
        "start_seconds": round(spec.start_seconds, 3),
        "end_seconds": round(spec.end_seconds, 3),
        "length_seconds": round(spec.length_seconds, 3),
        "start": format_time(spec.start_seconds),
        "end": format_time(spec.end_seconds),
        "chunk_mode": spec.mode,
        "vad_speech_segment_count": len(spec.speech_segments_ms),
        "vad_speech_segments_ms": spec.speech_segments_ms,
    }


def build_result_row(
    *,
    spec: ChunkSpec,
    chunk_path: Path,
    source: Path,
    model_id: str,
    backend: str,
    language: str,
    device: str,
    infer_seconds: float,
    text: str,
) -> dict[str, Any]:
    row = spec.to_job(source, chunk_path)
    row.update(
        {
            "audio": str(chunk_path),
            "model_id": model_id,
            "backend": backend,
            "language": language,
            "device": device,
            "infer_seconds": round(infer_seconds, 3),
            "char_count": len(text),
            "transcription": text,
        }
    )
    row.pop("chunk_path", None)
    return sanitize_result_row(row)


def sanitize_result_row(row: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(row)
    original = str(sanitized.get("transcription") or "")
    filtered, events = filter_pathological_repetitions(original)
    sanitized["transcription"] = filtered
    sanitized["char_count"] = len(filtered)
    if events:
        sanitized["repetition_filter"] = {
            "applied": True,
            "original_char_count": len(original),
            "removed_chars": sum(int(event["removed_chars"]) for event in events),
            "runs": events,
        }
    return sanitized


def start_qwen3_workers(worker_count: int, args: argparse.Namespace) -> list[dict[str, Any]]:
    worker_script = Path(__file__).with_name("qwen3_asr_worker.py").resolve()
    if not worker_script.is_file():
        raise FileNotFoundError(f"Missing Qwen3 worker script: {worker_script}")
    workers: list[dict[str, Any]] = []
    for worker_id in range(1, worker_count + 1):
        command = [
            sys.executable,
            str(worker_script),
            "--model-id",
            args.model_id,
            "--cache-dir",
            str(Path(args.cache_dir).expanduser().resolve()),
            "--device",
            args.device,
            "--language",
            args.language,
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--worker-id",
            str(worker_id),
        ]
        process = subprocess.Popen(
            command,
            cwd=Path.cwd(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )
        worker = {"id": worker_id, "process": process}
        try:
            event = read_worker_event(worker)
            if event.get("event") != "ready" or not event.get("ok", True):
                raise RuntimeError(str(event.get("error") or f"Qwen3 worker {worker_id} failed to start"))
            print(
                f"Qwen3 worker {worker_id} ready | load {event.get('load_seconds', '?')}s",
                flush=True,
            )
            workers.append(worker)
        except Exception:
            stop_qwen3_workers(workers + [worker])
            raise
    return workers


def send_worker_job(worker: dict[str, Any], job: dict[str, Any]) -> None:
    process = worker["process"]
    assert process.stdin is not None
    process.stdin.write(json.dumps({"event": "transcribe", "job": job}, ensure_ascii=False) + "\n")
    process.stdin.flush()


def send_worker_shutdown(worker: dict[str, Any]) -> None:
    process = worker["process"]
    if process.poll() is not None:
        return
    try:
        assert process.stdin is not None
        process.stdin.write(json.dumps({"event": "shutdown"}) + "\n")
        process.stdin.flush()
    except Exception:
        pass


def stop_qwen3_workers(workers: list[dict[str, Any]]) -> None:
    for worker in workers:
        send_worker_shutdown(worker)
    for worker in workers:
        process = worker["process"]
        if process.poll() is not None:
            continue
        try:
            process.terminate()
            process.wait(timeout=3)
        except Exception:
            process.kill()


def read_worker_event(worker: dict[str, Any]) -> dict[str, Any]:
    process = worker["process"]
    assert process.stdout is not None
    while True:
        line = process.stdout.readline()
        if not line:
            raise RuntimeError(f"Qwen3 worker {worker['id']} returned no data.")
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            return event


def probe_duration(source: Path) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(source),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    return float(completed.stdout.strip())


def extract_chunk(source: Path, destination: Path, start_seconds: float, length_seconds: float) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_seconds:.3f}",
        "-t",
        f"{length_seconds:.3f}",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        str(destination),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())


def chunk_name(index: int, start: float, end: float) -> str:
    return f"chunk_{index:04d}_{format_time(start).replace(':', '-')}_{format_time(end).replace(':', '-')}.wav"


def format_time(seconds: float) -> str:
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def write_summary(
    *,
    output_root: Path,
    source: Path,
    rows: list[dict[str, Any]],
    model_id: str,
    backend: str,
    language: str,
    device: str,
    duration: float,
    chunk_seconds: int,
    chunk_mode: str,
    chunk_plan_meta: dict[str, Any],
    load_seconds: float,
    complete: bool,
) -> None:
    summary = {
        "source_audio": str(source),
        "model_id": model_id,
        "backend": backend,
        "language": language,
        "device": device,
        "duration_seconds": round(duration, 3),
        "duration": format_time(duration),
        "chunk_seconds": chunk_seconds,
        "chunk_mode": chunk_mode,
        "chunk_plan": chunk_plan_meta,
        "load_seconds": round(load_seconds, 3),
        "complete": complete,
        "completed_chunks": len(rows),
        "total_chars": sum(item.get("char_count", 0) for item in rows),
        "items": rows,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "transcript.md").write_text(render_summary_markdown(summary), encoding="utf-8")
    (output_root / "transcript.txt").write_text(render_plain_text(rows), encoding="utf-8")


def write_item_outputs(row: dict[str, Any], result_dir: Path) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "transcript.txt").write_text(row["transcription"] + "\n", encoding="utf-8")
    (result_dir / "raw_result.json").write_text(
        json.dumps(row, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (result_dir / "transcript.md").write_text(render_item_markdown(row), encoding="utf-8")


def render_item_markdown(row: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"# Qwen3-ASR 片段 {row['index']}",
            "",
            f"- 时间：{row['start']} - {row['end']}",
            f"- 后端：{row.get('backend', 'mlx')}",
            f"- 分块模式：{row.get('chunk_mode', 'fixed')}",
            f"- VAD语音段：{row.get('vad_speech_segment_count', 0)}",
            f"- 音频：`{row['audio']}`",
            f"- 推理耗时：{row['infer_seconds']} 秒",
            f"- 字数：{row['char_count']}",
            "",
            "## 正文",
            "",
            row["transcription"],
            "",
        ]
    )


def render_summary_markdown(summary: dict[str, Any]) -> str:
    chunk_plan = summary.get("chunk_plan") or {}
    lines = [
        "# Qwen3-ASR 分块转写",
        "",
        f"- 源音频：`{summary['source_audio']}`",
        f"- 模型：`{summary['model_id']}`",
        f"- 后端：{summary.get('backend', 'mlx')}",
        f"- 语言：`{summary['language']}`",
        f"- 设备：`{summary['device']}`",
        f"- 音频时长：{summary['duration']}",
        f"- 分块模式：{summary.get('chunk_mode', 'fixed')}",
        f"- 目标分块长度：{summary['chunk_seconds']} 秒",
        f"- VAD原始语音段：{chunk_plan.get('raw_vad_segment_count', 0)}",
        f"- 已完成片段：{summary['completed_chunks']}",
        f"- 总字数：{summary['total_chars']}",
        f"- 完成状态：{'已完成' if summary['complete'] else '进行中/中断可续跑'}",
        "",
        "| 片段 | 时间 | 模式 | VAD语音段 | ASR耗时 | 字数 |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for item in summary["items"]:
        lines.append(
            f"| {item['index']} | {item['start']} - {item['end']} | "
            f"{item.get('chunk_mode', 'fixed')} | {item.get('vad_speech_segment_count', 0)} | "
            f"{item['infer_seconds']}s | {item['char_count']} |"
        )
    lines.extend(["", "## 正文", ""])
    for item in summary["items"]:
        lines.extend(
            [
                f"### 片段 {item['index']}（{item['start']} - {item['end']}）",
                "",
                item["transcription"],
                "",
            ]
        )
    return "\n".join(lines)


def render_plain_text(rows: list[dict[str, Any]]) -> str:
    blocks = []
    for item in rows:
        blocks.append(f"[{item['start']} - {item['end']}]\n{item['transcription']}")
    return "\n\n".join(blocks) + ("\n" if blocks else "")


if __name__ == "__main__":
    raise SystemExit(main())
