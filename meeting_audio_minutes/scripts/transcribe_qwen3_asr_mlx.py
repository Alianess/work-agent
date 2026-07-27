from __future__ import annotations

from pathlib import Path
import json
import os
import time
from typing import Any


DEFAULT_MLX_MODEL_ID = "meeting_audio_minutes/model_cache/mlx-community/Qwen3-ASR-1.7B-8bit"


def configure_project_cache(cache_dir: Path) -> None:
    os.environ.setdefault("HF_HOME", str(cache_dir / "huggingface"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_dir / "huggingface" / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir / "huggingface" / "hub"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir / "xdg"))


def patch_transformers_tokenizer_register() -> None:
    """Keep mlx-lm 0.31.x compatible with the current Transformers 5 API."""

    from transformers import AutoTokenizer

    original = AutoTokenizer.register
    if getattr(original, "_work_agent_mlx_safe", False):
        return

    def safe_register(config_class: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(config_class, str):
            return None
        return original(config_class, *args, **kwargs)

    safe_register._work_agent_mlx_safe = True  # type: ignore[attr-defined]
    AutoTokenizer.register = staticmethod(safe_register)


def load_mlx_model(model_id: str) -> Any:
    patch_transformers_tokenizer_register()
    from mlx_audio.stt.utils import load_model

    return load_model(model_id)


def normalize_language(language: str) -> str:
    language = (language or "").strip().lower()
    if language in {"chinese", "zh", "zh-cn", "cn", "mandarin"}:
        return "zh"
    return language or "zh"


def filter_pathological_repetitions(
    text: str,
    *,
    min_repetitions: int = 12,
    keep_repetitions: int = 3,
    max_unit_chars: int = 12,
) -> tuple[str, list[dict[str, Any]]]:
    """Collapse exact, abnormally long token loops without changing normal stutters."""
    if not text or min_repetitions <= keep_repetitions:
        return text, []

    output: list[str] = []
    events: list[dict[str, Any]] = []
    cursor = 0
    text_length = len(text)
    while cursor < text_length:
        matched = False
        for unit_length in range(1, min(max_unit_chars, text_length - cursor) + 1):
            unit = text[cursor : cursor + unit_length]
            repetitions = 1
            while text.startswith(unit, cursor + repetitions * unit_length):
                repetitions += 1
            if repetitions < min_repetitions:
                continue

            kept = min(keep_repetitions, repetitions)
            output.append(unit * kept)
            events.append(
                {
                    "unit": unit,
                    "repetitions": repetitions,
                    "kept_repetitions": kept,
                    "removed_chars": (repetitions - kept) * unit_length,
                }
            )
            cursor += repetitions * unit_length
            matched = True
            break

        if not matched:
            output.append(text[cursor])
            cursor += 1

    return "".join(output), events


def transcribe_one_mlx(
    *,
    model: Any,
    source: Path,
    language: str,
    max_new_tokens: int,
    chunk_duration: float = 30.0,
) -> str:
    segments = model.generate(
        str(source),
        language=normalize_language(language),
        max_tokens=max_new_tokens,
        chunk_duration=chunk_duration,
        verbose=False,
    )
    return str(getattr(segments, "text", "") or "").strip()


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Transcribe one audio file with Qwen3-ASR MLX 8bit.")
    parser.add_argument("audio")
    parser.add_argument("--output-dir", default="meet_files/asr_outputs/qwen3_asr_mlx")
    parser.add_argument("--model-id", default=DEFAULT_MLX_MODEL_ID)
    parser.add_argument("--language", default="zh")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--chunk-duration", type=float, default=30.0)
    args = parser.parse_args()

    source = Path(args.audio).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    output_root = Path(args.output_dir).expanduser().resolve() / source.stem / "qwen3-asr-mlx8"
    output_root.mkdir(parents=True, exist_ok=True)

    load_started = time.perf_counter()
    model = load_mlx_model(args.model_id)
    load_seconds = time.perf_counter() - load_started
    infer_started = time.perf_counter()
    text = transcribe_one_mlx(
        model=model,
        source=source,
        language=args.language,
        max_new_tokens=args.max_new_tokens,
        chunk_duration=args.chunk_duration,
    )
    original_char_count = len(text)
    text, repetition_events = filter_pathological_repetitions(text)
    infer_seconds = time.perf_counter() - infer_started
    row = {
        "audio": str(source),
        "name": source.stem,
        "model_id": args.model_id,
        "backend": "mlx",
        "language": normalize_language(args.language),
        "device": "mlx-metal",
        "load_seconds": round(load_seconds, 3),
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
    (output_root / "transcript.txt").write_text(text + "\n", encoding="utf-8")
    (output_root / "raw_result.json").write_text(
        json.dumps(row, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Done. Transcript: {output_root / 'transcript.txt'}")
    print(f"- load seconds: {load_seconds:.1f}")
    print(f"- infer seconds: {infer_seconds:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
