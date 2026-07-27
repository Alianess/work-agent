from __future__ import annotations

from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path
import subprocess
import sys

from .audio import AudioPreprocessError, preprocess_audio


DEFAULT_QWEN3_MODEL_ID = "meeting_audio_minutes/model_cache/mlx-community/Qwen3-ASR-1.7B-8bit"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source = Path(args.audio).expanduser()
    if not source.exists():
        print(f"Audio file not found: {source}", file=sys.stderr)
        return 2

    run_dir = resolve_run_dir(args, source)
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        print("[1/2] Preparing audio with FFmpeg spectral denoise...")
        denoise_backend = "none" if args.no_denoise else args.denoise_backend
        prepared_audio = preprocess_audio(
            source,
            run_dir / "audio",
            denoise_backend=denoise_backend,
            use_postfilter=False,
            sample_rate=args.sample_rate,
        )
        if prepared_audio.warning:
            print(f"[warn] {prepared_audio.warning}")
        print(f"[info] Denoise backend: {prepared_audio.denoise_backend}")

        print("[2/2] Running Qwen3-ASR with VAD-boundary chunking...")
        transcript_path = run_qwen3_chunked(prepared_audio.path, run_dir / "qwen3", args)
    except (AudioPreprocessError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Failed: {error}", file=sys.stderr)
        return 1

    print("")
    print(f"Done. Output directory: {run_dir}")
    print(f"- prepared audio: {prepared_audio.path}")
    print(f"- transcript: {transcript_path}")
    return 0


def run_qwen3_chunked(audio_path: Path, output_dir: Path, args: Namespace) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    script_path = project_root / "scripts" / "transcribe_qwen3_asr_chunked.py"
    if not script_path.is_file():
        raise FileNotFoundError(f"Missing Qwen3-ASR script: {script_path}")

    command = [
        sys.executable,
        str(script_path),
        str(audio_path),
        "--output-dir",
        str(output_dir),
        "--backend",
        "mlx",
        "--model-id",
        args.model_id,
        "--cache-dir",
        str(Path(args.cache_dir).expanduser()),
        "--device",
        args.device,
        "--language",
        args.language,
        "--chunk-mode",
        "vad",
        "--chunk-seconds",
        str(args.chunk_seconds),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--workers",
        "1",
        "--skip-existing",
    ]
    completed = subprocess.run(command, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Qwen3-ASR failed with exit code {completed.returncode}")

    candidates = sorted(output_dir.rglob("transcript.txt"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError("Qwen3-ASR finished but transcript.txt was not found.")
    return candidates[0]


def parse_args(argv: list[str] | None) -> Namespace:
    parser = ArgumentParser(
        prog="meeting-minutes",
        description="Transcribe noisy Chinese meeting audio with Qwen3-ASR and VAD-boundary chunking.",
    )
    parser.add_argument("audio", help="Path to the meeting recording.")
    parser.add_argument(
        "-o",
        "--output-dir",
        help="Output directory. Defaults to outputs/<audio-stem>_<timestamp>/.",
    )
    parser.add_argument("--model-id", default=DEFAULT_QWEN3_MODEL_ID, help="Qwen3-ASR model id or local snapshot path.")
    parser.add_argument("--cache-dir", default="meeting_audio_minutes/model_cache", help="Project-local model cache.")
    parser.add_argument("--language", default="zh", help="Qwen3-ASR language hint.")
    parser.add_argument("--device", default="mlx-metal", help="Qwen3-ASR MLX device.")
    parser.add_argument("--chunk-seconds", type=int, default=120, help="Target max VAD-merged Qwen3 chunk length.")
    parser.add_argument("--workers", type=int, default=1, help="Kept for compatibility; MLX uses one Metal worker.")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--sample-rate", type=int, default=16000, help="Prepared mono WAV sample rate.")
    parser.add_argument(
        "--denoise-backend",
        choices=["ffmpeg", "none", "auto", "deepfilter"],
        default="ffmpeg",
        help="Default is ffmpeg afftdn, which tested best for the current meeting recordings.",
    )
    parser.add_argument("--no-denoise", action="store_true", help="Skip denoise and only normalize audio format.")
    return parser.parse_args(argv)


def resolve_run_dir(args: Namespace, source: Path) -> Path:
    if args.output_dir:
        return Path(args.output_dir).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (Path.cwd() / "outputs" / f"{source.stem}_{timestamp}").resolve()
