from __future__ import annotations

from pathlib import Path
import argparse
import subprocess


SAMPLE_RATE = 16000
STANDARD_FILTER = (
    "highpass=f=75,lowpass=f=7600,dynaudnorm=f=150:g=12,"
    "loudnorm=I=-18:LRA=11:TP=-1.5,aresample=16000"
)
SPECTRAL_FILTER = (
    "afftdn=nf=-25,dynaudnorm=f=150:g=12,"
    "loudnorm=I=-18:LRA=11:TP=-1.5,aresample=16000"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare a long meeting recording for Qwen3-ASR with the selected denoise preset."
    )
    parser.add_argument("audio", help="Source meeting recording.")
    parser.add_argument("--output", required=True, help="Output 16 kHz mono WAV.")
    parser.add_argument(
        "--preset",
        choices=["standard", "ffmpeg_spectral"],
        default="ffmpeg_spectral",
        help="standard only normalizes audio; ffmpeg_spectral adds mild spectral denoise.",
    )
    parser.add_argument(
        "--keep-stage",
        action="store_true",
        help="Keep the intermediate standardized WAV next to the final output.",
    )
    args = parser.parse_args()

    source = Path(args.audio).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    destination = Path(args.output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    stage_path = destination.with_name(f"{destination.stem}.standardized.wav")
    print(f"Preparing source: {source}", flush=True)
    print(f"Preset: {args.preset}", flush=True)

    extract_standard(source, stage_path)
    if args.preset == "standard":
        stage_path.replace(destination)
    else:
        apply_spectral(stage_path, destination)
        if not args.keep_stage:
            stage_path.unlink(missing_ok=True)

    print(f"Done. Output: {destination}", flush=True)
    return 0


def extract_standard(source: Path, destination: Path) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        "-af",
        STANDARD_FILTER,
        str(destination),
    ]
    run(command)


def apply_spectral(source: Path, destination: Path) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        "-af",
        SPECTRAL_FILTER,
        str(destination),
    ]
    run(command)


def run(command: list[str]) -> None:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())


if __name__ == "__main__":
    raise SystemExit(main())
