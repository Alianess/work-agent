from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import os
import shutil
import subprocess


SAMPLE_RATE = 48000


@dataclass(frozen=True)
class SegmentSpec:
    label: str
    start_seconds: int


@dataclass(frozen=True)
class DeepFilterConfig:
    label: str
    args: tuple[str, ...]


SEGMENTS = [
    SegmentSpec("00_start", 0),
    SegmentSpec("10_min", 10 * 60),
    SegmentSpec("30_min", 30 * 60),
]

CONFIGS = [
    DeepFilterConfig("05_dfnet3_default", ()),
    DeepFilterConfig("06_dfnet3_pf", ("--pf",)),
    DeepFilterConfig("07_dfnet3_pf_atten12", ("--pf", "--atten-lim", "12")),
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create DeepFilterNet denoise trial clips from a long meeting recording."
    )
    parser.add_argument("audio", help="Source meeting recording.")
    parser.add_argument(
        "--deepfilter-bin",
        required=True,
        help="Path to the project-local deepFilter executable.",
    )
    parser.add_argument(
        "--model-base-dir",
        default="DeepFilterNet3",
        help=(
            "DeepFilterNet model name or local model directory. "
            "Use a local directory to avoid network downloads."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="meet_files/denoise_trials",
        help="Directory for trial clips.",
    )
    parser.add_argument(
        "--clip-seconds",
        type=int,
        default=75,
        help="Length of each trial clip.",
    )
    args = parser.parse_args()

    source = Path(args.audio).expanduser().resolve()
    deepfilter_bin = Path(args.deepfilter_bin).expanduser().resolve()
    if not deepfilter_bin.exists():
        raise FileNotFoundError(f"deepFilter executable not found: {deepfilter_bin}")
    deepfilter_python = deepfilter_bin.with_name("python")
    wrapper_script = Path(__file__).with_name("deepfilter_enhance_single.py").resolve()
    if not deepfilter_python.exists():
        raise FileNotFoundError(f"DeepFilterNet python not found: {deepfilter_python}")
    if not wrapper_script.exists():
        raise FileNotFoundError(f"DeepFilterNet wrapper not found: {wrapper_script}")

    output_root = Path(args.output_dir).expanduser().resolve() / f"{source.stem}_samples"
    raw_dir = output_root / "deepfilter_raw_48k"
    enhanced_root = output_root / "deepfilter_wav"
    listen_dir = output_root / "listen_m4a"
    cache_home = output_root / ".deepfilter_home"
    for directory in (raw_dir, enhanced_root, listen_dir, cache_home):
        directory.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["HOME"] = str(cache_home)
    env["XDG_CACHE_HOME"] = str(cache_home / ".cache")
    env.pop("HTTP_PROXY", None)
    env.pop("HTTPS_PROXY", None)
    env.pop("ALL_PROXY", None)
    env.pop("http_proxy", None)
    env.pop("https_proxy", None)
    env.pop("all_proxy", None)

    model_base_dir = str(Path(args.model_base_dir).expanduser().resolve()) if Path(args.model_base_dir).expanduser().exists() else args.model_base_dir

    for segment in SEGMENTS:
        raw_wav = raw_dir / f"{segment.label}_raw_48k.wav"
        extract_clip(source, raw_wav, segment.start_seconds, args.clip_seconds)
        encode_listen_copy(raw_wav, listen_dir / f"{segment.label}_00_raw_48k.m4a")

        for config in CONFIGS:
            config_dir = enhanced_root / segment.label / config.label
            config_dir.mkdir(parents=True, exist_ok=True)
            command = [
                str(deepfilter_python),
                str(wrapper_script),
                "--model-base-dir",
                model_base_dir,
                "--output-dir",
                str(config_dir),
                "--no-suffix",
                "--log-level",
                "info",
                *config.args,
                str(raw_wav),
            ]
            run(command, env=env)
            enhanced_wav = find_enhanced_wav(config_dir, raw_wav.name)
            canonical_wav = enhanced_root / f"{segment.label}_{config.label}.wav"
            shutil.copy2(enhanced_wav, canonical_wav)
            encode_listen_copy(canonical_wav, listen_dir / f"{segment.label}_{config.label}.m4a")

    print(f"Done. Listen clips: {listen_dir}")
    return 0


def extract_clip(source: Path, destination: Path, start_seconds: int, clip_seconds: int) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        str(start_seconds),
        "-t",
        str(clip_seconds),
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        str(destination),
    ]
    run(command)


def find_enhanced_wav(output_dir: Path, raw_name: str) -> Path:
    preferred = output_dir / raw_name
    if preferred.exists():
        return preferred
    candidates = sorted(output_dir.glob("*.wav"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No enhanced wav found in {output_dir}")
    return candidates[0]


def encode_listen_copy(source: Path, destination: Path) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-c:a",
        "aac",
        "-b:a",
        "96k",
        str(destination),
    ]
    run(command)


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    completed = subprocess.run(command, capture_output=True, text=True, check=False, env=env)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())


if __name__ == "__main__":
    raise SystemExit(main())
