from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import subprocess

import noisereduce as nr
import numpy as np
import soundfile as sf


SAMPLE_RATE = 16000


@dataclass(frozen=True)
class SegmentSpec:
    label: str
    start_seconds: int


@dataclass(frozen=True)
class NoiseReduceConfig:
    label: str
    stationary: bool
    prop_decrease: float
    n_std_thresh_stationary: float = 1.5
    time_constant_s: float = 2.0


SEGMENTS = [
    SegmentSpec("00_start", 0),
    SegmentSpec("10_min", 10 * 60),
    SegmentSpec("30_min", 30 * 60),
]

CONFIGS = [
    NoiseReduceConfig(
        label="01_light_nonstationary",
        stationary=False,
        prop_decrease=0.72,
        time_constant_s=3.0,
    ),
    NoiseReduceConfig(
        label="02_medium_nonstationary",
        stationary=False,
        prop_decrease=0.86,
        time_constant_s=2.0,
    ),
    NoiseReduceConfig(
        label="03_strong_stationary",
        stationary=True,
        prop_decrease=0.92,
        n_std_thresh_stationary=1.25,
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create denoise trial clips from a long meeting recording."
    )
    parser.add_argument("audio", help="Source meeting recording.")
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
    output_root = Path(args.output_dir).expanduser().resolve() / f"{source.stem}_samples"
    raw_dir = output_root / "raw"
    wav_dir = output_root / "wav"
    listen_dir = output_root / "listen_m4a"
    for directory in (raw_dir, wav_dir, listen_dir):
        directory.mkdir(parents=True, exist_ok=True)

    for segment in SEGMENTS:
        raw_wav = raw_dir / f"{segment.label}_raw.wav"
        extract_clip(source, raw_wav, segment.start_seconds, args.clip_seconds)
        audio, sr = sf.read(raw_wav, dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sr != SAMPLE_RATE:
            raise RuntimeError(f"Expected {SAMPLE_RATE} Hz clip, got {sr} Hz: {raw_wav}")

        encode_listen_copy(raw_wav, listen_dir / f"{segment.label}_00_raw.m4a")
        ffmpeg_spectral = wav_dir / f"{segment.label}_04_ffmpeg_spectral.wav"
        run_ffmpeg_filter(raw_wav, ffmpeg_spectral)
        encode_listen_copy(ffmpeg_spectral, listen_dir / f"{segment.label}_04_ffmpeg_spectral.m4a")

        for config in CONFIGS:
            reduced = reduce_noise(audio, SAMPLE_RATE, config)
            wav_path = wav_dir / f"{segment.label}_{config.label}.wav"
            sf.write(wav_path, reduced, SAMPLE_RATE, subtype="PCM_16")
            encode_listen_copy(wav_path, listen_dir / f"{segment.label}_{config.label}.m4a")

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
        "-af",
        "highpass=f=75,lowpass=f=7600,dynaudnorm=f=150:g=12,loudnorm=I=-18:LRA=11:TP=-1.5",
        str(destination),
    ]
    run(command)


def reduce_noise(audio: np.ndarray, sample_rate: int, config: NoiseReduceConfig) -> np.ndarray:
    if config.stationary:
        reduced = nr.reduce_noise(
            y=audio,
            sr=sample_rate,
            stationary=True,
            prop_decrease=config.prop_decrease,
            n_std_thresh_stationary=config.n_std_thresh_stationary,
        )
    else:
        reduced = nr.reduce_noise(
            y=audio,
            sr=sample_rate,
            stationary=False,
            prop_decrease=config.prop_decrease,
            time_constant_s=config.time_constant_s,
            freq_mask_smooth_hz=500,
            time_mask_smooth_ms=80,
        )
    return np.asarray(np.clip(reduced, -1.0, 1.0), dtype=np.float32)


def run_ffmpeg_filter(source: Path, destination: Path) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-af",
        "afftdn=nf=-25,dynaudnorm=f=150:g=12,loudnorm=I=-18:LRA=11:TP=-1.5",
        str(destination),
    ]
    run(command)


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


def run(command: list[str]) -> None:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())


if __name__ == "__main__":
    raise SystemExit(main())
