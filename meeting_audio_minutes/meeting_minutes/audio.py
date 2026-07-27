from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import time


DEEPFILTER_SAMPLE_RATE = 48000
DEFAULT_DEEPFILTER_TIMEOUT_SECONDS = 600
DEFAULT_FFMPEG_TIMEOUT_SECONDS = 1200
DEFAULT_DEEPFILTER_MAX_DURATION_SECONDS = 2700
STALE_DEEPFILTER_SECONDS = 300


@dataclass(frozen=True)
class PreparedAudio:
    path: Path
    filter_chain: str
    denoise_backend: str
    warning: str | None = None


class AudioPreprocessError(RuntimeError):
    """Raised when FFmpeg cannot prepare audio for local ASR."""


def ensure_ffmpeg() -> str:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise AudioPreprocessError(
            "FFmpeg was not found. Install it first, for example: brew install ffmpeg"
        )
    return ffmpeg_path


def build_filter_chain(denoise: bool) -> str:
    filters = [
        "highpass=f=80",
        "lowpass=f=7800",
    ]
    if denoise:
        filters.append("afftdn=nf=-25")
    filters.extend(
        [
            "dynaudnorm=f=150:g=15",
            "loudnorm=I=-16:LRA=11:TP=-1.5",
        ]
    )
    return ",".join(filters)


def build_normalization_chain() -> str:
    return "dynaudnorm=f=150:g=15,loudnorm=I=-16:LRA=11:TP=-1.5"


def preprocess_audio(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    denoise_backend: str = "auto",
    use_postfilter: bool = True,
    sample_rate: int = 16000,
    deepfilter_timeout_seconds: int = DEFAULT_DEEPFILTER_TIMEOUT_SECONDS,
    ffmpeg_timeout_seconds: int = DEFAULT_FFMPEG_TIMEOUT_SECONDS,
    deepfilter_max_duration_seconds: int = DEFAULT_DEEPFILTER_MAX_DURATION_SECONDS,
) -> PreparedAudio:
    source = Path(input_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Audio file not found: {source}")

    destination_dir = Path(output_dir).expanduser().resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{source.stem}.meeting_ready.wav"

    backend = normalize_backend(denoise_backend)
    if destination.exists() and destination.stat().st_size > 0:
        return PreparedAudio(
            path=destination,
            filter_chain="reused existing meeting_ready.wav",
            denoise_backend="reused",
        )

    if backend in {"auto", "deepfilter"}:
        try:
            if backend == "auto":
                duration_seconds = probe_duration_seconds(source)
                if (
                    duration_seconds is not None
                    and duration_seconds > max(1, int(deepfilter_max_duration_seconds))
                ):
                    raise AudioPreprocessError(
                        "Audio is too long for default CPU DeepFilterNet auto mode "
                        f"({duration_seconds:.0f}s > {deepfilter_max_duration_seconds}s)."
                    )
                stale_reason = stale_deepfilter_reason(source, destination_dir / "deepfilter")
                if stale_reason:
                    raise AudioPreprocessError(stale_reason)
            enhanced = enhance_with_deepfilter(
                source,
                destination_dir / "deepfilter",
                use_postfilter=use_postfilter,
                deepfilter_timeout_seconds=deepfilter_timeout_seconds,
                ffmpeg_timeout_seconds=ffmpeg_timeout_seconds,
            )
            post_model_chain = build_filter_chain(denoise=False)
            _run_ffmpeg(
                enhanced,
                destination,
                sample_rate,
                post_model_chain,
                timeout_seconds=ffmpeg_timeout_seconds,
            )
            return PreparedAudio(
                path=destination,
                filter_chain=f"deepfilter -> {post_model_chain}",
                denoise_backend="deepfilter",
            )
        except AudioPreprocessError as model_error:
            if backend == "deepfilter":
                raise
            warning = (
                "DeepFilterNet model denoise was unavailable or failed, so the "
                f"pipeline fell back to FFmpeg spectral denoise. Reason: "
                f"{_compact_error(str(model_error))}"
            )
        else:
            warning = None
    else:
        warning = None

    primary_chain = (
        build_normalization_chain()
        if backend == "none"
        else build_filter_chain(denoise=True)
    )
    try:
        _run_ffmpeg(
            source,
            destination,
            sample_rate,
            primary_chain,
            timeout_seconds=ffmpeg_timeout_seconds,
        )
        return PreparedAudio(
            path=destination,
            filter_chain=primary_chain,
            denoise_backend="ffmpeg" if backend != "none" else "none",
            warning=warning,
        )
    except AudioPreprocessError as primary_error:
        if backend == "none":
            raise

        fallback_chain = build_filter_chain(denoise=False)
        try:
            _run_ffmpeg(
                source,
                destination,
                sample_rate,
                fallback_chain,
                timeout_seconds=ffmpeg_timeout_seconds,
            )
        except AudioPreprocessError as fallback_error:
            raise AudioPreprocessError(
                f"{primary_error}\nFallback without denoise also failed: {fallback_error}"
            ) from fallback_error

        warning = (
            "The denoise filter failed, so audio was prepared with the fallback "
            f"filter chain. FFmpeg said: {_compact_error(str(primary_error))}"
        )
        return PreparedAudio(
            path=destination,
            filter_chain=fallback_chain,
            denoise_backend="ffmpeg-fallback",
            warning=warning,
        )


def normalize_backend(backend: str) -> str:
    normalized = backend.strip().lower()
    valid = {"auto", "deepfilter", "ffmpeg", "none"}
    if normalized not in valid:
        raise AudioPreprocessError(
            f"Unknown denoise backend: {backend}. Expected one of: {', '.join(sorted(valid))}"
        )
    return normalized


def enhance_with_deepfilter(
    source: Path,
    work_dir: Path,
    *,
    use_postfilter: bool,
    deepfilter_timeout_seconds: int,
    ffmpeg_timeout_seconds: int,
) -> Path:
    command = find_deepfilter_command()
    work_dir.mkdir(parents=True, exist_ok=True)

    model_input = work_dir / f"{source.stem}.deepfilter_input.wav"
    model_output_dir = work_dir / "enhanced"
    model_output_dir.mkdir(parents=True, exist_ok=True)

    _run_ffmpeg(
        source,
        model_input,
        DEEPFILTER_SAMPLE_RATE,
        "aresample=async=1:first_pts=0",
        timeout_seconds=ffmpeg_timeout_seconds,
    )

    if command.name == "deep-filter":
        args = [str(command), "-o", str(model_output_dir)]
    else:
        args = [str(command), "--output-dir", str(model_output_dir)]
    if use_postfilter:
        args.append("--pf")
    args.append(str(model_input))

    try:
        completed = subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=False,
            timeout=max(1, int(deepfilter_timeout_seconds)),
        )
    except subprocess.TimeoutExpired as error:
        raise AudioPreprocessError(
            f"DeepFilterNet timed out after {deepfilter_timeout_seconds}s"
        ) from error
    if completed.returncode != 0:
        details = completed.stderr.strip() or completed.stdout.strip()
        raise AudioPreprocessError(details or "DeepFilterNet failed with no details.")

    candidates = sorted(
        model_output_dir.glob("*.wav"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise AudioPreprocessError(
            f"DeepFilterNet finished but no enhanced wav was found in {model_output_dir}"
        )
    return candidates[0]


def stale_deepfilter_reason(source: Path, work_dir: Path) -> str | None:
    model_input = work_dir / f"{source.stem}.deepfilter_input.wav"
    model_output_dir = work_dir / "enhanced"
    if not model_input.exists() or model_input.stat().st_size <= 0:
        return None
    if any(model_output_dir.glob("*.wav")):
        return None
    age_seconds = time.time() - model_input.stat().st_mtime
    if age_seconds < STALE_DEEPFILTER_SECONDS:
        return None
    return (
        "Found a stale DeepFilterNet input wav from a previous run, but no enhanced "
        f"wav in {model_output_dir}. Skipping model denoise in auto mode."
    )


def probe_duration_seconds(source: Path) -> float | None:
    ffprobe_path = shutil.which("ffprobe")
    if not ffprobe_path:
        return None
    try:
        completed = subprocess.run(
            [
                ffprobe_path,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(source),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    text = (completed.stdout or "").strip()
    try:
        return float(text)
    except ValueError:
        return None


def find_deepfilter_command() -> Path:
    for name in ("deepFilter", "deep-filter"):
        found = shutil.which(name)
        if found:
            return Path(found)
    raise AudioPreprocessError(
        "DeepFilterNet command was not found. Install it with: "
        "pip install torch torchaudio deepfilternet"
    )


def _run_ffmpeg(
    source: Path,
    destination: Path,
    sample_rate: int,
    filter_chain: str,
    *,
    timeout_seconds: int = DEFAULT_FFMPEG_TIMEOUT_SECONDS,
) -> None:
    ffmpeg_path = ensure_ffmpeg()
    command = [
        ffmpeg_path,
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
        str(sample_rate),
        "-af",
        filter_chain,
        str(destination),
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=max(1, int(timeout_seconds)),
        )
    except subprocess.TimeoutExpired as error:
        raise AudioPreprocessError(f"FFmpeg timed out after {timeout_seconds}s") from error
    if completed.returncode != 0:
        details = completed.stderr.strip() or completed.stdout.strip()
        raise AudioPreprocessError(details or "Unknown FFmpeg error")


def _compact_error(message: str, *, limit: int = 240) -> str:
    message = " ".join(message.split())
    if len(message) <= limit:
        return message
    return f"{message[:limit].rstrip()}..."
