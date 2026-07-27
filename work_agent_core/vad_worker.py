from __future__ import annotations

import base64
import json
import sys
import warnings
from typing import Any

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

try:
    import webrtcvad
except Exception as import_error:  # pragma: no cover - exercised by caller fallback
    print(
        json.dumps(
            {
                "event": "error",
                "ok": False,
                "error": f"webrtcvad import failed: {import_error}",
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    raise


def main() -> int:
    vad_cache: dict[int, webrtcvad.Vad] = {}
    print(
        json.dumps(
            {
                "event": "ready",
                "ok": True,
                "provider": "webrtcvad",
                "version": getattr(webrtcvad, "__version__", "unknown"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            if request.get("event") == "shutdown":
                return 0
            response = classify_request(request, vad_cache)
        except Exception as error:
            response = {
                "event": "result",
                "id": request.get("id") if isinstance(request, dict) else None,
                "ok": False,
                "error": str(error),
            }
        print(json.dumps(response, ensure_ascii=False), flush=True)
    return 0


def classify_request(request: dict[str, Any], vad_cache: dict[int, webrtcvad.Vad]) -> dict[str, Any]:
    request_id = str(request.get("id") or "")
    sample_rate = int(request.get("sample_rate") or 16000)
    frame_ms = int(request.get("frame_ms") or 30)
    aggressiveness = int(request.get("aggressiveness") or 3)
    frames_base64 = request.get("frames_base64")

    if sample_rate not in {8000, 16000, 32000, 48000}:
        raise ValueError("sample_rate must be one of 8000, 16000, 32000, 48000.")
    if frame_ms not in {10, 20, 30}:
        raise ValueError("frame_ms must be one of 10, 20, 30.")
    if aggressiveness < 0 or aggressiveness > 3:
        raise ValueError("aggressiveness must be between 0 and 3.")
    if not isinstance(frames_base64, list) or not frames_base64:
        raise ValueError("frames_base64 must be a non-empty list.")
    if len(frames_base64) > 100:
        raise ValueError("Too many VAD frames in one request.")

    vad = vad_cache.get(aggressiveness)
    if vad is None:
        vad = webrtcvad.Vad(aggressiveness)
        vad_cache[aggressiveness] = vad

    expected_bytes = int(sample_rate * frame_ms / 1000) * 2
    speech_frames: list[bool] = []
    for encoded in frames_base64:
        if not isinstance(encoded, str):
            raise ValueError("Each VAD frame must be base64 text.")
        frame = base64.b64decode(encoded, validate=True)
        if len(frame) != expected_bytes:
            raise ValueError(
                f"Invalid frame size: got {len(frame)} bytes, expected {expected_bytes}."
            )
        speech_frames.append(bool(vad.is_speech(frame, sample_rate)))

    return {
        "event": "result",
        "id": request_id,
        "ok": True,
        "provider": "webrtcvad",
        "sample_rate": sample_rate,
        "frame_ms": frame_ms,
        "speech_frames": speech_frames,
        "speech_count": sum(1 for item in speech_frames if item),
    }


if __name__ == "__main__":
    raise SystemExit(main())
