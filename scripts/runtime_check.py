from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
import json
import shutil
import sys


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PYTHON = WORKSPACE_ROOT / ".venv" / "bin" / "python"
REQUIRED_MODULES = (
    "defusedxml",
    "docx",
    "funasr",
    "lxml",
    "mlx_audio",
    "openpyxl",
    "pdfplumber",
    "pptx",
    "pypdf",
    "reportlab",
    "requests",
    "torch",
    "webrtcvad",
    "yaml",
)
REQUIRED_BINARIES = ("ffmpeg", "node", "npm", "pdftoppm", "soffice")
LEGACY_ENVIRONMENTS = (
    WORKSPACE_ROOT / ".venv_agent",
    WORKSPACE_ROOT / "meeting_audio_minutes" / ".venv_project",
    WORKSPACE_ROOT / "meeting_audio_minutes" / ".venv_deepfilter",
)


def main() -> int:
    modules = {name: find_spec(name) is not None for name in REQUIRED_MODULES}
    binaries = {name: shutil.which(name) for name in REQUIRED_BINARIES}
    active = Path(sys.executable).resolve() == EXPECTED_PYTHON.resolve()
    legacy = [
        str(path.relative_to(WORKSPACE_ROOT))
        for path in LEGACY_ENVIRONMENTS
        if path.exists()
    ]
    payload = {
        "ok": active and all(modules.values()) and all(binaries.values()),
        "contract": "single-project-runtime-v1",
        "python": {
            "expected": str(EXPECTED_PYTHON),
            "actual": sys.executable,
            "version": sys.version.split()[0],
            "active": active,
        },
        "modules": modules,
        "binaries": binaries,
        "legacy_environments": legacy,
        "note": (
            "legacy_environments 只用于迁移审计，不会加入 PATH；"
            "DeepFilterNet 已退出主链路，统一使用 FFmpeg 降噪。"
        ),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
