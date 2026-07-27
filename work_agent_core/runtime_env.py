from __future__ import annotations

from pathlib import Path
import os
import sys


PROJECT_VENV_DIR = ".venv"
# Compatibility name for callers that imported the old constant. There is no
# second "agent" environment anymore.
AGENT_VENV_DIR = PROJECT_VENV_DIR


def runtime_bin_directories(workspace_root: str | Path | None = None) -> list[Path]:
    """Return deterministic executable search roots for the managed service.

    launchd starts jobs with a minimal PATH, so Homebrew and the bundled Codex
    runtime are not visible unless we add them explicitly.
    """
    root = Path(workspace_root or Path.cwd()).resolve()
    candidates = [
        root / PROJECT_VENV_DIR / "bin",
        Path(os.environ["WORK_AGENT_RUNTIME_BIN"]).expanduser()
        if os.environ.get("WORK_AGENT_RUNTIME_BIN")
        else None,
        Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/bin",
        Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/bin/override",
        Path("/opt/homebrew/bin"),
        Path("/usr/local/bin"),
        Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin",
    ]
    result: list[Path] = []
    for candidate in candidates:
        if candidate is None or not candidate.is_dir() or candidate in result:
            continue
        result.append(candidate)
    return result


def runtime_search_path(workspace_root: str | Path | None = None, current_path: str = "") -> str:
    entries = [str(path) for path in runtime_bin_directories(workspace_root)]
    entries.extend(part for part in current_path.split(os.pathsep) if part)
    return os.pathsep.join(dict.fromkeys(entries))


def find_runtime_executable(name: str, workspace_root: str | Path | None = None) -> str | None:
    from shutil import which

    return which(name, path=runtime_search_path(workspace_root, os.environ.get("PATH", "")))


def project_agent_python(workspace_root: str | Path | None = None) -> Path | None:
    """Return the one supported project interpreter.

    Deliberately do not fall back to system Python, the Codex runtime, Conda,
    or a skill-local venv. A missing `.venv` is a setup error, not a reason to
    silently execute with a different dependency set.
    """
    root = Path(workspace_root or Path.cwd()).resolve()
    candidates = [
        root / PROJECT_VENV_DIR / "bin" / "python",
        root / PROJECT_VENV_DIR / "Scripts" / "python.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.absolute()
    return None


def project_node(workspace_root: str | Path | None = None) -> Path | None:
    """Return the managed Node executable exposed to project commands."""
    root = Path(workspace_root or Path.cwd()).resolve()
    requested = os.getenv("WORK_AGENT_NODE", "").strip()
    candidates = [
        Path(requested).expanduser() if requested else None,
        Path("/opt/homebrew/bin/node"),
        Path("/usr/local/bin/node"),
        Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate.absolute()
    return None


def apply_project_agent_environment(
    environment: dict[str, str],
    workspace_root: str | Path,
) -> dict[str, str]:
    result = dict(environment)
    python_path = project_agent_python(workspace_root)
    if python_path is None:
        return result

    bin_dir = python_path.parent
    result["PATH"] = runtime_search_path(workspace_root, result.get("PATH", ""))
    result["VIRTUAL_ENV"] = str(bin_dir.parent)
    result["WORK_AGENT_PYTHON"] = str(python_path)
    result["WORK_AGENT_OFFICE_PYTHON"] = str(python_path)
    node_path = project_node(workspace_root)
    if node_path is not None:
        result["WORK_AGENT_NODE"] = str(node_path)
    result["PIP_REQUIRE_VIRTUALENV"] = "true"
    result["PYTHONNOUSERSITE"] = "1"
    return result


def runtime_contract_status(workspace_root: str | Path | None = None) -> dict[str, object]:
    """Return a small, user-visible snapshot of the active runtime contract."""
    root = Path(workspace_root or Path.cwd()).resolve()
    python_path = project_agent_python(root)
    node_path = project_node(root)
    current_python = Path(sys.executable).absolute()
    legacy_paths = [
        root / ".venv_agent",
        root / "meeting_audio_minutes" / ".venv_project",
        root / "meeting_audio_minutes" / ".venv_deepfilter",
    ]
    return {
        "contract": "single-project-runtime-v1",
        "python": {
            "path": str(python_path) if python_path else None,
            "ready": bool(python_path),
            "active": bool(
                python_path
                and current_python.resolve() == python_path.resolve()
            ),
            "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        },
        "node": {
            "path": str(node_path) if node_path else None,
            "ready": bool(node_path),
        },
        "native_tools": {
            name: find_runtime_executable(name, root)
            for name in ("ffmpeg", "soffice", "pdftoppm")
        },
        "legacy_environments": [
            str(path.relative_to(root)) for path in legacy_paths if path.exists()
        ],
    }


__all__ = [
    "AGENT_VENV_DIR",
    "apply_project_agent_environment",
    "find_runtime_executable",
    "project_agent_python",
    "project_node",
    "PROJECT_VENV_DIR",
    "runtime_contract_status",
    "runtime_bin_directories",
    "runtime_search_path",
]
