"""Robust skill manifest loading, tool discovery, and environment precheck.

This module is the foundation layer for skill integration. It is deliberately
free of runtime-execution concerns: it only *reads* skill folders and produces
structured manifests. Execution stays in `skill_runtime.SkillRuntime`.

Design goals (so adding a new skill never requires editing core code):

1. A skill declares native tools via either
   - `work_agent.json` -> `tools[]` (centralized), or
   - `scripts/<name>.tool.json` next to each script (drop-in).
2. Frontmatter parsing is a real YAML-subset parser (no PyYAML dependency):
   it handles multi-line `description`, block lists, inline lists, and quoted
   strings. Pure `key: value` skills keep working unchanged.
3. A startup health check summarizes, per skill: which tools registered, which
   were skipped and why, and which declared script files are missing.
4. `probe_skill_environment` returns, per skill, the availability of the
   binaries / Python modules it declares as dependencies, so the agent can
   pick execution paths adaptively instead of guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
import json
import os
import re
import shutil
import subprocess
import sys

from .runtime_env import find_runtime_executable, project_agent_python


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

SKILLS_DIR = Path("work_agent_skills")
EXTRA_SKILL_DIRS: list[Path] = [Path("meeting_audio_minutes/skills")]
DEFAULT_MAX_SKILL_CHARS = 20000

SCRIPT_SUFFIXES = {".py", ".js", ".sh", ".ps1"}
HEALTH_REPORT_NAME = "skill_health.json"
HEALTH_REPORT_DIR = Path("meet_files") / "skill_reports"

# Conventional dependency declarations a skill may carry. A skill may put a
# `dependencies` block in `work_agent.json`:
#   "dependencies": {
#     "python_modules": ["openpyxl", "python-docx", "pypdf"],
#     "binaries": ["libreoffice", "pandoc", "pdftoppm"],
#     "node_modules": ["docx", "pptxgenjs"],
#     "paths": [".venv/bin/python"]
#   }
DEFAULT_DEPENDENCY_KEYS = ("python_modules", "binaries", "node_modules", "paths")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DeclaredTool:
    """A tool declared by a skill, ready to be registered as a native tool."""

    name: str
    description: str
    parameters: dict[str, Any]
    execution: dict[str, Any]
    skill_id: str
    script_path: str
    source: str  # "work_agent.json" | "script-manifest"


@dataclass(frozen=True)
class SkillToolRegistration:
    """Result of attempting to register one skill-declared tool."""

    skill_id: str
    tool_name: str
    registered: bool
    reason: str = ""
    source: str = ""
    script_path: str = ""


@dataclass(frozen=True)
class SkillHealthReport:
    """Per-skill summary produced by `build_skill_health_report`."""

    skill_id: str
    path: str
    ok: bool
    tools: list[SkillToolRegistration]
    missing_scripts: list[str]
    issues: list[str]

    def to_payload(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "path": self.path,
            "ok": self.ok,
            "tools": [
                {
                    "tool_name": t.tool_name,
                    "registered": t.registered,
                    "reason": t.reason,
                    "source": t.source,
                    "script_path": t.script_path,
                }
                for t in self.tools
            ],
            "missing_scripts": self.missing_scripts,
            "issues": self.issues,
        }


# ---------------------------------------------------------------------------
# Frontmatter parser (YAML subset, no PyYAML dependency)
# ---------------------------------------------------------------------------

def parse_skill_markdown(text: str) -> tuple[dict[str, Any], str]:
    """Parse the YAML frontmatter block of a SKILL.md file.

    Returns (frontmatter_dict, body). Supports:
      - scalar `key: value` pairs (with optional quotes)
      - multi-line scalars via `key: |` or `key: >` block scalars
      - block lists (`- item` lines under a `key:` with no value)
      - inline lists `[a, b, c]`
      - inline dicts `{a: 1, b: 2}` (shallow)
      - simple nested mappings (one level of indent)

    Falls back gracefully: if anything is malformed, the offending key keeps
    its raw string value rather than crashing the whole skill load.
    """
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    raw_frontmatter = parts[1]
    body = parts[2].lstrip()
    try:
        return _parse_yaml_block(raw_frontmatter), body
    except Exception:
        # Last-resort: key:value line scan so a malformed skill never breaks
        # the whole registry.
        data: dict[str, Any] = {}
        for line in raw_frontmatter.splitlines():
            if ":" not in line or line.lstrip().startswith("-"):
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                data[key] = value
        return data, body


def _parse_yaml_block(raw: str) -> dict[str, Any]:
    lines = [ln for ln in raw.splitlines()]
    data: dict[str, Any] = {}
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            index += 1
            continue
        if stripped.startswith("-"):
            # A stray list item with no key above it; skip.
            index += 1
            continue
        if ":" not in stripped:
            index += 1
            continue
        key, value = _split_kv(stripped)
        key = key.strip()
        value = value.strip()
        if value == "":
            # Could be a block list or nested mapping. Look ahead.
            collected, next_index = _collect_block(lines, index + 1, indent=_indent_of(line))
            if isinstance(collected, list) and collected:
                data[key] = collected
            elif isinstance(collected, dict) and collected:
                data[key] = collected
            else:
                data[key] = ""
            index = next_index
            continue
        if value in ("|", "|-", "|+", ">", ">-", ">+"):
            block, next_index = _collect_block_scalar(lines, index + 1, indent=_indent_of(line), folded=(value[0] == ">"))
            data[key] = block
            index = next_index
            continue
        data[key] = _parse_scalar(value)
        index += 1
    return data


def _split_kv(line: str) -> tuple[str, str]:
    # Split on the first colon that is not inside quotes.
    in_single = in_double = False
    for i, ch in enumerate(line):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == ":" and not in_single and not in_double:
            return line[:i], line[i + 1 :]
    return line, ""


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _collect_block(lines: list[str], start: int, indent: int) -> tuple[Any, int]:
    """Collect a block list or nested mapping starting at `start`."""
    items: list[Any] = []
    mapping: dict[str, Any] = {}
    index = start
    is_list = False
    is_mapping = False
    while index < len(lines):
        line = lines[index]
        if not line.strip() or line.strip().startswith("#"):
            index += 1
            continue
        if _indent_of(line) <= indent:
            break
        stripped = line.strip()
        if stripped.startswith("- "):
            is_list = True
            item_value = stripped[2:].strip()
            if ":" in item_value and not (item_value.startswith("[") or item_value.startswith("{")):
                k, v = _split_kv(item_value)
                mapping[_parse_scalar(k.strip())] = _parse_scalar(v.strip()) if v.strip() else ""
                # list of dicts not supported here; keep as mapping capture
            else:
                items.append(_parse_scalar(item_value))
            index += 1
            continue
        if ":" in stripped:
            is_mapping = True
            k, v = _split_kv(stripped)
            mapping[k.strip()] = _parse_scalar(v.strip()) if v.strip() else ""
            index += 1
            continue
        # Non-list, non-mapping indented line: treat as continuation of a prior scalar.
        index += 1
    if is_list:
        return items, index
    if is_mapping:
        return mapping, index
    return None, index


def _collect_block_scalar(lines: list[str], start: int, indent: int, *, folded: bool) -> tuple[str, int]:
    buffer: list[str] = []
    index = start
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            buffer.append("")
            index += 1
            continue
        if _indent_of(line) <= indent:
            break
        buffer.append(line.strip())
        index += 1
    text = "\n".join(buffer).strip()
    if folded:
        text = re.sub(r"\n+", " ", text)
    return text, index


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in _split_top_level(inner, ",")]
    if value.startswith("{") and value.endswith("}"):
        inner = value[1:-1].strip()
        if not inner:
            return {}
        result: dict[str, Any] = {}
        for part in _split_top_level(inner, ","):
            if ":" in part:
                k, v = _split_kv(part)
                result[k.strip()] = _parse_scalar(v.strip())
        return result
    if value.lower() in ("true", "yes"):
        return True
    if value.lower() in ("false", "no"):
        return False
    if value.lower() in ("null", "~"):
        return None
    # int?
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?\d+\.\d+", value):
        return float(value)
    return value


def _split_top_level(text: str, sep: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    in_single = in_double = False
    buf: list[str] = []
    for ch in text:
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch in "[" and not in_single and not in_double:
            depth += 1
        elif ch == "]" and not in_single and not in_double:
            depth -= 1
        elif ch == sep and depth == 0 and not in_single and not in_double:
            parts.append("".join(buf))
            buf = []
            continue
        buf.append(ch)
    if buf:
        parts.append("".join(buf))
    return parts


# ---------------------------------------------------------------------------
# Tool manifest discovery
# ---------------------------------------------------------------------------

def discover_declared_tools(skill_dir: Path, workspace_root: Path) -> list[tuple[dict[str, Any], str, str]]:
    """Return [(tool_decl, source, script_path)] for a skill.

    Sources, in priority order:
      1. `work_agent.json` -> `tools[]` entries (each with `execution.script_path`)
      2. `scripts/<name>.tool.json` sidecar manifests (drop-in discovery)

    Sidecar manifests may be a single object or a list of tool declarations,
    and each declaration must have at least `name`, `description`, `parameters`,
    and `execution.script_path`. Duplicates by tool name within a skill are
    de-duplicated (first wins).
    """
    declarations: list[tuple[dict[str, Any], str, str]] = []
    seen_names: set[str] = set()

    # 1. Centralized tools[] in work_agent.json
    config = read_skill_config(skill_dir)
    raw_tools = config.get("tools")
    if isinstance(raw_tools, list):
        for raw_tool in raw_tools:
            if not isinstance(raw_tool, dict):
                continue
            name = str(raw_tool.get("name") or "").strip()
            if not name or name in seen_names:
                continue
            script_path = _extract_script_path(raw_tool)
            if script_path:
                seen_names.add(name)
                declarations.append((raw_tool, "work_agent.json", script_path))

    # 2. Sidecar *.tool.json next to scripts
    scripts_dir = skill_dir / "scripts"
    if scripts_dir.is_dir():
        for sidecar in sorted(scripts_dir.rglob("*.tool.json")):
            try:
                data = json.loads(sidecar.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            tool_list = data if isinstance(data, list) else [data]
            for tool_decl in tool_list:
                if not isinstance(tool_decl, dict):
                    continue
                name = str(tool_decl.get("name") or "").strip()
                if not name or name in seen_names:
                    continue
                script_path = _extract_script_path(tool_decl)
                if not script_path:
                    # Default: the sidecar sits next to a script of the same stem.
                    stem = sidecar.stem[:-5] if sidecar.name.endswith(".tool.json") else sidecar.stem
                    candidate = _find_script_by_stem(scripts_dir, stem)
                    if candidate:
                        relative = candidate.relative_to(skill_dir).as_posix()
                        execution = dict(tool_decl.get("execution") or {})
                        execution.setdefault("type", "script")
                        execution.setdefault("script_path", relative)
                        tool_decl = {**tool_decl, "execution": execution}
                        script_path = relative
                if script_path:
                    seen_names.add(name)
                    declarations.append((tool_decl, "script-manifest", script_path))

    return declarations


def _extract_script_path(tool_decl: dict[str, Any]) -> str:
    execution = tool_decl.get("execution")
    if isinstance(execution, dict):
        path = str(execution.get("script_path") or "").strip()
        if path:
            return path
    return ""


def _find_script_by_stem(scripts_dir: Path, stem: str) -> Path | None:
    for suffix in SCRIPT_SUFFIXES:
        candidate = scripts_dir / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def read_skill_config(skill_dir: Path) -> dict[str, Any]:
    config_path = skill_dir / "work_agent.json"
    if not config_path.is_file():
        return {}
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


# ---------------------------------------------------------------------------
# Skill manifest loading (frontmatter + config + discovered tools)
# ---------------------------------------------------------------------------

def skill_roots(root: Path) -> list[Path]:
    return [root / SKILLS_DIR, *[root / path for path in EXTRA_SKILL_DIRS]]


def load_skill_manifests(workspace_root: str | Path) -> list["SkillManifest"]:
    root = Path(workspace_root).resolve()
    manifests: list[SkillManifest] = []
    for skills_dir in skill_roots(root):
        if not skills_dir.is_dir():
            continue
        for skill_dir in sorted(path for path in skills_dir.iterdir() if path.is_dir()):
            manifest = load_single_skill_manifest(root, skill_dir)
            if manifest is not None:
                manifests.append(manifest)
    return manifests


def load_single_skill_manifest(root: Path, skill_dir: Path) -> "SkillManifest | None":
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.is_file():
        return None
    frontmatter, _ = parse_skill_markdown(skill_md.read_text(encoding="utf-8", errors="replace"))
    config = read_skill_config(skill_dir)
    skill_id = str(config.get("id") or frontmatter.get("name") or skill_dir.name).strip()
    if not skill_id:
        return None
    label = str(config.get("label") or humanize_skill_id(skill_id))
    description = str(config.get("description") or frontmatter.get("description") or "").strip()
    when_to_use = str(config.get("when_to_use") or description)
    outputs = [str(item) for item in config.get("outputs") or []]
    native_tools = summarize_declared_skill_tools(config.get("tools"))
    return SkillManifest(
        id=skill_id,
        label=label,
        mention=str(config.get("mention") or f"@{label}"),
        description=description,
        when_to_use=when_to_use,
        tool_name=str(config.get("tool_name") or "") or None,
        outputs=outputs,
        source_url=str(config.get("source_url") or "") or None,
        path=str(skill_dir.relative_to(root)),
        default_enabled=bool(config.get("default_enabled", False)),
        native_tools=native_tools,
    )


def summarize_declared_skill_tools(raw_tools: Any) -> list[dict[str, str]]:
    summaries: list[dict[str, str]] = []
    if not isinstance(raw_tools, list):
        return summaries
    for raw_tool in raw_tools:
        if not isinstance(raw_tool, dict):
            continue
        name = str(raw_tool.get("name") or "").strip()
        if not name:
            continue
        summaries.append(
            {
                "name": name,
                "description": str(raw_tool.get("description") or "").strip(),
            }
        )
    return summaries


# ---------------------------------------------------------------------------
# Tool name + declaration validation
# ---------------------------------------------------------------------------

_TOOL_NAME_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")


def is_valid_tool_name(name: str) -> bool:
    return bool(_TOOL_NAME_RE.fullmatch(name or ""))


def validate_tool_declaration(tool_decl: dict[str, Any], skill_id: str) -> tuple[bool, str]:
    """Return (ok, reason). ok=False means the tool must be skipped."""
    name = str(tool_decl.get("name") or "").strip()
    if not name:
        return False, "missing name"
    if not is_valid_tool_name(name):
        return False, f"invalid tool name: {name!r}"
    description = str(tool_decl.get("description") or "").strip()
    if not description:
        return False, f"{skill_id}.{name}: missing description"
    parameters = tool_decl.get("parameters")
    if not isinstance(parameters, dict):
        return False, f"{skill_id}.{name}: parameters must be an object"
    execution = tool_decl.get("execution")
    if not isinstance(execution, dict):
        return False, f"{skill_id}.{name}: execution must be an object"
    exec_type = str(execution.get("type") or "script").strip().lower()
    if exec_type != "script":
        return False, f"{skill_id}.{name}: unsupported execution type {exec_type!r}"
    script_path = str(execution.get("script_path") or "").strip()
    if not script_path:
        return False, f"{skill_id}.{name}: missing execution.script_path"
    return True, ""


# ---------------------------------------------------------------------------
# Environment precheck (dependency probe)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SkillEnvironmentProbe:
    skill_id: str
    python_modules: dict[str, bool]
    binaries: dict[str, bool]
    node_modules: dict[str, bool]
    paths: dict[str, bool]
    declared: bool
    ready: bool

    def to_payload(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "python_modules": self.python_modules,
            "binaries": self.binaries,
            "node_modules": self.node_modules,
            "paths": self.paths,
            "declared": self.declared,
            "ready": self.ready,
        }


def probe_skill_environment(
    workspace_root: str | Path,
    *,
    skill_id: str | None = None,
) -> dict[str, Any]:
    """Probe declared dependencies for one or all skills.

    Reads `dependencies` from each skill's `work_agent.json`:
      {
        "python_modules": ["openpyxl", "python-docx"],
        "binaries": ["libreoffice", "pandoc"],
        "node_modules": ["docx", "pptxgenjs"],
        "paths": [".venv/bin/python"]
      }
    Python modules are imported under the office_python interpreter when set,
    so the result reflects the interpreter skills actually run under.
    """
    root = Path(workspace_root).resolve()
    python_exe = office_python()
    probes: list[SkillEnvironmentProbe] = []
    manifests = load_skill_manifests(root)
    if skill_id:
        manifests = [m for m in manifests if m.id == skill_id]
        if not manifests:
            return {"ok": False, "error": f"未知技能：{skill_id}"}
    for manifest in manifests:
        skill_dir = root / manifest.path
        deps = read_skill_config(skill_dir).get("dependencies") or {}
        if not isinstance(deps, dict):
            deps = {}
        python_modules = _probe_python_modules(deps.get("python_modules") or [], python_exe)
        binaries = _probe_binaries(deps.get("binaries") or [], root)
        node_modules = _probe_node_modules(deps.get("node_modules") or [])
        paths = _probe_paths(deps.get("paths") or [], root)
        declared = any(key in deps for key in DEFAULT_DEPENDENCY_KEYS)
        ready = declared and all(
            all(values.values()) for values in (python_modules, binaries, node_modules, paths)
        )
        probes.append(
            SkillEnvironmentProbe(
                skill_id=manifest.id,
                python_modules=python_modules,
                binaries=binaries,
                node_modules=node_modules,
                paths=paths,
                declared=declared,
                ready=ready,
            )
        )
    return {"ok": True, "python_executable": str(python_exe), "skills": [p.to_payload() for p in probes]}


def _probe_python_modules(modules: list[str], python_exe: Path) -> dict[str, bool]:
    result: dict[str, bool] = {}
    for module in modules:
        name = str(module).strip()
        if not name:
            continue
        # docx -> python-docx import name normalization handled by the skill.
        check_name = name.split("==")[0].split(">=")[0].split("<=")[0].strip()
        try:
            code = f"import importlib.util; print('ok' if importlib.util.find_spec({check_name!r}) else 'no')"
            completed = subprocess.run(
                [str(python_exe), "-c", code],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            result[name] = completed.returncode == 0 and "ok" in (completed.stdout or "")
        except Exception:
            result[name] = False
    return result


def _probe_binaries(binaries: list[str], workspace_root: Path) -> dict[str, bool]:
    result: dict[str, bool] = {}
    for binary in binaries:
        name = str(binary).strip()
        if not name:
            continue
        result[name] = find_runtime_executable(name, workspace_root) is not None
    return result


def _probe_paths(paths: list[str], workspace_root: Path) -> dict[str, bool]:
    result: dict[str, bool] = {}
    for raw_path in paths:
        name = str(raw_path).strip()
        if not name:
            continue
        path = Path(name).expanduser()
        if not path.is_absolute():
            path = workspace_root / path
        result[name] = path.exists()
    return result


def _probe_node_modules(modules: list[str]) -> dict[str, bool]:
    node = find_runtime_executable("node")
    result: dict[str, bool] = {}
    for module in modules:
        name = str(module).strip()
        if not name:
            continue
        if not node:
            result[name] = False
            continue
        try:
            completed = subprocess.run(
                [node, "-e", f"require.resolve({name!r}); console.log('ok')"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            result[name] = completed.returncode == 0 and "ok" in (completed.stdout or "")
        except Exception:
            result[name] = False
    return result


# ---------------------------------------------------------------------------
# Startup health report
# ---------------------------------------------------------------------------

def build_skill_health_report(workspace_root: str | Path) -> dict[str, Any]:
    """Build a per-skill registration/health report.

    This does NOT register tools (registration needs a ToolRegistry and is the
    runtime's job). It validates declarations and checks that every declared
    script file exists, producing a structured report the runtime can emit and
    that `validate_work_agent_skill` can surface.
    """
    root = Path(workspace_root).resolve()
    reports: list[SkillHealthReport] = []
    for manifest in load_skill_manifests(root):
        skill_dir = root / manifest.path
        tools: list[SkillToolRegistration] = []
        missing_scripts: list[str] = []
        issues: list[str] = []
        for tool_decl, source, script_path in discover_declared_tools(skill_dir, root):
            ok, reason = validate_tool_declaration(tool_decl, manifest.id)
            full_script = skill_dir / script_path
            script_exists = full_script.is_file()
            if not script_exists:
                missing_scripts.append(script_path)
            if not ok:
                tools.append(
                    SkillToolRegistration(
                        skill_id=manifest.id,
                        tool_name=str(tool_decl.get("name") or ""),
                        registered=False,
                        reason=reason,
                        source=source,
                        script_path=script_path,
                    )
                )
                continue
            tools.append(
                SkillToolRegistration(
                    skill_id=manifest.id,
                    tool_name=str(tool_decl.get("name") or ""),
                    registered=script_exists,
                    reason="" if script_exists else f"脚本不存在：{script_path}",
                    source=source,
                    script_path=script_path,
                )
            )
        if missing_scripts:
            issues.append(f"缺失脚本：{', '.join(missing_scripts)}")
        reports.append(
            SkillHealthReport(
                skill_id=manifest.id,
                path=manifest.path,
                ok=all(t.registered for t in tools) and not missing_scripts,
                tools=tools,
                missing_scripts=missing_scripts,
                issues=issues,
            )
        )
    return {
        "ok": all(r.ok for r in reports),
        "skills": [r.to_payload() for r in reports],
        "python_executable": str(office_python()),
    }


def write_skill_health_report(workspace_root: str | Path) -> Path:
    report = build_skill_health_report(workspace_root)
    root = Path(workspace_root).resolve()
    report_dir = root / HEALTH_REPORT_DIR
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / HEALTH_REPORT_NAME
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Helpers re-exported for skill_runtime (kept here to avoid circular imports)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SkillManifest:
    id: str
    label: str
    mention: str
    description: str
    when_to_use: str
    tool_name: str | None
    outputs: list[str]
    source_url: str | None
    path: str
    default_enabled: bool = False
    native_tools: list[dict[str, str]] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "label": self.label,
            "mention": self.mention,
            "description": self.description,
            "when_to_use": self.when_to_use,
            "path": self.path,
            "default_enabled": self.default_enabled,
        }
        if self.tool_name:
            payload["tool_name"] = self.tool_name
        if self.outputs:
            payload["outputs"] = self.outputs
        if self.source_url:
            payload["source_url"] = self.source_url
        if self.native_tools:
            payload["native_tools"] = self.native_tools
        return payload


def humanize_skill_id(skill_id: str) -> str:
    labels = {
        "skill-creator": "技能创建器",
        "docx": "Word 文档",
        "xlsx": "Excel 表格",
        "pptx": "PPT 演示",
        "pdf": "PDF 文档",
    }
    return labels.get(skill_id, skill_id.replace("-", " ").title())


def normalize_skill_id(value: str) -> str:
    cleaned = str(value or "").strip().lower().replace("_", "-")
    cleaned = re.sub(r"[^a-z0-9-]+", "-", cleaned)
    cleaned = re.sub(r"-+", "-", cleaned).strip("-")
    return cleaned


def office_python() -> Path:
    agent_python = project_agent_python(Path.cwd())
    candidates = [
        os.getenv("WORK_AGENT_OFFICE_PYTHON"),
        str(agent_python) if agent_python else None,
        str(Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"),
        sys.executable,
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    return Path(sys.executable)


__all__ = [
    "SKILLS_DIR",
    "EXTRA_SKILL_DIRS",
    "DEFAULT_MAX_SKILL_CHARS",
    "SkillManifest",
    "DeclaredTool",
    "SkillToolRegistration",
    "SkillHealthReport",
    "SkillEnvironmentProbe",
    "parse_skill_markdown",
    "discover_declared_tools",
    "read_skill_config",
    "skill_roots",
    "load_skill_manifests",
    "load_single_skill_manifest",
    "summarize_declared_skill_tools",
    "is_valid_tool_name",
    "validate_tool_declaration",
    "probe_skill_environment",
    "build_skill_health_report",
    "write_skill_health_report",
    "humanize_skill_id",
    "normalize_skill_id",
    "office_python",
]
