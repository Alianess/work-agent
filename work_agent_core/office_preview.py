from __future__ import annotations

from pathlib import Path
from urllib.parse import quote
import os
import shutil
import subprocess


OFFICE_TO_PDF_EXTENSIONS = {
    ".doc",
    ".docx",
    ".odt",
    ".ppt",
    ".pptx",
    ".odp",
    ".xls",
    ".xlsx",
    ".ods",
}


def convert_office_to_pdf(source_path: Path, *, output_dir: Path, workspace_root: Path) -> Path:
    source_path = source_path.resolve()
    output_dir = output_dir.resolve()
    workspace_root = workspace_root.resolve()
    if source_path.suffix.lower() not in OFFICE_TO_PDF_EXTENSIONS:
        raise ValueError(f"不支持转 PDF 的文件类型：{source_path.suffix or '无扩展名'}")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    soffice = find_soffice()
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_dir = workspace_root / "tmp" / "libreoffice-profile"
    profile_dir.mkdir(parents=True, exist_ok=True)

    command = [
        str(soffice),
        "--headless",
        f"-env:UserInstallation={profile_uri(profile_dir)}",
        "--convert-to",
        "pdf",
        "--outdir",
        str(output_dir),
        str(source_path),
    ]
    result = subprocess.run(
        command,
        cwd=workspace_root,
        env=office_preview_env(workspace_root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=300,
        check=False,
    )
    expected = output_dir / f"{source_path.stem}.pdf"
    if result.returncode != 0 or not expected.is_file():
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"LibreOffice 转 PDF 失败：{truncate(detail, 3000)}")
    return expected


def find_soffice() -> Path:
    configured = os.environ.get("WORK_AGENT_SOFFICE", "").strip()
    candidates = [
        Path(configured) if configured else None,
        Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/bin/override/soffice",
        Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/bin/soffice",
        Path("/Applications/LibreOffice.app/Contents/MacOS/soffice"),
    ]
    discovered = shutil.which("soffice") or shutil.which("libreoffice")
    if discovered:
        candidates.append(Path(discovered))
    for candidate in candidates:
        if candidate and candidate.is_file():
            return candidate
    raise FileNotFoundError("未找到 soffice；无法生成 Office 文件预览。")


def office_preview_env(workspace_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    poppler_root = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/poppler"
    fonts_conf = poppler_root / "etc/fonts/fonts.conf"
    fonts_dir = poppler_root / "etc/fonts"
    if fonts_conf.is_file():
        env["FONTCONFIG_FILE"] = str(fonts_conf)
    if fonts_dir.is_dir():
        env["FONTCONFIG_PATH"] = str(fonts_dir)
    cache_home = workspace_root / "tmp" / "fontconfig-cache"
    cache_home.mkdir(parents=True, exist_ok=True)
    env["XDG_CACHE_HOME"] = str(cache_home)
    return env


def profile_uri(path: Path) -> str:
    return "file://" + quote(str(path.resolve()))


def truncate(value: str, max_chars: int) -> str:
    text = value.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...[truncated]"
