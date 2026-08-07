from __future__ import annotations

"""Account-scoped document workspace operations.

The first operation is intentionally narrow: merge user-selected PDFs in the
exact order supplied by the client.  Inputs and outputs remain in the account
workspace so a browser upload never leaves the local Work Agent service.
"""

from pathlib import Path
from typing import Iterable
import re
import time
import uuid


OFFICE_WORKSPACE_RELATIVE_ROOT = Path("meet_files") / "office_workspace"
PDF_INPUTS_RELATIVE_ROOT = OFFICE_WORKSPACE_RELATIVE_ROOT / "pdf_inputs"
PDF_OUTPUTS_RELATIVE_ROOT = OFFICE_WORKSPACE_RELATIVE_ROOT / "pdf_outputs"
MAX_PDF_SOURCE_BYTES = 250 * 1024 * 1024
MAX_PDF_MERGE_SOURCES = 50
MAX_PDF_MERGE_TOTAL_BYTES = 300 * 1024 * 1024


def save_pdf_input(workspace_root: str | Path, *, name: str, data: bytes) -> tuple[Path, int]:
    """Validate and retain one PDF upload under the current account workspace."""

    root = Path(workspace_root).resolve()
    display_name = _pdf_filename(name)
    if not data:
        raise ValueError("PDF 文件不能为空。")
    if len(data) > MAX_PDF_SOURCE_BYTES:
        raise ValueError(
            f"单个 PDF 最大支持 {MAX_PDF_SOURCE_BYTES // (1024 * 1024)} MB。"
        )
    if not data.lstrip().startswith(b"%PDF-"):
        raise ValueError("文件不是有效的 PDF。")

    target_dir = (root / PDF_INPUTS_RELATIVE_ROOT).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    _ensure_inside(target_dir, root / OFFICE_WORKSPACE_RELATIVE_ROOT)
    target = _unique_path(target_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-{display_name}")
    target.write_bytes(data)
    try:
        pages = _pdf_page_count(target)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    return target, pages


def merge_pdfs(
    workspace_root: str | Path,
    *,
    source_paths: Iterable[str],
    output_name: str,
) -> tuple[Path, int, int]:
    """Merge validated, account-local PDF inputs in their supplied order."""

    root = Path(workspace_root).resolve()
    inputs_root = (root / PDF_INPUTS_RELATIVE_ROOT).resolve()
    outputs_root = (root / PDF_OUTPUTS_RELATIVE_ROOT).resolve()
    raw_paths = [str(path or "").strip() for path in source_paths]
    if len(raw_paths) < 2:
        raise ValueError("请至少按顺序添加两个 PDF。")
    if len(raw_paths) > MAX_PDF_MERGE_SOURCES:
        raise ValueError(f"单次最多合并 {MAX_PDF_MERGE_SOURCES} 个 PDF。")

    source_files: list[Path] = []
    total_bytes = 0
    for raw_path in raw_paths:
        source = (root / raw_path).resolve()
        _ensure_inside(source, inputs_root)
        if not source.is_file() or source.suffix.lower() != ".pdf":
            raise ValueError("合并列表中存在不可用的 PDF。请重新添加该文件。")
        total_bytes += source.stat().st_size
        if total_bytes > MAX_PDF_MERGE_TOTAL_BYTES:
            raise ValueError("本次合并的 PDF 总大小不能超过 300 MB。")
        source_files.append(source)

    output_filename = _pdf_filename(output_name)
    outputs_root.mkdir(parents=True, exist_ok=True)
    _ensure_inside(outputs_root, root / OFFICE_WORKSPACE_RELATIVE_ROOT)
    output_path = _unique_path(outputs_root / output_filename)
    temporary_output = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.part")

    try:
        from pypdf import PdfReader, PdfWriter

        writer = PdfWriter()
        total_pages = 0
        for source in source_files:
            reader = PdfReader(str(source), strict=False)
            if reader.is_encrypted:
                raise ValueError(f"“{source.name}”受密码保护，暂不支持合并。")
            page_count = len(reader.pages)
            if page_count <= 0:
                raise ValueError(f"“{source.name}”没有可合并的页面。")
            total_pages += page_count
            for page in reader.pages:
                writer.add_page(page)
        writer.add_metadata({"/Title": output_path.stem, "/Producer": "Work Agent PDF Merge"})
        with temporary_output.open("wb") as stream:
            writer.write(stream)

        verified_reader = PdfReader(str(temporary_output), strict=False)
        if verified_reader.is_encrypted or len(verified_reader.pages) != total_pages:
            raise ValueError("合并后的 PDF 未通过完整性校验。")
        temporary_output.replace(output_path)
        return output_path, total_pages, len(source_files)
    except Exception:
        temporary_output.unlink(missing_ok=True)
        raise


def relative_workspace_path(workspace_root: str | Path, path: Path) -> str:
    return str(path.resolve().relative_to(Path(workspace_root).resolve()))


def _pdf_page_count(path: Path) -> int:
    from pypdf import PdfReader

    reader = PdfReader(str(path), strict=False)
    if reader.is_encrypted:
        raise ValueError(f"“{path.name}”受密码保护，暂不支持。")
    pages = len(reader.pages)
    if pages <= 0:
        raise ValueError("PDF 没有可读取的页面。")
    return pages


def _pdf_filename(raw_name: str) -> str:
    name = str(raw_name or "").strip().replace("\\", "/").split("/")[-1]
    stem = Path(name).stem.strip() if name else ""
    stem = re.sub(r"[\x00-\x1f<>:\"|?*]", "-", stem).strip(" .-")
    if not stem:
        stem = "合并后的文件"
    return f"{stem[:120]}.pdf"


def _unique_path(candidate: Path) -> Path:
    if not candidate.exists():
        return candidate
    for index in range(2, 10_000):
        alternative = candidate.with_name(f"{candidate.stem} ({index}){candidate.suffix}")
        if not alternative.exists():
            return alternative
    raise RuntimeError("无法为 PDF 创建唯一文件名。")


def _ensure_inside(path: Path, parent: Path) -> None:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError as error:
        raise ValueError("只能使用当前文件办公区中的 PDF。") from error
