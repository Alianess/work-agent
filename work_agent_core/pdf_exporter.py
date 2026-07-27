"""Create a PDF document from Markdown content.

Supports H1/H2/H3 headings, paragraphs, bullet lists, and Markdown tables.
Requires `reportlab`. Run via the office_python interpreter.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import re


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    markdown = _read_markdown(args)
    if not markdown.strip():
        raise ValueError("Markdown 内容为空。")
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    create_pdf(markdown, output_path, title=args.title or "")
    print(
        json.dumps(
            {"output_path": str(output_path)},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a PDF document from Markdown.")
    parser.add_argument("--markdown-path")
    parser.add_argument("--markdown-content")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--title", default="")
    return parser.parse_args(argv)


def _read_markdown(args: argparse.Namespace) -> str:
    if args.markdown_path:
        return Path(args.markdown_path).read_text(encoding="utf-8", errors="replace")
    return args.markdown_content or ""


def create_pdf(markdown: str, output_path: Path, *, title: str = "") -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
    )

    font_name = _register_cjk_font()

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CNTitle",
        parent=styles["Title"],
        fontName=font_name,
        fontSize=22,
        leading=30,
        spaceAfter=14,
    )
    h1_style = ParagraphStyle("CNH1", parent=styles["Heading1"], fontName=font_name, fontSize=18, leading=24, spaceBefore=14, spaceAfter=8)
    h2_style = ParagraphStyle("CNH2", parent=styles["Heading2"], fontName=font_name, fontSize=15, leading=21, spaceBefore=10, spaceAfter=6)
    h3_style = ParagraphStyle("CNH3", parent=styles["Heading3"], fontName=font_name, fontSize=13, leading=18, spaceBefore=8, spaceAfter=4)
    body_style = ParagraphStyle("CNBody", parent=styles["BodyText"], fontName=font_name, fontSize=11, leading=18, spaceAfter=6)
    bullet_style = ParagraphStyle("CNBullet", parent=body_style, leftIndent=18, bulletIndent=6)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        title=title or None,
    )
    story = []
    if title:
        story.append(Paragraph(_escape(title), title_style))
        story.append(Spacer(1, 6))

    for block in _iter_blocks(markdown):
        kind = block[0]
        if kind == "h1":
            story.append(Paragraph(_escape(block[1]), h1_style))
        elif kind == "h2":
            story.append(Paragraph(_escape(block[1]), h2_style))
        elif kind == "h3":
            story.append(Paragraph(_escape(block[1]), h3_style))
        elif kind == "bullet":
            story.append(Paragraph(_escape(block[1]), bullet_style, bulletText="•"))
        elif kind == "para":
            story.append(Paragraph(_escape(block[1]), body_style))
        elif kind == "table":
            story.append(_build_table(block[1], body_style))
    doc.build(story)


def _register_cjk_font() -> str:
    """Register a CJK-capable TTF font and return the registered font name."""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    candidates = [
        ("CN-Songti", "/System/Library/Fonts/Supplemental/Songti.ttc"),
        ("CN-PingFang", "/System/Library/Fonts/PingFang.ttc"),
        ("CN-Heiti", "/System/Library/Fonts/Supplemental/Heiti.ttc"),
    ]
    for name, path in candidates:
        try:
            pdfmetrics.registerFont(TTFont(name, path))
            # Try to register bold/italic variants of the same family; if not present,
            # map them to the regular face so <b>/<i> tags still render.
            try:
                pdfmetrics.registerFontFamily(
                    name,
                    normal=name,
                    bold=name,
                    italic=name,
                    boldItalic=name,
                )
            except Exception:
                pass
            return name
        except Exception:
            continue
    # Last-resort CID font (renders CJK without an external file).
    try:
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont

        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        try:
            pdfmetrics.registerFontFamily(
                "STSong-Light",
                normal="STSong-Light",
                bold="STSong-Light",
                italic="STSong-Light",
                boldItalic="STSong-Light",
            )
        except Exception:
            pass
        return "STSong-Light"
    except Exception as error:
        raise RuntimeError("未找到可用的中文字体；无法生成 PDF。") from error


def _iter_blocks(markdown: str):
    lines = markdown.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        if stripped.startswith("### "):
            yield ("h3", stripped[4:].strip())
            i += 1
            continue
        if stripped.startswith("## "):
            yield ("h2", stripped[3:].strip())
            i += 1
            continue
        if stripped.startswith("# "):
            yield ("h1", stripped[2:].strip())
            i += 1
            continue
        if re.match(r"^\s*([-*•]|\d+\.)\s+", stripped):
            text = re.sub(r"^\s*([-*•]|\d+\.)\s+", "", stripped)
            yield ("bullet", text)
            i += 1
            continue
        if stripped.startswith("|") and stripped.endswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|") and lines[i].strip().endswith("|"):
                row_line = lines[i].strip()
                cells = [c.strip() for c in row_line[1:-1].split("|")]
                if all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                    i += 1
                    continue
                rows.append(cells)
                i += 1
            if rows:
                yield ("table", rows)
            continue
        # Plain paragraph
        yield ("para", stripped)
        i += 1


def _build_table(rows: list[list[str]], body_style) -> "Table":
    from reportlab.platypus import Paragraph, Table, TableStyle
    from reportlab.lib import colors

    data = [[Paragraph(_escape(cell), body_style) for cell in row] for row in rows]
    table = Table(data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4A5568")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _unescape_inline(md: str) -> str:
    # Convert basic inline markdown (bold/italic/code) to reportlab markup.
    md = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", md)
    md = re.sub(r"\*(.+?)\*", r"<i>\1</i>", md)
    md = re.sub(r"`(.+?)`", r'<font name="Courier">\1</font>', md)
    return md


if __name__ == "__main__":
    raise SystemExit(main())
