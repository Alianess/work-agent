from __future__ import annotations

import base64
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pypdf import PdfReader
from reportlab.pdfgen import canvas

from work_agent_core import web_server
from work_agent_core.office_workspace import (
    PDF_INPUTS_RELATIVE_ROOT,
    PDF_OUTPUTS_RELATIVE_ROOT,
    merge_pdfs,
    relative_workspace_path,
    save_pdf_input,
)


def make_pdf_bytes(label: str) -> bytes:
    stream = io.BytesIO()
    document = canvas.Canvas(stream)
    document.setTitle(label)
    document.drawString(72, 720, label)
    document.showPage()
    document.save()
    return stream.getvalue()


class PdfWorkspaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name).resolve()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_merge_preserves_the_user_supplied_order_and_verifies_pages(self) -> None:
        first, first_pages = save_pdf_input(self.root, name="first.pdf", data=make_pdf_bytes("first"))
        second, second_pages = save_pdf_input(self.root, name="second.pdf", data=make_pdf_bytes("second"))

        output, page_count, source_count = merge_pdfs(
            self.root,
            source_paths=[
                relative_workspace_path(self.root, second),
                relative_workspace_path(self.root, first),
            ],
            output_name="项目材料汇编",
        )

        self.assertEqual(first_pages + second_pages, 2)
        self.assertEqual(source_count, 2)
        self.assertEqual(page_count, 2)
        self.assertEqual(output.parent, self.root / PDF_OUTPUTS_RELATIVE_ROOT)
        reader = PdfReader(str(output))
        self.assertEqual(len(reader.pages), 2)
        self.assertIn("second", reader.pages[0].extract_text())
        self.assertIn("first", reader.pages[1].extract_text())

    def test_rejects_non_pdf_upload_and_paths_outside_the_office_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "有效的 PDF"):
            save_pdf_input(self.root, name="not-pdf.pdf", data=b"plain text")

        outside = self.root / "outside.pdf"
        outside.write_bytes(make_pdf_bytes("outside"))
        valid, _ = save_pdf_input(self.root, name="valid.pdf", data=make_pdf_bytes("valid"))
        with self.assertRaisesRegex(ValueError, "文件办公区"):
            merge_pdfs(
                self.root,
                source_paths=[relative_workspace_path(self.root, valid), "outside.pdf"],
                output_name="不应生成",
            )

    def test_web_payload_stores_inputs_and_exposes_the_merged_result_to_file_library(self) -> None:
        with patch.object(web_server, "account_workspace_root", return_value=self.root):
            first = web_server.add_office_pdf_payload(
                {
                    "name": "甲.pdf",
                    "mime_type": "application/pdf",
                    "content_base64": base64.b64encode(make_pdf_bytes("甲")).decode("ascii"),
                }
            )
            second = web_server.add_office_pdf_payload(
                {
                    "name": "乙.pdf",
                    "mime_type": "application/pdf",
                    "content_base64": base64.b64encode(make_pdf_bytes("乙")).decode("ascii"),
                }
            )
            merged = web_server.merge_office_pdfs_payload(
                {
                    "source_paths": [second["input"]["path"], first["input"]["path"]],
                    "output_name": "合并结果",
                }
            )
            library = web_server.list_files_payload("meet_files", limit=100)

        self.assertEqual(merged["pages"], 2)
        self.assertTrue(merged["output"]["path"].startswith("meet_files/office_workspace/pdf_outputs/"))
        library_items = {item["path"]: item for item in library["files"]}
        self.assertIn(merged["output"]["path"], library_items)
        self.assertEqual(library_items[first["input"]["path"]]["library_section"], "office_input")
        self.assertEqual(library_items[merged["output"]["path"]]["library_section"], "office_output")
        self.assertTrue(library_items[merged["output"]["path"]]["download_url"].startswith("/api/file/download?path="))
        self.assertTrue((self.root / PDF_INPUTS_RELATIVE_ROOT).is_dir())


if __name__ == "__main__":
    unittest.main()
