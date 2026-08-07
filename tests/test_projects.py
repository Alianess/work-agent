from __future__ import annotations

import base64
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from unittest.mock import patch

from openpyxl import load_workbook

from work_agent_core import web_server
from work_agent_core.project_timeline import parse_timeline_date


class ProjectPayloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name).resolve()
        self.workspace_patch = patch.object(
            web_server,
            "account_workspace_root",
            return_value=self.workspace,
        )
        self.workspace_patch.start()

    def tearDown(self) -> None:
        self.workspace_patch.stop()
        self.temp_dir.cleanup()

    def test_project_round_trip_and_agent_context(self) -> None:
        created = web_server.create_project_payload(
            {
                "name": "科技项目可行性报告",
                "instructions": "优先引用正式政策原文，不确定数据标注待核验。",
            }
        )["project"]

        self.assertEqual(created["memory_scope"], "project_only")
        self.assertEqual(created["file_count"], 0)
        self.assertTrue((self.workspace / created["root"] / "project.json").is_file())

        uploaded = web_server.add_project_file_payload(
            created["id"],
            {
                "name": "政策依据.txt",
                "mime_type": "text/plain",
                "content_base64": base64.b64encode("正式材料".encode("utf-8")).decode("ascii"),
            },
        )
        file_path = uploaded["file"]["path"]
        self.assertEqual(uploaded["project"]["file_count"], 1)

        paths, context, project_id = web_server.resolve_project_chat_context(created["id"], [])
        self.assertEqual(project_id, created["id"])
        self.assertIn(file_path, paths)
        self.assertIn("科技项目可行性报告", context)
        self.assertIn("仅项目", context)
        self.assertIn("优先引用正式政策原文", context)

        deleted = web_server.delete_project_file_payload(created["id"], {"path": file_path})
        self.assertTrue(deleted["ok"])
        self.assertEqual(deleted["project"]["file_count"], 0)

    def test_concurrent_project_and_attachment_writes_preserve_every_entry(self) -> None:
        project = web_server.create_project_payload({"name": "并发写入测试"})["project"]

        def upload_project_file(index: int) -> None:
            web_server.add_project_file_bytes_payload(
                project["id"],
                {"name": f"项目资料-{index}.txt", "mime_type": "text/plain"},
                f"project-{index}".encode("utf-8"),
            )

        def upload_attachment(index: int) -> None:
            web_server.add_attachment_bytes_payload(
                {
                    "name": f"附件-{index}.txt",
                    "mime_type": "text/plain",
                    "last_modified": index + 1,
                },
                f"attachment-{index}".encode("utf-8"),
            )

        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(upload_project_file, range(8)))
            list(executor.map(upload_attachment, range(8)))

        detail = web_server.project_detail_payload(project["id"])["project"]
        self.assertEqual(detail["file_count"], 8)
        attachment_dir = self.workspace / "meet_files" / "attachments"
        attachment_index = web_server.load_attachment_index(attachment_dir)
        self.assertEqual(len(attachment_index), 8)
        self.assertEqual(
            len([path for path in attachment_dir.iterdir() if path.is_file() and not path.name.startswith(".")]),
            8,
        )
        self.assertFalse(any(path.name.endswith(".tmp") for path in self.workspace.rglob("*")))

    def test_project_timeline_understands_common_chinese_dates(self) -> None:
        self.assertEqual(
            parse_timeline_date("7月31日", reference_year=2026),
            date(2026, 7, 31),
        )

    def test_invalid_or_foreign_project_paths_are_rejected(self) -> None:
        created = web_server.create_project_payload({"name": "项目 A"})["project"]
        foreign = self.workspace / "meet_files" / "projects" / "project-deadbeefdead" / "sources"
        foreign.mkdir(parents=True)
        foreign_file = foreign / "secret.txt"
        foreign_file.write_text("secret", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "当前项目资料目录"):
            web_server.delete_project_file_payload(
                created["id"],
                {"path": str(foreign_file.relative_to(self.workspace))},
            )

        with self.assertRaisesRegex(ValueError, "项目不存在"):
            web_server.resolve_project_chat_context("project-aaaaaaaaaaaa", [])

    def test_project_materials_are_grouped_and_only_current_versions_are_prominent(self) -> None:
        project = web_server.create_project_payload({"name": "本轮科技"})["project"]
        names = [
            "本轮科技项目可研报告初稿.docx",
            "本轮科技项目可研报告定稿.pdf",
            "项目启动会_会议沟通内容整理_内部留档版.md",
            "项目启动会_会议纪要_工作提交版.docx",
            "项目启动会_录音转写.md",
            "人工智能教育政策依据.pdf",
        ]
        for index, name in enumerate(names):
            web_server.add_project_file_payload(
                project["id"],
                {
                    "name": name,
                    "mime_type": "application/octet-stream",
                    "content_base64": base64.b64encode(f"file-{index}".encode()).decode(),
                },
            )

        payload = web_server.project_detail_payload(project["id"])["project"]
        groups = {group["id"]: group for group in payload["material_groups"]}

        self.assertEqual(payload["file_count"], 6)
        self.assertEqual(payload["material_count"], 3)
        self.assertEqual(payload["hidden_file_count"], 3)
        self.assertEqual(
            groups["deliverable"]["materials"][0]["name"],
            "本轮科技项目可研报告定稿.pdf",
        )
        self.assertEqual(groups["deliverable"]["materials"][0]["document_status"], "final")
        self.assertEqual(len(groups["deliverable"]["materials"][0]["history"]), 1)
        self.assertEqual(
            groups["meeting"]["materials"][0]["name"],
            "项目启动会_会议纪要_工作提交版.docx",
        )
        self.assertNotIn(
            "项目启动会_录音转写.md",
            [
                material["name"]
                for group in payload["material_groups"]
                for material in group["materials"]
            ],
        )

        _paths, context, _project_id = web_server.resolve_project_chat_context(project["id"], [])
        self.assertIn("当前有效材料", context)
        self.assertIn("本轮科技项目可研报告定稿.pdf", context)
        self.assertNotIn("本轮科技项目可研报告初稿.docx", context)

    def test_project_timeline_is_one_excel_source_for_web_and_agent(self) -> None:
        project = web_server.create_project_payload({"name": "本轮科技"})["project"]
        self.assertFalse(project["timeline"]["exists"])

        created = web_server.create_project_timeline_payload(project["id"])["project"]
        timeline = created["timeline"]
        self.assertTrue(timeline["exists"])
        self.assertEqual(timeline["nodes"], [])
        timeline_path = self.workspace / timeline["path"]
        self.assertTrue(timeline_path.is_file())

        updated = web_server.update_project_timeline_payload(
            project["id"],
            {
                "changes": [
                    {
                        "action": "add",
                        "values": {
                            "workstream": "设计装修",
                            "planned_date": "2026-08-20",
                            "title": "装修方案定稿",
                            "status": "推进中",
                            "next_action": "确认预算",
                            "owner": "合作方",
                        },
                    }
                ]
            },
        )["project"]
        self.assertEqual(updated["timeline"]["summary"]["total"], 1)
        self.assertEqual(updated["timeline"]["nodes"][0]["node_id"], "M-001")
        self.assertEqual(updated["timeline"]["nodes"][0]["planned_date"], "2026-08-20")
        self.assertEqual(updated["timeline"]["nodes"][0]["next_action"], "确认预算")

        workbook = load_workbook(timeline_path)
        sheet = workbook["项目推进"]
        sheet["F4"] = "已完成"
        workbook.save(timeline_path)
        workbook.close()

        refreshed = web_server.project_detail_payload(project["id"])["project"]
        self.assertEqual(refreshed["timeline"]["nodes"][0]["status"], "已完成")
        self.assertEqual(refreshed["timeline"]["summary"]["completed"], 1)
        self.assertTrue(
            any((self.workspace / created["root"] / "history" / "timeline").glob("*.xlsx"))
        )

        paths, context, _project_id = web_server.resolve_project_chat_context(project["id"], [])
        self.assertIn(timeline["path"], paths)
        self.assertIn("manage_project_timeline", context)
        self.assertIn("装修方案定稿", context)

    def test_conversation_payload_restores_project_id_from_session_metadata(self) -> None:
        project = web_server.create_project_payload({"name": "本轮科技"})["project"]
        conversation_id = "local-project-restore"
        conversation_dir = self.workspace / "conversation_history"
        session_path = conversation_dir / "sessions" / f"{conversation_id}.json"
        session_path.parent.mkdir(parents=True)
        session_path.write_text(
            json.dumps({"id": conversation_id, "metadata": {"project_id": project["id"]}}, ensure_ascii=False),
            encoding="utf-8",
        )
        history_path = conversation_dir / "conversations.json"
        history_path.write_text(
            json.dumps(
                {"items": [{"id": conversation_id, "title": "项目进展", "group": "最近", "messages": []}]},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        with (
            patch.object(web_server, "user_conversation_dir", return_value=conversation_dir),
            patch.object(web_server, "user_conversation_history_path", return_value=history_path),
        ):
            payload = web_server.load_conversations_payload()

        self.assertEqual(payload["items"][0]["projectId"], project["id"])

    def test_move_conversation_to_and_out_of_project_persists_archive_and_session(self) -> None:
        project = web_server.create_project_payload({"name": "机器人项目"})["project"]
        conversation_id = "local-project-move"
        conversation_dir = self.workspace / "conversation_history"
        history_path = conversation_dir / "conversations.json"
        history_path.parent.mkdir(parents=True)
        attachment_path = self.workspace / "meet_files" / "attachments" / "项目测算表.docx"
        attachment_path.parent.mkdir(parents=True)
        attachment_path.write_bytes(b"conversation-file")
        unrelated_path = self.workspace / "meet_files" / "attachments" / "其他项目.docx"
        unrelated_path.write_bytes(b"unrelated")
        history_path.write_text(
            json.dumps(
                {
                    "items": [
                        {
                            "id": conversation_id,
                            "title": "项目讨论",
                            "group": "最近",
                            "messages": [
                                {
                                    "role": "user",
                                    "content": (
                                        "请分析附件\n\n参考附件：\n"
                                        "- [文档] 项目测算表.docx: "
                                        "meet_files/attachments/项目测算表.docx"
                                    ),
                                }
                            ],
                            "activities": {
                                "2": {
                                    "events": [
                                        {
                                            "event": "activity",
                                            "phase": "observation",
                                            "tool_name": "list_workspace_files",
                                            "title": "列出文件",
                                            "detail": (
                                                "meet_files/attachments/项目测算表.docx\n"
                                                "meet_files/attachments/其他项目.docx"
                                            ),
                                        }
                                    ]
                                }
                            },
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        session_store = web_server.SessionStore(
            self.workspace,
            session_dir=conversation_dir / "sessions",
        )

        with (
            patch.object(web_server, "WORKSPACE_ROOT", self.workspace),
            patch.object(web_server, "user_conversation_dir", return_value=conversation_dir),
            patch.object(web_server, "user_conversation_history_path", return_value=history_path),
            patch.object(web_server, "get_session_store", return_value=session_store),
        ):
            moved = web_server.move_conversation_to_project_payload(
                {"conversation_id": conversation_id, "project_id": project["id"]}
            )
            archived = json.loads(history_path.read_text(encoding="utf-8"))
            self.assertEqual(moved["project_id"], project["id"])
            self.assertEqual(moved["copied_count"], 1)
            self.assertEqual(len(moved["files"]), 1)
            self.assertTrue((self.workspace / moved["files"][0]["path"]).is_file())
            self.assertEqual(archived["items"][0]["projectId"], project["id"])
            self.assertEqual(session_store.load(conversation_id).metadata["project_id"], project["id"])
            chat_files = web_server.conversation_files_payload(conversation_id)
            self.assertEqual([file["name"] for file in chat_files["files"]], ["项目测算表.docx"])

            removed = web_server.move_conversation_to_project_payload(
                {"conversation_id": conversation_id, "project_id": ""}
            )
            archived = json.loads(history_path.read_text(encoding="utf-8"))
            self.assertIsNone(removed["project_id"])
            self.assertNotIn("projectId", archived["items"][0])
            self.assertEqual(session_store.load(conversation_id).metadata["project_id"], "")

    def test_meeting_minutes_sync_is_idempotent_and_refreshes_changed_files(self) -> None:
        project = web_server.create_project_payload({"name": "会议成果项目"})["project"]
        archive_dir = self.workspace / "meet_files" / "会议项目" / "项目启动会"
        archive_dir.mkdir(parents=True)
        output_paths = {
            "asr": archive_dir / "01_录音转写.md",
            "internal": archive_dir / "02_内部纪要.md",
            "work_md": archive_dir / "03_工作纪要.md",
            "work_docx": archive_dir / "03_工作纪要.docx",
        }
        for key, path in output_paths.items():
            path.write_bytes(f"{key}-v1".encode("utf-8"))
        manifest_path = archive_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "title": "项目启动会",
                    "canonical_outputs": {
                        key: str(path.relative_to(self.workspace))
                        for key, path in output_paths.items()
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        first = web_server.sync_meeting_to_project_payload(
            project["id"],
            {"manifest_path": str(manifest_path.relative_to(self.workspace))},
        )
        self.assertEqual(first["copied_count"], 4)
        self.assertEqual(first["unchanged_count"], 0)
        self.assertEqual(first["project"]["file_count"], 4)
        self.assertTrue(
            all("sources/会议纪要/项目启动会/" in item["path"] for item in first["files"])
        )

        second = web_server.sync_meeting_to_project_payload(
            project["id"],
            {"manifest_path": str(manifest_path.relative_to(self.workspace))},
        )
        self.assertEqual(second["copied_count"], 0)
        self.assertEqual(second["unchanged_count"], 4)

        output_paths["work_md"].write_text("work_md-v2", encoding="utf-8")
        refreshed = web_server.sync_meeting_to_project_payload(
            project["id"],
            {"manifest_path": str(manifest_path.relative_to(self.workspace))},
        )
        self.assertEqual(refreshed["copied_count"], 1)
        self.assertEqual(refreshed["unchanged_count"], 3)

    def test_meeting_sync_rejects_manifest_outside_current_account_archive(self) -> None:
        project = web_server.create_project_payload({"name": "项目 A"})["project"]
        manifest_path = self.workspace / "其他目录" / "manifest.json"
        manifest_path.parent.mkdir(parents=True)
        manifest_path.write_text("{}", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "当前账户会议归档"):
            web_server.sync_meeting_to_project_payload(
                project["id"],
                {"manifest_path": str(manifest_path.relative_to(self.workspace))},
            )


if __name__ == "__main__":
    unittest.main()
