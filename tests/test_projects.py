from __future__ import annotations

import base64
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from work_agent_core import web_server


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
