from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from work_agent_core.work_reports import WorkReportStore


class WorkReportStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.store = WorkReportStore(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_json(self, relative: str, payload: object) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def test_collect_uses_activity_timestamp_and_successful_artifact(self) -> None:
        timestamp = int(datetime(2026, 7, 31, 10, 15).timestamp())
        self.write_json(
            "conversation_history/conversations.json",
            {
                "items": [
                    {
                        "id": "chat-1",
                        "title": "项目方案",
                        "messages": [
                            {"role": "assistant", "content": "你好"},
                            {"role": "user", "content": "完善合作方案并生成文档"},
                            {"role": "assistant", "content": "方案已完成并核验。"},
                        ],
                        "activities": {
                            "2": {
                                "events": [
                                    {
                                        "event": "activity",
                                        "ts_ms": timestamp * 1000,
                                        "turn_id": "turn-1",
                                        "activity_type": "file_edit",
                                        "command_status": "success",
                                        "file_path": "outputs/合作方案.docx",
                                    }
                                ]
                            }
                        },
                    }
                ]
            },
        )

        result = self.store.collect(report_type="daily", target_date="2026-07-31")

        self.assertEqual(result["start_date"], "2026-07-31")
        self.assertEqual(result["artifacts"], ["outputs/合作方案.docx"])
        self.assertEqual(result["evidence"][0]["conversation_title"], "项目方案")
        self.assertIn("完善合作方案", result["evidence"][0]["user_request"])
        self.assertIn("已完成", result["evidence"][0]["result"])

    def test_save_daily_report_removes_missing_date(self) -> None:
        saved = self.store.save_report(
            report_type="daily",
            target_date="2026-07-31",
            content="# 2026-07-31 工作简报\n\n完成测试。",
            source_coverage="full",
        )

        self.assertEqual(saved["content_path"], "work_reports/daily/2026-07-31.md")
        self.assertTrue(saved["verified"])
        self.assertGreater(saved["content_bytes"], 0)
        verified = self.store.read_report(report_type="daily", target_date="2026-07-31")
        self.assertTrue(verified["verified"])
        self.assertEqual(verified["content_sha256"], saved["content_sha256"])
        self.assertIn("完成测试", verified["content"])
        collected = self.store.collect(report_type="daily", target_date="2026-07-31")
        self.assertEqual(collected["missing_daily_reports"], [])
        self.assertIn("完成测试", collected["daily_reports"][0]["content"])

    def test_collect_reads_activity_from_per_conversation_archive_storage(self) -> None:
        timestamp = int(datetime(2026, 7, 31, 9, 0).timestamp())
        self.write_json(
            "conversation_history/conversations.json",
            {
                "storage": "per_item",
                "order": ["chat-per-item"],
                "items": [{"id": "chat-per-item", "title": "清单条目"}],
            },
        )
        self.write_json(
            "conversation_history/archive_items/chat-per-item.json",
            {
                "id": "chat-per-item",
                "title": "完整条目",
                "messages": [
                    {"role": "user", "content": "完成项目核验"},
                    {"role": "assistant", "content": "核验完成"},
                ],
                "activities": {
                    "1": {"events": [{"ts_ms": timestamp * 1000, "command_status": "success"}]}
                },
            },
        )

        result = self.store.collect(report_type="daily", target_date="2026-07-31")

        self.assertEqual(result["evidence"][0]["conversation_title"], "完整条目")
        self.assertIn("完成项目核验", result["evidence"][0]["user_request"])

    def test_calendar_override_controls_adjusted_workday(self) -> None:
        self.write_json(
            "work_reports/calendar_overrides.json",
            {"source": "test", "days": {"2026-08-01": True, "2026-08-03": False}},
        )

        self.assertTrue(self.store.is_workday(date(2026, 8, 1)))
        self.assertFalse(self.store.is_workday(date(2026, 8, 3)))

    def test_official_2026_calendar_is_global_for_every_account(self) -> None:
        self.assertFalse(self.store.is_workday(date(2026, 9, 25)))
        self.assertFalse(self.store.is_workday(date(2026, 10, 1)))
        self.assertFalse(self.store.is_workday(date(2026, 10, 7)))
        self.assertTrue(self.store.is_workday(date(2026, 10, 10)))
        status = self.store.calendar_status(date(2026, 1, 1), date(2026, 12, 31))
        self.assertEqual(status["years_using_weekday_fallback"], [])
        self.assertEqual(status["document_number"], "国办发明电〔2025〕7号")

    def test_update_calendar_merges_verified_overrides(self) -> None:
        result = self.store.update_calendar(
            source="https://example.gov/holiday-notice",
            days={"2026-10-01": False, "2026-10-10": True},
        )

        self.assertEqual(result["updated_count"], 2)
        self.assertFalse(self.store.is_workday(date(2026, 10, 1)))
        self.assertTrue(self.store.is_workday(date(2026, 10, 10)))

    def test_audit_is_throttled_and_preserves_missing_signature(self) -> None:
        now = datetime(2026, 8, 3, 19, 0)
        first = self.store.audit_if_due(now=now, interval_seconds=1800)
        second = self.store.audit_if_due(now=datetime(2026, 8, 3, 19, 10), interval_seconds=1800)

        self.assertTrue(first["notify"])
        self.assertTrue(first["missing_daily_reports"])
        self.assertTrue(second["skipped"])
        self.assertEqual(second["missing_signature"], first["missing_signature"])

    def test_period_evidence_keeps_early_and_late_dates(self) -> None:
        items = []
        for day in (20, 31):
            timestamp = int(datetime(2026, 7, day, 10, 0).timestamp())
            for index in range(80):
                items.append(
                    {
                        "timestamp": timestamp + index,
                        "date": f"2026-07-{day:02d}",
                        "conversation_title": f"day-{day}",
                        "artifacts": [],
                    }
                )
        from work_agent_core.work_reports import balanced_evidence

        selected = balanced_evidence(items)
        dates = {item["date"] for item in selected}
        self.assertEqual(dates, {"2026-07-20", "2026-07-31"})
        self.assertEqual(len(selected), 50)

    def test_calendar_month_returns_compact_day_counts_and_report_markers(self) -> None:
        timestamp = int(datetime(2026, 7, 31, 10, 15).timestamp())
        self.write_json(
            "conversation_history/conversations.json",
            {
                "items": [{
                    "id": "chat-1",
                    "title": "项目方案",
                    "messages": [
                        {"role": "user", "content": "完成方案"},
                        {"role": "assistant", "content": "方案已完成"},
                    ],
                    "activities": {"1": {"events": [{"ts_ms": timestamp * 1000}]}},
                }]
            },
        )
        self.store.save_report(
            report_type="daily",
            target_date="2026-07-31",
            content="# 工作简报",
            source_coverage="full",
        )

        result = self.store.calendar_month(year=2026, month=7)
        day = next(item for item in result["days"] if item["date"] == "2026-07-31")

        self.assertEqual(len(result["days"]), 31)
        self.assertEqual(day["evidence_count"], 1)
        self.assertEqual(day["report_types"], ["daily"])
        self.assertNotIn("content", result["reports"][0])

    def test_day_detail_includes_daily_and_covering_biweekly_report(self) -> None:
        self.store.save_report(
            report_type="daily",
            target_date="2026-07-31",
            content="# 日报",
        )
        self.store.save_report(
            report_type="biweekly",
            start_date="2026-07-20",
            end_date="2026-07-31",
            content="# 双周报",
        )

        result = self.store.day_detail(target_date="2026-07-31")

        self.assertIn("# 日报", result["daily_report"]["content"])
        self.assertEqual(
            {item["report_type"] for item in result["covering_reports"]},
            {"daily", "biweekly"},
        )

    def test_biweekly_collection_uses_daily_report_instead_of_covered_raw_evidence(self) -> None:
        timestamp = int(datetime(2026, 7, 31, 10, 15).timestamp())
        self.write_json(
            "conversation_history/conversations.json",
            {
                "items": [{
                    "id": "chat-1",
                    "title": "项目方案",
                    "messages": [
                        {"role": "user", "content": "原始请求"},
                        {"role": "assistant", "content": "原始结果"},
                    ],
                    "activities": {"1": {"events": [{"ts_ms": timestamp * 1000}]}},
                }]
            },
        )
        self.store.save_report(
            report_type="daily",
            target_date="2026-07-31",
            content="# 已压缩日报",
        )

        result = self.store.collect(
            report_type="biweekly",
            start_date="2026-07-20",
            end_date="2026-07-31",
        )

        self.assertEqual(result["evidence"], [])
        self.assertEqual(result["covered_by_daily_report_days"], ["2026-07-31"])
        self.assertEqual(result["evidence_counts_by_date"]["2026-07-31"], 1)
        self.assertIn("已压缩日报", result["daily_reports"][0]["content"])

    def test_turn_archive_projection_keeps_full_public_text_and_folds_tool_bulk(self) -> None:
        timestamp = int(datetime(2026, 7, 31, 10, 15).timestamp())
        user_text = "完整用户问题" * 300
        path_text = "先核对工作路径，再形成最终交付。"
        final_text = "完整最终答复" * 400
        self.write_json(
            "conversation_history/conversations.json",
            {"items": [{"id": "chat-1", "title": "项目方案", "messages": [], "activities": {}}]},
        )
        self.write_json(
            "conversation_history/sessions/chat-1.json",
            {
                "id": "chat-1",
                "messages": [
                    {"role": "user", "content": user_text},
                    {
                        "role": "assistant",
                        "content": path_text,
                        "tool_calls": [{
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": '{"path":"evidence.md"}'},
                        }],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call-1",
                        "name": "read_file",
                        "content": "不应进入日报的详细工具回显" * 1000,
                    },
                    {"role": "assistant", "content": final_text},
                ],
            },
        )
        self.write_json(
            "conversation_history/turns/turn-1.json",
            {
                "id": "turn-1",
                "conversation_id": "chat-1",
                "created_at": timestamp,
                "status": "succeeded",
                "final_message": final_text,
                "events": [],
            },
        )

        result = self.store.collect(report_type="daily", target_date="2026-07-31")
        evidence = result["evidence"][0]

        self.assertEqual(evidence["user_request"], user_text)
        self.assertEqual(evidence["public_path"], [path_text])
        self.assertEqual(evidence["result"], final_text)
        self.assertEqual(evidence["source"], "recall_archive_projection")
        self.assertNotIn("详细工具回显", json.dumps(result, ensure_ascii=False))

    def test_mechanical_failed_turns_are_not_work_evidence(self) -> None:
        timestamp = int(datetime(2026, 7, 31, 10, 15).timestamp())
        self.write_json("conversation_history/conversations.json", {"items": []})
        self.write_json(
            "conversation_history/turns/turn-failed.json",
            {
                "id": "turn-failed",
                "conversation_id": "chat-1",
                "created_at": timestamp,
                "status": "failed",
                "final_message": "这次没有成功：模型流式响应中断，LLM stream failed: network error",
                "events": [],
            },
        )

        result = self.store.collect(report_type="daily", target_date="2026-07-31")

        self.assertEqual(result["raw_evidence_count"], 0)
        self.assertEqual(result["evidence"], [])

    def test_skill_routes_report_corrections_through_date_evidence(self) -> None:
        skill_text = (
            Path(__file__).resolve().parents[1]
            / "work_agent_skills/work-reports/SKILL.md"
        ).read_text(encoding="utf-8")

        self.assertIn("supplementing an existing", skill_text)
        self.assertIn("collect_work_report_evidence(report_type='daily'", skill_text)
        self.assertIn("Do not use `recall_chat_history(scope='compressed')`", skill_text)
        self.assertIn("current-chat compressed miss as an account-wide miss", skill_text)
        self.assertIn("Never pass\n  `.docx`", skill_text)


if __name__ == "__main__":
    unittest.main()
