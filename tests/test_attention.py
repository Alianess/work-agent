from __future__ import annotations

import json
import tempfile
import unittest

from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from work_agent_core import web_server
from work_agent_core.attention import (
    FunctionObserver,
    Observation,
    ObserverContext,
    ObserverRegistry,
    SpokenLedger,
    compose_message,
    select_observations,
)
from work_agent_core.observers import build_default_registry
from work_agent_core.session_log import SessionHeader, SessionLog
from work_agent_core.session_log_store import SessionLogStore
from work_agent_core.work_ledger import build_work_ledger


def context_for(root: Path) -> ObserverContext:
    return ObserverContext(
        workspace_root=root, data_root=root, now=datetime.now().astimezone()
    )


class ObserverRegistryTests(unittest.TestCase):
    def test_one_failing_observer_does_not_silence_the_rest(self) -> None:
        def broken(_context):
            raise RuntimeError("boom")

        registry = ObserverRegistry(
            [
                FunctionObserver("broken", broken),
                FunctionObserver(
                    "working",
                    lambda _c: [Observation(key="k", summary="仍然能说话", source="working")],
                ),
            ]
        )
        found = registry.run(context_for(Path(".")))
        self.assertEqual([item.summary for item in found], ["仍然能说话"])

    def test_noticing_something_new_is_a_registration(self) -> None:
        registry = ObserverRegistry()
        registry.register(
            FunctionObserver("late", lambda _c: [Observation(key="x", summary="新的关注点", source="late")])
        )
        self.assertEqual(len(registry.run(context_for(Path(".")))), 1)


class RestraintTests(unittest.TestCase):
    """An assistant that repeats itself is worse than one that stays quiet."""

    def test_low_salience_remarks_are_dropped(self) -> None:
        observations = [
            Observation(key="a", summary="重要", salience=0.9, source="s"),
            Observation(key="b", summary="无所谓", salience=0.2, source="s"),
        ]
        self.assertEqual([item.summary for item in select_observations(observations)], ["重要"])

    def test_at_most_three_remarks_per_pass(self) -> None:
        observations = [
            Observation(key=str(index), summary=f"第{index}条", salience=0.9, source="s")
            for index in range(10)
        ]
        self.assertEqual(len(select_observations(observations)), 3)

    def test_the_same_remark_is_not_repeated_within_its_quiet_window(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            ledger = SpokenLedger(Path(root) / "spoken.json")
            observation = Observation(key="k", summary="说过了", quiet_hours=10.0, source="s")
            self.assertEqual(len(ledger.filter_unsaid([observation], now=0.0)), 1)
            ledger.mark_spoken([observation], now=0.0)
            self.assertEqual(ledger.filter_unsaid([observation], now=3600.0), [])
            # Past the quiet window it may be raised again.
            self.assertEqual(len(ledger.filter_unsaid([observation], now=11 * 3600.0)), 1)


class BuiltInObserverTests(unittest.TestCase):
    def test_work_the_assistant_did_is_recalled_not_rediscovered(self) -> None:
        """Version sprawl comes from the record, not from counting files.

        The assistant made every one of these writes; inferring them back from
        whatever is left on disk loses who asked and cannot tell a revision
        from an unrelated file in the same folder.
        """

        log = SessionLog(SessionHeader(session_id="c1"))
        log.append("turn/start", {"turn_id": "t1"})
        log.append("step/start", {"step": 1})
        for name in ("初稿", "送审稿", "报市稿"):
            log.append(
                "tool/call",
                {
                    "call_id": f"call-{name}",
                    "name": "write_text_file",
                    "arguments": json.dumps({"path": f"meet_files/材料/某材料/{name}.md"}),
                },
            )
        context = ObserverContext(
            workspace_root=Path("."),
            data_root=Path("."),
            now=datetime.now().astimezone(),
            ledger=build_work_ledger(log),
        )
        found = [item for item in build_default_registry().run(context) if item.source == "material-versions"]
        self.assertEqual(len(found), 1)
        self.assertIn("3 份不同的稿子", found[0].summary)
        self.assertIn("改过 3 次", found[0].summary)

    def test_attention_ledger_does_not_recover_a_live_turn(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            store = SessionLogStore(Path(root) / "session_log.sqlite3")
            log = SessionLog(SessionHeader(session_id="live-chat"))
            log.append("turn/start", {"turn_id": "turn-live"})
            log.append("user/message", {"content": "仍在执行"})
            store.put_header(log.header)
            store.append(log.header.session_id, log.events)

            with patch.object(web_server, "get_session_log_store", return_value=store):
                web_server.account_work_ledger(SimpleNamespace(id=1))

            self.assertEqual(
                [event.type for event in store.read("live-chat")],
                ["turn/start", "user/message"],
            )

    def _ledger_for_writes(self, paths: list[str]):
        log = SessionLog(SessionHeader(session_id="c1"))
        for index, path in enumerate(paths):
            log.append(
                "tool/call",
                {
                    "call_id": f"call-{index}",
                    "name": "write_text_file",
                    "arguments": json.dumps({"path": path}),
                },
            )
        return build_work_ledger(log)

    def test_renderings_of_one_draft_are_one_document_not_three(self) -> None:
        """The md source and its docx export share a stem; that is one draft.

        Without this, every properly archived meeting (internal notes plus a
        submitted md/docx rendering) looks like version sprawl.
        """

        with tempfile.TemporaryDirectory() as root:
            names = [
                "meet_files/会议项目/某会/某会_会议纪要_工作提交版.md",
                "meet_files/会议项目/某会/某会_会议纪要_工作提交版.docx",
                "meet_files/会议项目/某会/某会_会议沟通内容整理_内部留档版.md",
            ]
            context = ObserverContext(
                workspace_root=Path(root),
                data_root=Path(root),
                now=datetime.now().astimezone(),
                ledger=self._ledger_for_writes(names),
            )
            found = [item for item in build_default_registry().run(context) if item.source == "material-versions"]
        self.assertEqual(found, [])

    def test_manifest_registered_outputs_have_a_current_draft(self) -> None:
        """A manifest's canonical_outputs IS the current-draft registry."""

        with tempfile.TemporaryDirectory() as root:
            folder = Path(root) / "meet_files" / "会议项目" / "某会"
            folder.mkdir(parents=True)
            names = [
                "meet_files/会议项目/某会/某会_会议沟通内容整理_内部留档版.md",
                "meet_files/会议项目/某会/某会_会议纪要_工作提交版.md",
                "meet_files/会议项目/某会/某会_会议纪要_工作提交版.docx",
            ]
            (folder / "manifest.json").write_text(
                json.dumps(
                    {
                        "canonical_outputs": {
                            "internal": names[0],
                            "work_md": names[1],
                            "work_docx": names[2],
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            context = ObserverContext(
                workspace_root=Path(root),
                data_root=Path(root),
                now=datetime.now().astimezone(),
                ledger=self._ledger_for_writes(names),
            )
            found = [item for item in build_default_registry().run(context) if item.source == "material-versions"]
        self.assertEqual(found, [])

    def test_stray_drafts_the_manifest_does_not_list_are_still_noticed(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root) / "meet_files" / "会议项目" / "某会"
            folder.mkdir(parents=True)
            canonical = "meet_files/会议项目/某会/某会_会议纪要_工作提交版.md"
            strays = [
                "meet_files/会议项目/某会/初稿.md",
                "meet_files/会议项目/某会/送审稿.md",
                "meet_files/会议项目/某会/报市稿.md",
            ]
            (folder / "manifest.json").write_text(
                json.dumps({"canonical_outputs": {"work_md": canonical}}, ensure_ascii=False),
                encoding="utf-8",
            )
            context = ObserverContext(
                workspace_root=Path(root),
                data_root=Path(root),
                now=datetime.now().astimezone(),
                ledger=self._ledger_for_writes([canonical, *strays]),
            )
            found = [item for item in build_default_registry().run(context) if item.source == "material-versions"]
        self.assertEqual(len(found), 1)
        self.assertIn("3 份不同的稿子", found[0].summary)

    def test_machine_bookkeeping_files_are_not_drafts(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            names = [
                "meet_files/会议项目/某会/manifest.json",
                "meet_files/会议项目/某会/某会_会议沟通内容整理_内部留档版.md",
                "meet_files/会议项目/某会/某会_会议纪要_工作提交版.md",
            ]
            context = ObserverContext(
                workspace_root=Path(root),
                data_root=Path(root),
                now=datetime.now().astimezone(),
                ledger=self._ledger_for_writes(names),
            )
            found = [item for item in build_default_registry().run(context) if item.source == "material-versions"]
        self.assertEqual(found, [])

    def test_without_a_ledger_it_says_nothing_rather_than_scanning(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root) / "meet_files" / "材料" / "某材料"
            folder.mkdir(parents=True)
            for name in ("初稿", "送审稿", "报市稿"):
                (folder / f"{name}.md").write_text("x", encoding="utf-8")
            found = build_default_registry().run(context_for(Path(root)))
        self.assertEqual([item for item in found if item.source == "material-versions"], [])

    def test_a_recording_without_minutes_is_noticed(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            attachments = Path(root) / "meet_files" / "attachments"
            attachments.mkdir(parents=True)
            (attachments / "20260817-105219-新录音 31.m4a").write_bytes(b"x")
            found = build_default_registry().run(context_for(Path(root)))
        recordings = [item for item in found if item.source == "unprocessed-recording"]
        self.assertEqual(len(recordings), 1)

    def test_a_recording_that_already_has_an_archive_is_left_alone(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            data_root = Path(root)
            attachments = data_root / "meet_files" / "attachments"
            attachments.mkdir(parents=True)
            recording = attachments / "20260817-105219-新录音 31.m4a"
            recording.write_bytes(b"x")
            archive = data_root / "meet_files" / "会议项目" / "具身智能产业发展建议材料撰写部署会"
            archive.mkdir(parents=True)
            minutes = archive / "工作提交版.md"
            minutes.write_text("会议纪要", encoding="utf-8")
            (archive / "manifest.json").write_text(
                json.dumps(
                    {
                        "transcript_path": (
                            "meet_files/asr_full/20260817-105219-新录音 31/qwen3/transcript.txt"
                        ),
                        "canonical_outputs": {
                            "work_md": str(minutes.relative_to(data_root)),
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            found = build_default_registry().run(context_for(data_root))
        self.assertEqual([item for item in found if item.source == "unprocessed-recording"], [])

    def test_an_empty_archive_folder_does_not_mark_a_recording_handled(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            data_root = Path(root)
            attachments = data_root / "meet_files" / "attachments"
            attachments.mkdir(parents=True)
            (attachments / "20260817-105219-新录音 31.m4a").write_bytes(b"x")
            (data_root / "meet_files" / "会议项目" / "新录音 31 部署会").mkdir(parents=True)
            found = build_default_registry().run(context_for(data_root))
        self.assertEqual(len([item for item in found if item.source == "unprocessed-recording"]), 1)

    def test_one_name_written_several_ways_is_noticed(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            archive = Path(root) / "meet_files" / "会议项目"
            archive.mkdir(parents=True)
            (archive / "纪要.md").write_text("燃气方的机器人小店已部署", encoding="utf-8")
            found = build_default_registry(
                alias_groups={"零次方": ["燃气方", "云智方"]}
            ).run(context_for(Path(root)))
        spellings = [item for item in found if item.source == "entity-spelling"]
        self.assertEqual(len(spellings), 1)
        self.assertIn("燃气方", spellings[0].summary)

    def test_ambiguous_version_family_asks_one_question(self) -> None:
        """名字像版本、内容不同 → 问一句，而不是替用户决定。"""

        from work_agent_core.recall.sync import RecallSync
        from work_agent_core.recall.tools import recall_index_for

        with tempfile.TemporaryDirectory() as root:
            data_root = Path(root)
            index = recall_index_for(data_root)
            sync = RecallSync(index, workspace_root=data_root)
            (data_root / "17号可研报告.md").write_text(
                "甲方案聚焦整机集成，围绕飞行平台与载荷协同设计展开论证。" * 20, encoding="utf-8"
            )
            (data_root / "20号可研报告.md").write_text(
                "乙方案聚焦空管系统，围绕通信导航监视设施建设标准展开论证。" * 20, encoding="utf-8"
            )
            sync.index_directory(data_root)
            found = build_default_registry().run(context_for(data_root))
        family = [item for item in found if item.source == "version-family"]
        self.assertEqual(len(family), 1)
        self.assertIn("17号可研报告", family[0].summary)
        self.assertIn("20号可研报告", family[0].summary)

    def test_folded_version_family_is_not_asked_about(self) -> None:
        """内容已确认版本关系、正常折叠的家族不再打扰。"""

        from work_agent_core.recall.sync import RecallSync
        from work_agent_core.recall.tools import recall_index_for

        with tempfile.TemporaryDirectory() as root:
            data_root = Path(root)
            index = recall_index_for(data_root)
            sync = RecallSync(index, workspace_root=data_root)
            body = "低空经济发展的政策依据与产业现状，涉及空域管理、适航审定和基础设施。" * 20
            (data_root / "17号可研报告.md").write_text(body + "\n\n旧版结论。", encoding="utf-8")
            (data_root / "20号可研报告.md").write_text(body + "\n\n新版结论。", encoding="utf-8")
            sync.index_directory(data_root)
            found = build_default_registry().run(context_for(data_root))
        self.assertEqual([item for item in found if item.source == "version-family"], [])


class CompositionTests(unittest.TestCase):
    def test_a_single_remark_is_said_plainly(self) -> None:
        message = compose_message([Observation(key="k", summary="录音还没整理。", source="s")])
        self.assertEqual(message, "录音还没整理。")

    def test_several_remarks_are_said_as_one_short_list(self) -> None:
        message = compose_message(
            [
                Observation(key="a", summary="甲", source="s"),
                Observation(key="b", summary="乙", source="s"),
            ]
        )
        self.assertIn("1. 甲", message)
        self.assertIn("2. 乙", message)


if __name__ == "__main__":
    unittest.main()
