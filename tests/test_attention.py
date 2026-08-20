from __future__ import annotations

import json
import tempfile
import unittest

from datetime import datetime, timedelta
from pathlib import Path

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
        self.assertIn("3 个文件", found[0].summary)
        self.assertIn("改过 3 次", found[0].summary)

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
            attachments = Path(root) / "meet_files" / "attachments"
            attachments.mkdir(parents=True)
            (attachments / "20260817-105219-新录音 31.m4a").write_bytes(b"x")
            archive = Path(root) / "meet_files" / "会议项目" / "新录音 31 部署会"
            archive.mkdir(parents=True)
            found = build_default_registry().run(context_for(Path(root)))
        self.assertEqual([item for item in found if item.source == "unprocessed-recording"], [])

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
