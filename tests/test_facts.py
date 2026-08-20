"""留在记忆库里的，是没有更好去处的东西。

带日期的承诺不在这里——它去 Apple 提醒事项：会同步到手机、会到点响、
勾掉就等于忘掉。这三件事记忆库都得自己重造一遍，且都做得更差。
"""

from __future__ import annotations

import unittest

from work_agent_core.facts import (
    ADD,
    DECISION,
    ENTITY,
    FACT_KINDS,
    Fact,
    NOOP,
    PREFERENCE,
    SETTING,
    UPDATE,
    existing_facts,
    reconcile,
    record_facts,
)
from work_agent_core.session_log import SessionHeader, SessionLog
from work_agent_core.session_runtime import ConversationRuntime


class KindTests(unittest.TestCase):
    def test_commitments_are_deliberately_not_a_fact_kind(self) -> None:
        self.assertEqual(FACT_KINDS, {ENTITY, PREFERENCE, SETTING, DECISION})
        self.assertNotIn("commitment", FACT_KINDS)

    def test_a_fact_carries_no_deadline_field(self) -> None:
        # A date here would be a second, worse calendar.
        self.assertNotIn("due_at", Fact(kind=ENTITY, subject="x", statement="y").to_data())


class ReconcileTests(unittest.TestCase):
    """每轮追加一份新副本，是记忆库变成噪音仓库的标准路径。"""

    def setUp(self) -> None:
        self.known = [
            Fact(kind=ENTITY, subject="零次方", statement="正名零次方，曾被误写为燃气方"),
            Fact(kind=PREFERENCE, subject="报市材料", statement="用业务员口吻，不要咨询稿腔"),
        ]

    def test_an_unseen_fact_is_added(self) -> None:
        candidate = Fact(kind=ENTITY, subject="逐际动力", statement="正名逐际动力")
        self.assertEqual(reconcile(candidate, self.known)[0], ADD)

    def test_repeating_a_known_fact_changes_nothing(self) -> None:
        candidate = Fact(kind=ENTITY, subject="零次方", statement="正名零次方，曾被误写为燃气方")
        self.assertEqual(reconcile(candidate, self.known)[0], NOOP)

    def test_a_changed_statement_updates_rather_than_duplicates(self) -> None:
        candidate = Fact(kind=PREFERENCE, subject="报市材料", statement="改用条目式，一条一句")
        operation, resolved = reconcile(candidate, self.known)
        self.assertEqual(operation, UPDATE)
        self.assertEqual(resolved.statement, "改用条目式，一条一句")

    def test_the_same_subject_under_another_kind_is_a_different_fact(self) -> None:
        candidate = Fact(kind=DECISION, subject="报市材料", statement="牵头单位定为工信局")
        self.assertEqual(reconcile(candidate, self.known)[0], ADD)


class RecordingTests(unittest.TestCase):
    def test_facts_land_in_the_log_and_never_reach_the_model_history(self) -> None:
        runtime = ConversationRuntime(SessionLog(SessionHeader(session_id="c")))
        runtime.begin_turn("t1")
        runtime.begin_step(1)
        runtime.record_user("不是燃气方，是零次方")

        applied = record_facts(
            runtime,
            [Fact(kind=ENTITY, subject="零次方", statement="正名零次方，曾被误写为燃气方")],
        )
        self.assertEqual([operation for operation, _ in applied], [ADD])
        self.assertEqual(len(existing_facts(runtime.log)), 1)
        # A fact is a record about the conversation, not a message inside it.
        self.assertEqual(
            [item["role"] for item in runtime.build_request_messages("系统提示")],
            ["system", "user"],
        )

    def test_recording_the_same_fact_twice_does_not_duplicate_it(self) -> None:
        runtime = ConversationRuntime(SessionLog(SessionHeader(session_id="c")))
        runtime.begin_turn("t1")
        runtime.begin_step(1)
        fact = Fact(kind=PREFERENCE, subject="双周报", statement="动词开头，不写工作亮点")
        record_facts(runtime, [fact])
        record_facts(runtime, [fact])
        self.assertEqual(len(existing_facts(runtime.log)), 1)

    def test_incomplete_candidates_are_dropped(self) -> None:
        runtime = ConversationRuntime(SessionLog(SessionHeader(session_id="c")))
        runtime.begin_turn("t1")
        runtime.begin_step(1)
        record_facts(
            runtime,
            [
                Fact(kind=ENTITY, subject="", statement="没有主语"),
                Fact(kind=ENTITY, subject="有主语", statement=""),
                Fact(kind="commitment", subject="报市材料", statement="周三前提交"),
            ],
        )
        # The commitment is dropped too: it belongs in Reminders, not here.
        self.assertEqual(existing_facts(runtime.log), [])


if __name__ == "__main__":
    unittest.main()
