"""删除必须落到每一处存储，否则"删了"只是从列表里消失。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from work_agent_core.recall.index import RecallIndex
from work_agent_core.recall.chunking import build_chat_tree
from work_agent_core.session_log import SessionHeader, SessionLog
from work_agent_core.session_log_store import SessionLogStore


class EventLogDeleteTests(unittest.TestCase):
    def test_deleting_a_session_removes_its_events_and_header(self) -> None:
        store = SessionLogStore(Path(tempfile.mkdtemp()) / "log.sqlite3")
        for session_id in ("keep", "drop"):
            log = SessionLog(SessionHeader(session_id=session_id))
            log.append("turn/start", {"turn_id": "t1"})
            log.append("step/start", {"step": 1})
            log.append("user/message", {"content": f"{session_id} 的内容"})
            store.put_header(log.header)
            store.append(session_id, log.events)

        removed = store.delete("drop")

        self.assertGreater(removed, 0)
        self.assertEqual(store.list_sessions(), ["keep"])
        self.assertIsNone(store.header("drop"))
        self.assertEqual(store.read("drop"), [])
        # 删一条不能碰到另一条
        self.assertTrue(store.read("keep"))

    def test_deleting_something_that_is_not_there_is_not_an_error(self) -> None:
        store = SessionLogStore(Path(tempfile.mkdtemp()) / "log.sqlite3")
        self.assertEqual(store.delete("nope"), 0)


class RecallDeleteTests(unittest.TestCase):
    def test_forgetting_a_conversation_clears_its_nodes_and_search_rows(self) -> None:
        index = RecallIndex(Path(tempfile.mkdtemp()) / "recall.sqlite3")
        for name in ("keep", "drop"):
            index.upsert_tree(
                build_chat_tree(
                    source_id=f"chat:{name}",
                    title=name,
                    turns=[
                        {
                            "title": "第 1 轮",
                            "occurred_at": 1,
                            "messages": [{"role": "user", "content": f"{name} 里提到中试基地的安排"}],
                        }
                    ],
                )
            )
        self.assertEqual(len(index.lexical_candidates("中试基地")), 2)

        index.forget_source("chat:drop")

        hits = index.lexical_candidates("中试基地")
        self.assertEqual(len(hits), 1)
        self.assertTrue(hits[0].startswith("chat:keep"))
        self.assertEqual(
            [row for row in index.stats().items() if row[0] == "sources"], [("sources", 1)]
        )


if __name__ == "__main__":
    unittest.main()
