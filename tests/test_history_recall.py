from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from work_agent_core.cross_chat_memory import CrossChatMemoryStore
from work_agent_core.history_recall import ChatHistoryRecall, extract_query_terms, register_history_recall_tool
from work_agent_core.retrieval_core import RetrievalBackendError
from work_agent_core.session_store import ConversationSession, SessionStore
from work_agent_core.tools import ToolRegistry


class FakeRetrievalBackend:
    enabled = True
    embedding_model = "fake-bge-m3"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            if any(marker in text for marker in ("滨湖宾馆", "出差住在哪里", "住宿地点")):
                vectors.append([1.0, 0.0, 0.0])
            elif any(marker in text for marker in ("机密代号", "隐语是什么")):
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return vectors

class FailingRetrievalBackend(FakeRetrievalBackend):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise RetrievalBackendError("embedding service offline")

class HistoryRecallTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rag_env = mock.patch.dict(
            os.environ,
            {"WORK_AGENT_HISTORY_RAG_ENABLED": "0"},
        )
        self.rag_env.start()
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.store = SessionStore(root, session_dir=root / "conversation_history" / "sessions")
        self.session = ConversationSession(
            id="conversation-test",
            messages=[
                {"role": "user", "content": "我们讨论示例制造项目，一期设备预算暂定为 280 万元。"},
                {"role": "assistant", "content": "已记录：示例制造，一期设备预算 280 万元，后续写入可行性报告。"},
                {"role": "user", "content": "公司名称纠正一下，不是示例制造厂，正式名称是示例制造。"},
                {"role": "assistant", "content": "明白，后续统一使用示例制造。"},
                {"role": "user", "content": "最近我们改为讨论会议实时转写。"},
                {"role": "assistant", "content": "好的，继续讨论实时转写。"},
                {"role": "user", "content": "我刚才问的预算是多少？"},
            ],
            summary="较早讨论过一个项目预算和公司名称纠正。",
            summary_message_count=4,
        )
        self.store.save(self.session)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()
        self.rag_env.stop()

    def test_chinese_query_generates_discriminative_bigrams(self) -> None:
        terms = extract_query_terms("回想之前示例制造的设备预算 280 万")
        self.assertIn("示例", terms)
        self.assertIn("制造", terms)
        self.assertIn("280", terms)
        self.assertNotIn("之前", terms)
        self.assertNotIn("想之", terms)

    def test_explicit_keywords_are_prioritized_before_query_noise(self) -> None:
        payload = json.loads(
            ChatHistoryRecall(self.store, self.session.id).search(
                {
                    "query": "请帮我回想之前那个事情到底是什么以及我们后来怎么处理的",
                    "keywords": ["示例制造", "280"],
                    "scope": "compressed",
                }
            )
        )
        self.assertEqual(payload["query_terms"][:3], ["示例", "例制", "制造"])
        self.assertTrue(payload["results"])

    def test_compressed_scope_returns_exact_old_passage(self) -> None:
        payload = json.loads(
            ChatHistoryRecall(self.store, self.session.id).search(
                {"query": "示例制造 一期设备预算 280 万", "scope": "compressed", "limit": 3}
            )
        )
        self.assertTrue(payload["ok"])
        self.assertFalse(payload["model_used"])
        self.assertTrue(payload["results"])
        self.assertIn("280 万元", payload["results"][0]["content"])
        self.assertLessEqual(payload["results"][0]["message_ordinal"], 4)

    def test_compressed_scope_excludes_recent_uncompressed_messages(self) -> None:
        payload = json.loads(
            ChatHistoryRecall(self.store, self.session.id).search(
                {"query": "会议 实时转写", "scope": "compressed"}
            )
        )
        self.assertEqual(payload["results"], [])

        payload_all = json.loads(
            ChatHistoryRecall(self.store, self.session.id).search(
                {"query": "会议 实时转写", "scope": "all"}
            )
        )
        self.assertTrue(payload_all["results"])
        self.assertIn("实时转写", payload_all["results"][0]["content"])

    def test_short_turn_is_recalled_with_neighboring_parent_episode(self) -> None:
        session = ConversationSession(
            id="episode-parent",
            messages=[
                {"role": "user", "content": "几点了？"},
                {"role": "assistant", "content": "现在是 20:05。"},
                {"role": "user", "content": "继续处理海盐计划。"},
                {"role": "assistant", "content": "海盐计划的下一节点是周五提交报告。"},
            ],
        )
        self.store.save(session)

        payload = json.loads(
            ChatHistoryRecall(self.store, session.id).search(
                {"query": "海盐计划 周五", "scope": "current"}
            )
        )

        result = payload["results"][0]
        self.assertEqual(result["return_mode"], "parent_episode")
        self.assertEqual(result["source_kind"], "recall_episode")
        self.assertIn("几点了", result["content"])
        self.assertIn("海盐计划", result["content"])
        self.assertTrue(result["matched_passages"])

    def test_core_tool_is_bound_to_current_conversation(self) -> None:
        registry = ToolRegistry()
        register_history_recall_tool(registry, self.store, self.session.id)
        tool = registry.get("recall_chat_history")
        payload = json.loads(tool.handler({"query": "正式名称 示例制造", "scope": "compressed"}))
        self.assertEqual(payload["conversation_id"], self.session.id)
        self.assertTrue(any("正式名称" in item["content"] for item in payload["results"]))

    def test_index_appends_new_messages_without_duplicates(self) -> None:
        recall = ChatHistoryRecall(self.store, self.session.id)
        recall.search({"query": "示例制造", "scope": "all"})
        with sqlite3.connect(recall.database_path) as connection:
            before = connection.execute(
                "SELECT count(*) FROM chat_history_fts WHERE conversation_id = ?",
                (self.session.id,),
            ).fetchone()[0]

        current = self.store.load(self.session.id)
        current.messages.extend(
            [
                {"role": "assistant", "content": "补充记录：二期预算暂未确定。"},
                {"role": "user", "content": "二期预算后来确定了吗？"},
            ]
        )
        self.store.save(current)
        payload = json.loads(recall.search({"query": "二期预算", "scope": "all"}))
        self.assertTrue(any("暂未确定" in item["content"] for item in payload["results"]))
        with sqlite3.connect(recall.database_path) as connection:
            after = connection.execute(
                "SELECT count(*) FROM chat_history_fts WHERE conversation_id = ?",
                (self.session.id,),
            ).fetchone()[0]
        self.assertGreater(after, 0)
        recall.search({"query": "二期预算", "scope": "all"})
        with sqlite3.connect(recall.database_path) as connection:
            repeated = connection.execute(
                "SELECT count(*) FROM chat_history_fts WHERE conversation_id = ?",
                (self.session.id,),
            ).fetchone()[0]
        self.assertEqual(repeated, after)

    def test_auto_scope_searches_project_chats_and_preserves_project_isolation(self) -> None:
        project_current = ConversationSession(
            id="project-current",
            messages=[{"role": "user", "content": "继续这个项目。"}],
            metadata={"project_id": "project-one"},
        )
        same_project = ConversationSession(
            id="project-same",
            messages=[{"role": "user", "content": "阿尔法电机采购数量最终确定为 73 台。"}],
            metadata={"project_id": "project-one"},
        )
        other_project = ConversationSession(
            id="project-other",
            messages=[{"role": "user", "content": "贝塔控制器的内部代号是 BX-99。"}],
            metadata={"project_id": "project-two"},
        )
        for session in (project_current, same_project, other_project):
            self.store.save(session)
        archive_path = self.store.session_dir.parent / "conversations.json"
        archive_path.write_text(
            json.dumps(
                {
                    "items": [
                        {"id": "project-current", "title": "当前项目聊天", "projectId": "project-one", "messages": []},
                        {"id": "project-same", "title": "电机采购讨论", "projectId": "project-one", "messages": []},
                        {"id": "project-other", "title": "其他项目机密", "projectId": "project-two", "messages": []},
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        project_recall = ChatHistoryRecall(self.store, "project-current", project_id="project-one")
        same_payload = json.loads(project_recall.search({"query": "阿尔法电机 73 台", "scope": "auto"}))
        self.assertEqual(same_payload["scope"], "project")
        self.assertEqual(same_payload["results"][0]["conversation_title"], "电机采购讨论")
        self.assertEqual(same_payload["results"][0]["project_id"], "project-one")

        isolated_payload = json.loads(
            project_recall.search({"query": "贝塔控制器 BX-99", "scope": "account"})
        )
        self.assertEqual(isolated_payload["scope"], "project")
        self.assertEqual(isolated_payload["results"], [])

        outside_payload = json.loads(
            ChatHistoryRecall(self.store, self.session.id).search(
                {"query": "贝塔控制器 BX-99", "scope": "account"}
            )
        )
        self.assertEqual(outside_payload["results"], [])

    def test_auto_scope_outside_projects_searches_all_account_chats(self) -> None:
        other = ConversationSession(
            id="account-other",
            messages=[{"role": "assistant", "content": "跨聊天记忆测试暗号是海盐柠檬 42。"}],
        )
        self.store.save(other)
        payload = json.loads(
            ChatHistoryRecall(self.store, self.session.id).search(
                {"query": "海盐柠檬 42", "scope": "auto"}
            )
        )
        self.assertEqual(payload["scope"], "account")
        self.assertEqual(payload["results"][0]["conversation_id"], "account-other")

    def test_hybrid_dense_recall_finds_semantic_match_without_shared_terms(self) -> None:
        other = ConversationSession(
            id="semantic-source",
            messages=[
                {
                    "role": "assistant",
                    "content": "差旅安排已经确认：统一入住滨湖宾馆，接送车辆早上八点发车。",
                }
            ],
            metadata={"title": "差旅安排"},
        )
        self.store.save(other)
        payload = json.loads(
            ChatHistoryRecall(
                self.store,
                self.session.id,
                retrieval_backend=FakeRetrievalBackend(),
            ).search(
                {"query": "上次出差住在哪里", "scope": "account"}
            )
        )

        self.assertEqual(payload["retrieval_status"]["mode"], "hybrid")
        self.assertTrue(payload["retrieval_status"]["dense"])
        result = next(
            item for item in payload["results"] if item["conversation_id"] == "semantic-source"
        )
        self.assertIn("滨湖宾馆", result["content"])
        self.assertIn("dense", result["matched_by"])

    def test_dense_recall_uses_loaded_session_whitelist_for_project_isolation(self) -> None:
        current = ConversationSession(
            id="dense-project-current",
            messages=[{"role": "user", "content": "继续。"}],
            metadata={"project_id": "project-one"},
        )
        same_project = ConversationSession(
            id="dense-project-same",
            messages=[{"role": "assistant", "content": "本项目机密代号是松针计划。"}],
            metadata={"project_id": "project-one"},
        )
        other_project = ConversationSession(
            id="dense-project-other",
            messages=[{"role": "assistant", "content": "其他项目机密代号是海浪计划。"}],
            metadata={"project_id": "project-two"},
        )
        for session in (current, same_project, other_project):
            self.store.save(session)

        payload = json.loads(
            ChatHistoryRecall(
                self.store,
                current.id,
                project_id="project-one",
                retrieval_backend=FakeRetrievalBackend(),
            ).search(
                {"query": "之前的隐语是什么", "scope": "auto", "limit": 8}
            )
        )

        ids = {item["conversation_id"] for item in payload["results"]}
        self.assertIn("dense-project-same", ids)
        self.assertNotIn("dense-project-other", ids)

    def test_durable_session_project_wins_over_stale_archive_project(self) -> None:
        current = ConversationSession(
            id="stale-current",
            messages=[{"role": "user", "content": "继续。"}],
            metadata={"project_id": "project-one"},
        )
        moved = ConversationSession(
            id="stale-moved",
            messages=[{"role": "assistant", "content": "迁移后的项目机密代号是海浪计划。"}],
            metadata={"project_id": "project-two"},
        )
        self.store.save(current)
        self.store.save(moved)
        (self.store.session_dir.parent / "conversations.json").write_text(
            json.dumps(
                {
                    "items": [
                        {"id": current.id, "projectId": "project-one"},
                        {"id": moved.id, "projectId": "project-one"},
                    ]
                }
            ),
            encoding="utf-8",
        )

        payload = json.loads(
            ChatHistoryRecall(
                self.store,
                current.id,
                project_id="project-one",
                retrieval_backend=FakeRetrievalBackend(),
            ).search(
                {"query": "项目的隐语是什么", "scope": "auto", "limit": 8}
            )
        )

        self.assertNotIn(
            moved.id,
            {item["conversation_id"] for item in payload["results"]},
        )

    def test_model_failure_degrades_to_bm25_without_losing_exact_recall(self) -> None:
        payload = json.loads(
            ChatHistoryRecall(
                self.store,
                self.session.id,
                retrieval_backend=FailingRetrievalBackend(),
            ).search(
                {"query": "示例制造 280 万", "scope": "compressed"}
            )
        )

        self.assertEqual(payload["retrieval_status"]["mode"], "bm25_fallback")
        self.assertTrue(payload["retrieval_status"]["degraded"])
        self.assertFalse(payload["model_used"])
        self.assertTrue(any("280 万元" in item["content"] for item in payload["results"]))

    def test_explicit_core_memory_is_returned_with_source_and_correction_state(self) -> None:
        CrossChatMemoryStore(self.store).upsert_many(
            [{"kind": "fact", "content": "示例制造一期设备预算为 280 万元。"}],
            conversation_id="memory-source", conversation_title="预算讨论", source_excerpt="预算讨论原文。",
            state="explicit",
        )
        payload = json.loads(
            ChatHistoryRecall(self.store, self.session.id).search(
                {"query": "示例制造 预算 280 万", "scope": "auto"}
            )
        )

        memory = next(item for item in payload["memory_results"] if item["conversation_id"] == "memory-source")
        self.assertEqual(memory["conversation_title"], "预算讨论")
        self.assertEqual(memory["state"], "explicit")
        self.assertIn("280 万元", memory["content"])


if __name__ == "__main__":
    unittest.main()
