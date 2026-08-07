from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import json
import unittest
from unittest.mock import Mock, patch

from work_agent_core.auth import AuthUser
from work_agent_core import web_server
from work_agent_core.weixin_channel import (
    WeixinChannelWorker,
    WeixinCredentials,
    WeixinGatewayManager,
    WeixinLoginSession,
    chunk_weixin_text,
)
from work_agent_core.message_channel import ChannelMessage, ChannelReply


class FakeWeixinClient:
    starts = 0

    def start_login(self, local_tokens=None):
        type(self).starts += 1
        suffix = type(self).starts
        return WeixinLoginSession(
            session_id=f"session-{suffix}",
            qrcode=f"code-{suffix}",
            qrcode_url=f"https://example.invalid/qr/{suffix}",
            started_at=time.monotonic(),
        )


class ReplayAfterSendTimeoutClient:
    def __init__(self) -> None:
        self.worker: WeixinChannelWorker | None = None
        self.update_calls = 0
        self.send_calls = 0

    def get_updates(self, _credentials, _sync_buf):
        self.update_calls += 1
        if self.update_calls >= 3 and self.worker is not None:
            self.worker._stop.set()
        return {
            "ret": 0,
            "msgs": [
                {
                    "message_type": 1,
                    "message_id": "message-replay-1",
                    "from_user_id": "user-1",
                    "context_token": "ctx-1",
                    "create_time_ms": 1_790_000_000_000,
                    "item_list": [{"type": 1, "text_item": {"text": "请跟进项目"}}],
                }
            ],
            "get_updates_buf": f"buf-{self.update_calls}",
        }

    def send_text(self, _credentials, **_kwargs):
        self.send_calls += 1
        if self.send_calls == 1:
            raise TimeoutError("simulated send timeout")


class WeixinChannelTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeWeixinClient.starts = 0

    def test_missing_daily_report_with_evidence_starts_automatic_worker(self) -> None:
        user = AuthUser(id=17, username="report-user", role="member", created_at=1)
        audit = {
            "missing_daily_reports": ["2026-08-07", "2026-08-06"],
            "evidence_counts_by_date": {"2026-08-07": 3, "2026-08-06": 0},
        }
        fake_thread = Mock()
        with (
            patch.object(web_server.threading, "Thread", return_value=fake_thread) as thread_type,
            patch.object(web_server, "append_friday_proactive_message") as append_message,
        ):
            web_server.schedule_automatic_daily_reports(user, audit)

        fake_thread.start.assert_called_once_with()
        self.assertEqual(thread_type.call_args.kwargs["target"], web_server.automatic_daily_report_worker)
        self.assertEqual(thread_type.call_args.kwargs["args"], (user, "2026-08-07"))
        append_message.assert_called_once()
        self.assertIn("2026-08-06", append_message.call_args.args[0])
        gap_message = web_server.work_report_gap_message(audit)
        self.assertNotIn("你可以直接回复", gap_message)
        self.assertIn("不需要回复“补日报”", gap_message)
        with web_server.AUTO_DAILY_REPORT_LOCK:
            web_server.AUTO_DAILY_REPORT_IN_PROGRESS.discard((user.id, "2026-08-07"))

    def test_automatic_daily_report_worker_is_deduplicated(self) -> None:
        user = AuthUser(id=18, username="dedupe-user", role="member", created_at=1)
        audit = {
            "missing_daily_reports": ["2026-08-07"],
            "evidence_counts_by_date": {"2026-08-07": 1},
        }
        key = (user.id, "2026-08-07")
        with web_server.AUTO_DAILY_REPORT_LOCK:
            web_server.AUTO_DAILY_REPORT_IN_PROGRESS.add(key)
        try:
            with patch.object(web_server.threading, "Thread") as thread_type:
                web_server.schedule_automatic_daily_reports(user, audit)
            thread_type.assert_not_called()
        finally:
            with web_server.AUTO_DAILY_REPORT_LOCK:
                web_server.AUTO_DAILY_REPORT_IN_PROGRESS.discard(key)

    def test_automatic_daily_report_falls_back_on_unavailable_model(self) -> None:
        primary = web_server.ModelProfile(
            name="deepseek",
            provider="openai-compatible",
            base_url="https://example.invalid/v1",
            model="deepseek",
            api_key_env="UNUSED",
        )
        fallback = web_server.ModelProfile(
            name="gpt-fallback",
            provider="openai-compatible",
            base_url="https://fallback.invalid/v1",
            model="gpt-fallback",
            api_key_env="UNUSED",
        )
        registry = web_server.ModelRegistry(
            {primary.name: primary, fallback.name: fallback},
            primary.name,
        )
        client = Mock()
        client.chat.side_effect = [
            RuntimeError('LLM request failed with HTTP 402: {"message":"Insufficient Balance"}'),
            SimpleNamespace(content="# 2026-08-07 工作简报\n\n## 今日完成\n1. 已完成。"),
        ]
        turn_runtime = Mock()

        with patch.object(web_server, "OpenAICompatibleClient", return_value=client):
            response, selected, attempted = web_server.automatic_report_model_response(
                [{"role": "user", "content": "evidence"}],
                registry=registry,
                preferred_profile=primary,
                turn_runtime=turn_runtime,
            )

        self.assertIn("工作简报", response.content)
        self.assertEqual(selected.name, fallback.name)
        self.assertEqual(attempted, [primary.name, fallback.name])
        turn_runtime.emit.assert_called_once()

    def test_concurrent_login_start_reuses_active_qr_session(self) -> None:
        manager = WeixinGatewayManager(
            on_message=lambda user_id, message: None,
            client_factory=FakeWeixinClient,
        )
        with tempfile.TemporaryDirectory() as directory:
            state_dir = Path(directory)
            first = manager.start_login(1, state_dir)
            second = manager.start_login(1, state_dir)
            renewed = manager.start_login(1, state_dir, force=True)

        self.assertEqual(first["session_id"], second["session_id"])
        self.assertNotEqual(first["session_id"], renewed["session_id"])
        self.assertEqual(FakeWeixinClient.starts, 2)

    def test_long_messages_are_split_without_losing_content(self) -> None:
        text = "第一段。" * 1_200
        chunks = chunk_weixin_text(text, limit=500)
        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(chunks), text)
        self.assertTrue(all(len(chunk) <= 500 for chunk in chunks))

    def test_replayed_update_resends_cached_reply_without_running_agent_twice(self) -> None:
        calls = 0

        def on_message(_message):
            nonlocal calls
            calls += 1
            return ChannelReply(text="已收到，我会继续跟进。")

        with tempfile.TemporaryDirectory() as directory:
            client = ReplayAfterSendTimeoutClient()
            worker = WeixinChannelWorker(
                state_dir=Path(directory),
                credentials=WeixinCredentials(account_id="bot-1", token="token", user_id="user-1"),
                on_message=on_message,
                client=client,  # type: ignore[arg-type]
            )
            client.worker = worker
            worker._run()

            self.assertEqual(calls, 1)
            self.assertEqual(client.send_calls, 2)
            ledger = json.loads((Path(directory) / "dedupe.json").read_text(encoding="utf-8"))
            self.assertEqual(ledger["message-replay-1"]["state"], "sent")

    def test_channel_sends_complete_aggregated_react_content(self) -> None:
        reply_text = "我先读取资料。\n\n资料读取完成。"
        auth_store = Mock()
        auth_store.get_user_by_id.return_value = AuthUser(
            id=1,
            username="friday-user",
            role="member",
            created_at=1,
        )
        message = ChannelMessage(
            channel="weixin",
            account_id="bot-1",
            conversation_id="friday-main",
            sender_id="user-1",
            message_id="message-1",
            timestamp_ms=1_790_000_000_000,
            text="请读取资料",
        )

        with (
            patch.object(web_server, "get_auth_store", return_value=auth_store),
            patch.object(
                web_server,
                "run_agent_chat_events",
                return_value=iter(
                    [
                        {"event": "draft_delta", "content": "我先读取资料。"},
                        {"event": "final", "content": reply_text},
                    ]
                ),
            ),
            patch.object(web_server, "append_weixin_exchange_to_archive") as append_exchange,
        ):
            reply = web_server.handle_weixin_channel_message(1, message)

        self.assertEqual(reply.text, reply_text)
        append_exchange.assert_called_once_with(
            "friday-main",
            "请读取资料",
            reply_text,
            timestamp_ms=message.timestamp_ms,
        )


if __name__ == "__main__":
    unittest.main()
