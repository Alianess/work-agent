from __future__ import annotations

from pathlib import Path
import tempfile
import time
import json
import unittest

from work_agent_core.weixin_channel import (
    WeixinChannelWorker,
    WeixinCredentials,
    WeixinGatewayManager,
    WeixinLoginSession,
    chunk_weixin_text,
)
from work_agent_core.message_channel import ChannelReply


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


if __name__ == "__main__":
    unittest.main()
