from __future__ import annotations

import base64
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from work_agent_core import web_server


class RealtimeTranscriptionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.storage_root = Path(self.temporary_directory.name).resolve()
        self.patches = [
            patch.object(web_server, "account_workspace_root", return_value=self.storage_root),
            patch.object(web_server, "require_executable", return_value="/usr/bin/ffmpeg"),
            patch.object(web_server, "run_process", return_value=None),
            patch.object(
                web_server,
                "wav_signal_stats",
                return_value={
                    "duration_ms": 8000,
                    "rms": 520.0,
                    "max_rms": 900.0,
                    "active_ratio": 0.6,
                    "has_voice_like_signal": True,
                },
            ),
            patch.object(
                web_server,
                "transcribe_with_asr_worker",
                return_value={"text": "这是测试转写。", "elapsed_ms": 320},
            ),
        ]
        for item in self.patches:
            item.start()

    def tearDown(self) -> None:
        for item in reversed(self.patches):
            item.stop()
        self.temporary_directory.cleanup()

    def speech_payload(self, **overrides: object) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": "segment.webm",
            "mime_type": "audio/webm",
            "content_base64": base64.b64encode(b"valid-container-placeholder").decode("ascii"),
            "use_denoise": True,
            "skip_if_silent": True,
        }
        payload.update(overrides)
        return payload

    def test_realtime_segment_uses_stable_session_folder_and_account_relative_paths(self) -> None:
        result = web_server.transcribe_speech_payload(
            self.speech_payload(
                realtime_session_id="rt-session-12345678",
                segment_index=3,
                started_at=1000,
                finished_at=9000,
                title="项目启动会",
            )
        )

        self.assertEqual(result["text"], "这是测试转写。")
        self.assertEqual(result["realtime_session_id"], "rt-session-12345678")
        self.assertEqual(result["segment_index"], 3)
        self.assertTrue(result["audio_path"].startswith("meet_files/realtime_sessions/rt-session-12345678/segments/0003/"))
        self.assertNotIn(str(self.storage_root), result["audio_path"])

        restored = web_server.realtime_transcript_session_payload("rt-session-12345678")
        self.assertEqual(restored["title"], "项目启动会")
        self.assertEqual(restored["segments"][0]["status"], "complete")
        self.assertEqual(restored["segments"][0]["text"], "这是测试转写。")

    def test_regular_voice_requests_do_not_overwrite_same_second(self) -> None:
        first = web_server.transcribe_speech_payload(self.speech_payload())
        second = web_server.transcribe_speech_payload(self.speech_payload())

        self.assertNotEqual(first["audio_path"], second["audio_path"])
        self.assertNotEqual(Path(first["audio_path"]).parent, Path(second["audio_path"]).parent)

    def test_failed_realtime_segment_is_checkpointed_for_retry(self) -> None:
        with patch.object(web_server, "run_process", side_effect=RuntimeError("音频容器无效")):
            with self.assertRaisesRegex(RuntimeError, "音频容器无效"):
                web_server.transcribe_speech_payload(
                    self.speech_payload(
                        realtime_session_id="rt-session-abcdefgh",
                        segment_index=1,
                        title="失败恢复测试",
                    )
                )

        restored = web_server.realtime_transcript_session_payload("rt-session-abcdefgh")
        segment = restored["segments"][0]
        self.assertEqual(segment["status"], "error")
        self.assertIn("音频容器无效", segment["error"])
        self.assertTrue((self.storage_root / segment["audio_path"]).is_file())

    def test_saved_markdown_is_complete_skill_input_and_updates_session_manifest(self) -> None:
        result = web_server.save_realtime_transcript_payload(
            {
                "title": "项目启动会",
                "session_id": "rt-session-save1234",
                "segments": [
                    {
                        "index": 1,
                        "text": "第一段内容。",
                        "started_at": 1000,
                        "finished_at": 5000,
                        "audio_path": "meet_files/realtime_sessions/rt-session-save1234/segments/0001/segment.webm",
                        "transcript_path": "meet_files/realtime_sessions/rt-session-save1234/segments/0001/asr/worker/transcript.txt",
                        "engine": "qwen3-asr-worker",
                        "asr_elapsed_ms": 300,
                    },
                    {
                        "index": 2,
                        "text": "第二段内容。",
                        "started_at": 5000,
                        "finished_at": 9000,
                    },
                ],
            }
        )

        output_path = self.storage_root / result["path"]
        content = output_path.read_text(encoding="utf-8")
        self.assertIn("## 完整转写", content)
        self.assertIn("第一段内容。\n\n第二段内容。", content)
        self.assertEqual(result["duration_ms"], 8000)

        restored = web_server.realtime_transcript_session_payload("rt-session-save1234")
        self.assertEqual(restored["status"], "saved")
        self.assertEqual(restored["output_path"], result["path"])
        self.assertEqual(len(restored["segments"]), 2)

    def test_existing_error_segment_blocks_partial_canonical_transcript(self) -> None:
        session_id = "rt-session-guard123"
        web_server.update_realtime_transcript_manifest(
            session_id,
            title="完整性保护测试",
            segment={"index": 1, "status": "complete", "text": "第一段。"},
            status="recording",
        )
        web_server.update_realtime_transcript_manifest(
            session_id,
            title="完整性保护测试",
            segment={"index": 2, "status": "error", "text": "", "error": "ASR timeout"},
            status="error",
        )

        with self.assertRaisesRegex(ValueError, "实时转写尚不完整"):
            web_server.save_realtime_transcript_payload(
                {
                    "title": "完整性保护测试",
                    "session_id": session_id,
                    "segments": [{"index": 1, "text": "第一段。"}],
                }
            )

    def test_invalid_realtime_session_id_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "会话标识无效"):
            web_server.transcribe_speech_payload(
                self.speech_payload(
                    realtime_session_id="../../escape",
                    segment_index=1,
                )
            )


if __name__ == "__main__":
    unittest.main()
