from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from work_agent_core.config import ModelProfile
from work_agent_core.llm import LLMResponse
from work_agent_core.skills.meeting_minutes import MeetingMinutesSkill


class _MinutesClient:
    """Deterministic in-process stand-in for the two existing minutes prompts."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def chat(self, messages: list[dict[str, str]], *, profile: ModelProfile) -> LLMResponse:
        self.prompts.append(messages[-1]["content"])
        content = (
            "# 内部留档版\n\n- 事实：已确认继续推进。\n"
            if len(self.prompts) == 1
            else "# 工作提交版\n\n## 一、会议结论\n\n继续推进后续对接。\n"
        )
        return LLMResponse(content=content, raw={})


class RealtimeMeetingHandoffTests(unittest.TestCase):
    def test_existing_realtime_transcript_reuses_meeting_minutes_skill_without_retranscribing(self) -> None:
        runtime_root = Path(__file__).resolve().parents[1]
        profile = ModelProfile(
            name="meeting-handoff-test",
            provider="openai-compatible",
            base_url="https://example.invalid/v1",
            model="test-model",
            api_key_env="UNUSED",
        )
        with tempfile.TemporaryDirectory() as directory:
            account_root = Path(directory)
            transcript = account_root / "meet_files" / "realtime_transcripts" / "session-1.md"
            transcript.parent.mkdir(parents=True)
            transcript.write_text("# 实时转写\n\n王工：下周确认交付时间。\n", encoding="utf-8")
            client = _MinutesClient()
            skill = MeetingMinutesSkill(
                workspace_root=account_root,
                runtime_workspace_root=runtime_root,
                client=client,  # type: ignore[arg-type]
                profile=profile,
            )

            payload = json.loads(
                skill.run(
                    {
                        "input_path": "meet_files/realtime_transcripts/session-1.md",
                        "output_dir": "meet_files",
                        "meeting_name": "实时会议",
                    }
                )
            )

            self.assertEqual(len(client.prompts), 2)
            self.assertIn("已使用现有转写文本，未重新进行音频转写", payload["processing_note"])
            self.assertTrue(Path(payload["asr_markdown_path"]).is_file())
            self.assertTrue(Path(payload["internal_path"]).is_file())
            self.assertTrue(Path(payload["work_markdown_path"]).is_file())
            self.assertTrue(Path(payload["work_docx_path"]).is_file())
            manifest = json.loads((account_root / payload["manifest_path"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["source_path"], "meet_files/realtime_transcripts/session-1.md")
            self.assertEqual(
                manifest["canonical_outputs"]["asr"],
                "meet_files/会议项目/实时会议/实时会议_会议沟通内容整理_ASR转写稿_Qwen3.md",
            )


if __name__ == "__main__":
    unittest.main()
