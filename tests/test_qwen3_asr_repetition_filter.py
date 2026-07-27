from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "meeting_audio_minutes" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from transcribe_qwen3_asr_mlx import filter_pathological_repetitions


def test_preserves_normal_spoken_repetition() -> None:
    text = "对对对，这个方案可以。好的，好的，谢谢谢谢。"

    filtered, events = filter_pathological_repetitions(text)

    assert filtered == text
    assert events == []


def test_collapses_single_character_decoding_loop() -> None:
    filtered, events = filter_pathological_repetitions("您是？" + "对" * 100 + "会议结束。")

    assert filtered == "您是？对对对会议结束。"
    assert events[0]["unit"] == "对"
    assert events[0]["repetitions"] == 100


def test_collapses_multi_character_decoding_loop() -> None:
    filtered, events = filter_pathological_repetitions("要不要吃饭？" + "不用" * 80 + "回头见。")

    assert filtered == "要不要吃饭？不用不用不用回头见。"
    assert events[0]["unit"] == "不用"
    assert events[0]["removed_chars"] == 154
