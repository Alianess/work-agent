from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import json
import os
import re
import urllib.error
import urllib.request

from .formatting import format_timestamp, get_segments, get_transcript_text


SUMMARY_KEYWORDS = {
    "目标": 3,
    "问题": 3,
    "风险": 3,
    "方案": 3,
    "计划": 3,
    "结论": 3,
    "决定": 3,
    "确认": 2,
    "需要": 2,
    "负责": 2,
    "推进": 2,
    "完成": 2,
    "上线": 2,
    "交付": 2,
    "预算": 2,
    "客户": 2,
}

ACTION_KEYWORDS = [
    "负责",
    "跟进",
    "推进",
    "完成",
    "提交",
    "整理",
    "确认",
    "对齐",
    "排期",
    "会后",
    "明天",
    "今天",
    "本周",
    "下周",
    "deadline",
    "todo",
    "action",
]

DECISION_KEYWORDS = [
    "决定",
    "确定",
    "结论",
    "同意",
    "通过",
    "采用",
    "先做",
    "暂不",
    "不做",
    "优先",
]

RISK_KEYWORDS = [
    "风险",
    "问题",
    "卡点",
    "阻塞",
    "噪音",
    "听不清",
    "复核",
    "延迟",
    "来不及",
    "不确定",
    "依赖",
    "缺少",
]


def build_local_minutes(
    result: dict[str, Any],
    *,
    source_name: str,
    transcript_path: str | Path | None = None,
    srt_path: str | Path | None = None,
) -> str:
    segments = get_segments(result)
    transcript = get_transcript_text(result)
    sentences = split_sentences(transcript)

    summary = pick_key_sentences(sentences, limit=6)
    decisions = pick_keyword_sentences(sentences, DECISION_KEYWORDS, limit=8)
    actions = pick_keyword_sentences(sentences, ACTION_KEYWORDS, limit=10)
    risks = pick_keyword_sentences(sentences, RISK_KEYWORDS, limit=8)
    timeline = build_timeline(segments)
    review_points = find_review_points(segments)
    duration = max((float(segment.get("end") or 0.0) for segment in segments), default=0.0)

    lines = [
        "# 会议纪要",
        "",
        f"- 音频文件: `{source_name}`",
        f"- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 录音时长: {format_timestamp(duration)}",
        f"- 转写片段数: {len(segments)}",
        "",
        "## 核心摘要",
        "",
    ]
    lines.extend(_bullet_lines(summary, fallback="未抽取到稳定摘要，请查看完整转写。"))

    lines.extend(["", "## 议题时间线", ""])
    if timeline:
        lines.extend(f"- `{start}` {text}" for start, text in timeline)
    else:
        lines.append("- 未生成时间线。")

    lines.extend(["", "## 决议与结论", ""])
    lines.extend(_bullet_lines(decisions, fallback="未识别到明确决议。"))

    lines.extend(["", "## 待办事项", ""])
    if actions:
        lines.extend(f"- [ ] {item}" for item in actions)
    else:
        lines.append("- [ ] 未识别到明确待办，请人工补充负责人和截止时间。")

    lines.extend(["", "## 风险与待确认", ""])
    lines.extend(_bullet_lines(risks, fallback="未识别到明显风险或阻塞点。"))

    lines.extend(["", "## 需人工复核的转写片段", ""])
    if review_points:
        lines.extend(f"- `{stamp}` {text}" for stamp, text in review_points)
    else:
        lines.append("- 未发现明显低置信片段。")

    output_lines = []
    if transcript_path:
        output_lines.append(f"- 完整转写: `{transcript_path}`")
    if srt_path:
        output_lines.append(f"- 字幕文件: `{srt_path}`")
    if output_lines:
        lines.extend(["", "## 输出文件", ""])
        lines.extend(output_lines)

    lines.extend(
        [
            "",
            "## 说明",
            "",
            "- 本纪要由本地规则自动抽取生成，适合快速过稿；正式对外发送前建议人工核对。",
            "- 当前转写链路不提供说话人分离，本工具当前不自动标注发言人。",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def build_llm_minutes(
    transcript: str,
    *,
    source_name: str,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: int = 120,
) -> str:
    api_key = api_key or os.getenv("MEETING_MINUTES_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing MEETING_MINUTES_LLM_API_KEY or OPENAI_API_KEY.")

    model = model or os.getenv("MEETING_MINUTES_LLM_MODEL")
    if not model:
        raise RuntimeError("Missing MEETING_MINUTES_LLM_MODEL.")

    base_url = (
        base_url
        or os.getenv("MEETING_MINUTES_LLM_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or "https://api.openai.com/v1"
    ).rstrip("/")
    endpoint = f"{base_url}/chat/completions"

    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是专业会议纪要助手。请只基于转写内容生成结构化纪要，"
                    "不要编造未出现的决策、负责人或时间。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"音频文件: {source_name}\n\n"
                    "请输出 Markdown，包含：核心摘要、议题时间线、决议与结论、"
                    "待办事项、风险与待确认、需要人工复核的点。\n\n"
                    f"会议转写如下：\n{transcript}"
                ),
            },
        ],
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM request failed with HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"LLM request failed: {error}") from error

    parsed = json.loads(body)
    content = parsed["choices"][0]["message"]["content"].strip()
    if not content:
        raise RuntimeError("LLM returned an empty meeting minutes response.")
    return content + "\n"


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    pieces = re.split(r"(?<=[。！？!?；;])\s*", text)
    if len(pieces) <= 1:
        pieces = re.split(r"(?<=\.)\s+", text)
    return [piece.strip(" ，,") for piece in pieces if len(piece.strip(" ，,")) >= 8]


def pick_key_sentences(sentences: list[str], *, limit: int) -> list[str]:
    ranked = sorted(
        ((score_sentence(sentence), index, sentence) for index, sentence in enumerate(sentences)),
        key=lambda item: (-item[0], item[1]),
    )
    return dedupe([sentence for score, _, sentence in ranked if score > 0], limit=limit)


def pick_keyword_sentences(
    sentences: list[str],
    keywords: list[str],
    *,
    limit: int,
) -> list[str]:
    matches = [
        sentence
        for sentence in sentences
        if any(keyword.lower() in sentence.lower() for keyword in keywords)
    ]
    return dedupe(matches, limit=limit)


def score_sentence(sentence: str) -> int:
    score = 0
    for keyword, weight in SUMMARY_KEYWORDS.items():
        if keyword in sentence:
            score += weight
    if 20 <= len(sentence) <= 180:
        score += 1
    return score


def dedupe(items: list[str], *, limit: int) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        key = re.sub(r"\W+", "", item.lower())[:48]
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(item)
        if len(output) >= limit:
            break
    return output


def build_timeline(
    segments: list[dict[str, Any]],
    *,
    bucket_seconds: int = 300,
    limit: int = 12,
) -> list[tuple[str, str]]:
    if not segments:
        return []

    buckets: dict[int, list[str]] = {}
    for segment in segments:
        start = float(segment.get("start") or 0.0)
        bucket = int(start // bucket_seconds) * bucket_seconds
        text = str(segment.get("text") or "").strip()
        if text:
            buckets.setdefault(bucket, []).append(text)

    timeline: list[tuple[str, str]] = []
    for bucket in sorted(buckets)[:limit]:
        bucket_text = " ".join(buckets[bucket])
        sentences = split_sentences(bucket_text)
        picked = pick_key_sentences(sentences, limit=2)
        if picked:
            summary = "；".join(picked)
        else:
            summary = bucket_text[:180].strip()
            if len(bucket_text) > 180:
                summary += "..."
        timeline.append((format_timestamp(float(bucket)), summary))
    return timeline


def find_review_points(
    segments: list[dict[str, Any]],
    *,
    limit: int = 8,
) -> list[tuple[str, str]]:
    points: list[tuple[str, str]] = []
    for segment in segments:
        avg_logprob = segment.get("avg_logprob")
        no_speech_prob = segment.get("no_speech_prob")
        is_low_confidence = (
            isinstance(avg_logprob, (int, float))
            and avg_logprob < -0.9
        ) or (
            isinstance(no_speech_prob, (int, float))
            and no_speech_prob > 0.6
        )
        if not is_low_confidence:
            continue
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        stamp = format_timestamp(float(segment.get("start") or 0.0))
        points.append((stamp, text))
        if len(points) >= limit:
            break
    return points


def _bullet_lines(items: list[str], *, fallback: str) -> list[str]:
    if not items:
        return [f"- {fallback}"]
    return [f"- {item}" for item in items]
