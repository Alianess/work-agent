#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Any
import json
import re
import sys

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from work_agent_core.memory import estimate_context_tokens
from work_agent_core.recall_archive import (
    compact_messages_for_archive,
    extract_completed_turns,
    render_episode_text,
)


SECRET_PATTERNS = (
    (re.compile(r"sk-[A-Za-z0-9_-]{10,}"), "sk-***"),
    (
        re.compile(
            r"(?i)(api[_-]?key|token|secret|password)([\"']?\s*[:=]\s*)([\"']?)[^\\s,\"'}]+"
        ),
        r"\1\2\3***",
    ),
    (re.compile(r"(?i)(bearer\\s+)[A-Za-z0-9._-]{12,}"), r"\1***"),
)


def main() -> int:
    parser = ArgumentParser(description="Evaluate deterministic ReAct recall-archive compression.")
    parser.add_argument("--session-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    args = parser.parse_args()

    session_path = Path(args.session_path).resolve()
    payload = json.loads(session_path.read_text(encoding="utf-8"))
    messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []
    turns = extract_completed_turns(messages)
    if args.start is not None:
        turns = [
            turn for turn in turns
            if int(turn.get("start_message_index") or 0) == args.start
            and (args.end is None or int(turn.get("end_message_index") or 0) == args.end)
        ]
    else:
        turns = [
            turn for turn in turns
            if turn.get("path_texts") and turn.get("actions")
        ]
        turns.sort(
            key=lambda turn: estimate_context_tokens(
                serialize_messages(
                    messages[
                        int(turn["start_message_index"]):int(turn["end_message_index"])
                    ]
                )
            ),
            reverse=True,
        )
    if not turns:
        raise SystemExit("No matching completed ReAct turn found.")

    turn = turns[0]
    start = int(turn["start_message_index"])
    end = int(turn["end_message_index"])
    raw_messages = messages[start:end]
    compacted_messages = compact_messages_for_archive(raw_messages)
    raw_text = serialize_messages(raw_messages)
    compacted_runtime_text = serialize_messages(compacted_messages)
    archive_text = render_episode_text(turn)

    raw_tokens = estimate_context_tokens(raw_text)
    runtime_tokens = estimate_context_tokens(compacted_runtime_text)
    archive_tokens = estimate_context_tokens(archive_text)
    raw_chars = len(raw_text)
    runtime_chars = len(compacted_runtime_text)
    archive_chars = len(archive_text)
    raw_tool_chars = sum(
        len(str(message.get("content") or ""))
        for message in raw_messages
        if message.get("role") == "tool"
    )
    compacted_tool_chars = sum(
        len(str(message.get("content") or ""))
        for message in compacted_messages
        if message.get("role") == "tool"
    )
    tool_names = extract_tool_names(raw_messages)

    checks = {
        "全部用户消息逐字存在": all(
            str(text or "") in archive_text for text in turn.get("user_texts") or []
        ),
        "全部公开实施路径逐字存在": all(
            str(text or "") in archive_text for text in turn.get("path_texts") or []
        ),
        "最终答复逐字存在": all(
            str(text or "") in archive_text for text in turn.get("final_texts") or []
        ),
        "全部工具名称仍可定位": all(name in archive_text for name in tool_names),
        "原始终端与工具长回显未进入工作片段": raw_tool_chars > 0
        and raw_tool_chars > compacted_tool_chars,
        "对话与工具事件保持原始先后顺序": events_appear_in_order(
            (turn.get("turns") or [{}])[0].get("events") or [],
            archive_text,
        ),
        "未把执行轨迹、最终答复或引用另拆成汇总区": not any(
            heading in archive_text
            for heading in ("执行轨迹：", "最终答复：", "关键产物与引用：")
        ),
    }

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "原始ReAct消息_脱敏副本.json"
    archive_path = output_dir / "压缩后的可回忆工作片段.md"
    report_path = output_dir / "压缩验收报告.md"

    raw_path.write_text(redact_secrets(raw_text), encoding="utf-8")
    archive_path.write_text(archive_text + "\n", encoding="utf-8")
    report_path.write_text(
        build_report(
            session_path=session_path,
            start=start,
            end=end,
            raw_path=raw_path,
            archive_path=archive_path,
            raw_message_count=len(raw_messages),
            public_path_count=len(turn.get("path_texts") or []),
            action_count=len(turn.get("actions") or []),
            tool_name_count=len(tool_names),
            raw_chars=raw_chars,
            runtime_chars=runtime_chars,
            archive_chars=archive_chars,
            raw_tokens=raw_tokens,
            runtime_tokens=runtime_tokens,
            archive_tokens=archive_tokens,
            raw_tool_chars=raw_tool_chars,
            compacted_tool_chars=compacted_tool_chars,
            checks=checks,
            archive_text=archive_text,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "report": str(report_path),
                "raw_sample": str(raw_path),
                "archive_sample": str(archive_path),
                "message_range": [start, end],
                "raw_estimated_tokens": raw_tokens,
                "archive_estimated_tokens": archive_tokens,
                "archive_reduction_percent": round(reduction(raw_tokens, archive_tokens), 2),
                "runtime_reduction_percent": round(reduction(raw_tokens, runtime_tokens), 2),
                "checks": checks,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def build_report(
    *,
    session_path: Path,
    start: int,
    end: int,
    raw_path: Path,
    archive_path: Path,
    raw_message_count: int,
    public_path_count: int,
    action_count: int,
    tool_name_count: int,
    raw_chars: int,
    runtime_chars: int,
    archive_chars: int,
    raw_tokens: int,
    runtime_tokens: int,
    archive_tokens: int,
    raw_tool_chars: int,
    compacted_tool_chars: int,
    checks: dict[str, bool],
    archive_text: str,
) -> str:
    check_lines = "\n".join(
        f"- {'通过' if passed else '失败'}：{name}" for name, passed in checks.items()
    )
    return f"""# 历史工作片段压缩验收报告

## 样本

- 源会话：`{session_path}`
- 原始消息范围：`[{start}, {end})`
- 原始 ReAct 消息：{raw_message_count} 条
- 公开实施路径原文：{public_path_count} 段
- 工具动作：{action_count} 条，涉及 {tool_name_count} 种工具
- [查看完整原始 ReAct 消息脱敏副本]({raw_path.name})
- [查看完整压缩后工作片段]({archive_path.name})

## 压缩结果

| 形态 | 字符数 | 估算 tokens | 相对原始减少 |
|---|---:|---:|---:|
| 原始 ReAct messages | {raw_chars:,} | {raw_tokens:,} | 0% |
| 折叠工具细节后的合法 runtime messages | {runtime_chars:,} | {runtime_tokens:,} | {reduction(raw_tokens, runtime_tokens):.2f}% |
| 长期检索使用的可回忆工作片段 | {archive_chars:,} | {archive_tokens:,} | {reduction(raw_tokens, archive_tokens):.2f}% |

- 原始工具结果正文：{raw_tool_chars:,} 字符
- 折叠后工具结果正文：{compacted_tool_chars:,} 字符
- 长期工作片段不复制工具长回显；动作、状态和关键引用留在原对话位置。

## 保真检查

{check_lines}

## 压缩后的完整工作片段

{archive_text}
"""


def serialize_messages(messages: list[dict[str, Any]]) -> str:
    return json.dumps(messages, ensure_ascii=False, indent=2)


def extract_tool_names(messages: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for message in messages:
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            function = call.get("function") if isinstance(call.get("function"), dict) else {}
            name = str(function.get("name") or call.get("name") or "").strip()
            if name and name not in names:
                names.append(name)
    return names


def events_appear_in_order(events: list[dict[str, Any]], rendered: str) -> bool:
    cursor = 0
    for event in events:
        text = str(event.get("text") or "").strip()
        if not text:
            continue
        found = rendered.find(text, cursor)
        if found < 0:
            return False
        cursor = found + len(text)
    return True


def redact_secrets(text: str) -> str:
    redacted = str(text or "")
    for pattern, replacement in SECRET_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def reduction(original: int, compacted: int) -> float:
    if original <= 0:
        return 0.0
    return max(0.0, (1.0 - compacted / original) * 100.0)


if __name__ == "__main__":
    raise SystemExit(main())
