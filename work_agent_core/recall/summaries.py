"""给章节生成一句话摘要：只看这一句，就能判断要不要打开它。

展开地图告诉模型打开一节要花多少 token，但没告诉它里面是什么。只有价格没有
货物，选择就还是盲的。摘要补的正是这一半。

V5_dev 在这件事上踩过一个坑值得记：他们最初用正文前 120 字截断当摘要，结果
约 11% 的摘要以警告或套话开头、信息量为零，模型在目录里照样选不准。所以摘要
必须是模型读完整节之后写的一句话，不能是截断。

离线批处理、可断点续跑：只给没有摘要且确实值得摘要的章节生成，失败跳过下次再来。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable
import sqlite3

from .index import RecallIndex


SUMMARY_MIN_TOKENS = 600
"""比这更短的章节不值得摘要——直接读正文比读摘要还便宜。"""

SUMMARY_MAX_INPUT_CHARS = 6000
SUMMARY_BATCH = 20

SUMMARY_PROMPT = (
    "用一句话概括下面这一节讲了什么，让人只看这一句就能判断要不要打开它。\n"
    "要求：不超过 60 字；写清楚这一节的具体内容（涉及哪些事、哪些数字、哪些主体），"
    "不要写“本节介绍了……”这类空话，不要以警告或套话开头，不要复述标题。\n"
    "只输出这一句，不要任何前缀。\n\n"
    "标题：{title}\n\n正文：\n{body}"
)


@dataclass
class SummaryReport:
    written: int = 0
    skipped: int = 0
    failed: int = 0
    remaining: int = 0


def ensure_summary_column(index: RecallIndex) -> None:
    """建表与老库补列都在 RecallIndex 里做了，这里只保留调用点的可读性。"""

    return None


def sections_needing_summary(
    index: RecallIndex, *, limit: int = SUMMARY_BATCH, min_tokens: int = SUMMARY_MIN_TOKENS
) -> list[dict[str, Any]]:
    ensure_summary_column(index)
    with index._connect() as connection:  # noqa: SLF001
        rows = connection.execute(
            """
            SELECT id, title, path, text, tokens FROM recall_nodes
            WHERE is_leaf = 0 AND summary = '' AND title != '' AND tokens >= ?
            ORDER BY summary_wanted DESC, tokens DESC
            LIMIT ?
            """,
            (int(min_tokens), int(limit)),
        ).fetchall()
    return [dict(row) for row in rows]


def summary_debt(index: RecallIndex, *, min_tokens: int = SUMMARY_MIN_TOKENS) -> int:
    ensure_summary_column(index)
    with index._connect() as connection:  # noqa: SLF001
        return int(
            connection.execute(
                "SELECT COUNT(*) AS c FROM recall_nodes"
                " WHERE is_leaf = 0 AND summary = '' AND title != ''"
                " AND summary_wanted > 0 AND tokens >= ?",
                (int(min_tokens),),
            ).fetchone()["c"]
        )


def store_summaries(index: RecallIndex, pairs: Iterable[tuple[str, str]]) -> int:
    ensure_summary_column(index)
    written = 0
    with index._connect() as connection:  # noqa: SLF001
        for node_id_value, summary in pairs:
            text = " ".join(str(summary or "").split())[:120]
            if not text:
                continue
            connection.execute(
                "UPDATE recall_nodes SET summary = ? WHERE id = ?", (text, node_id_value)
            )
            written += 1
    return written


def generate_summaries(
    index: RecallIndex,
    client: Any,
    profile: Any,
    *,
    batch: int = SUMMARY_BATCH,
    min_tokens: int = SUMMARY_MIN_TOKENS,
) -> SummaryReport:
    """跑一批。断点续跑：已有摘要的章节不会重复生成。"""

    report = SummaryReport()
    pending = sections_needing_summary(index, limit=batch, min_tokens=min_tokens)
    if not pending:
        report.remaining = 0
        return report

    written: list[tuple[str, str]] = []
    for row in pending:
        body = str(row["text"] or "")[:SUMMARY_MAX_INPUT_CHARS]
        if not body.strip():
            report.skipped += 1
            continue
        prompt = SUMMARY_PROMPT.format(title=str(row["title"] or ""), body=body)
        try:
            response = client.chat(
                [{"role": "user", "content": prompt}],
                profile=profile,
                temperature=0.2,
                max_tokens=200,
                reasoning_effort="light",
            )
        except Exception:
            report.failed += 1
            continue
        line = " ".join(str(response.content or "").split())
        if line:
            written.append((str(row["id"]), line))
        else:
            report.failed += 1

    report.written = store_summaries(index, written)
    report.remaining = summary_debt(index, min_tokens=min_tokens)
    return report
