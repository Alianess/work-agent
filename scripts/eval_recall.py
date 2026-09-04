"""recall 层检索评测：hit@k / MRR，BM25 与 hybrid 对比。

用法：
    .venv/bin/python scripts/eval_recall.py                 # 两种模式都跑
    .venv/bin/python scripts/eval_recall.py --mode bm25     # 只跑词法
    .venv/bin/python scripts/eval_recall.py --verbose       # 逐例明细

判定两级：
- 内容级（text_hash）：期望段落正文确实出现在 top-k（同内容 md/docx 渲染算同一答案）
- 来源级（source_id）：期望文件/会话出现在 top-k（更宽松，段落切分差异也容忍）

hybrid 模式需要 SILICONFLOW_API_KEY（从 .env 自动加载）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from work_agent_core.config import load_env_file  # noqa: E402
from work_agent_core.recall.index import RecallIndex  # noqa: E402
from work_agent_core.recall.search import RecallDeps, search  # noqa: E402
from work_agent_core.recall.tools import embedding_backend, rerank_backend  # noqa: E402

CASES_PATH = Path(__file__).resolve().parent / "eval_recall_cases.json"


def expand_expected(index: RecallIndex, cases: list[dict]) -> None:
    """用信息指纹反查等价答案，补全期望命中集。

    同一信息散布在纪要、ASR 逐字稿、聊天里，正文各不相同、hash 各不相同。
    手写 hash 只能列出已知副本；用独特子串反查能把"含这条信息的所有段落"
    都算作正确答案——评测度量的才是"信息能否被找到"，而不是"切分是否一致"。
    """

    with index._connect() as connection:  # noqa: SLF001 - 评测脚本读取同包内部
        for case in cases:
            hashes = set(case["expected"]["text_hashes"])
            sources = set(case["expected"]["source_ids"])
            for pattern in case.get("evidence_patterns") or []:
                rows = connection.execute(
                    "SELECT text_hash, source_id FROM recall_nodes"
                    " WHERE is_leaf=1 AND text LIKE ? AND text_hash IS NOT NULL",
                    (pattern,),
                ).fetchall()
                for row in rows:
                    if row["text_hash"]:
                        hashes.add(str(row["text_hash"]))
                        sources.add(str(row["source_id"]))
            case["expected"]["text_hashes"] = sorted(hashes)
            case["expected"]["source_ids"] = sorted(sources)


def run_mode(index: RecallIndex, cases: list[dict], *, hybrid: bool, top_k: int) -> dict:
    embedding = None
    rerank = None
    if hybrid:
        embedding = embedding_backend()
        rerank = rerank_backend()
        if embedding is None:
            print("  [warn] 未配置 SILICONFLOW_API_KEY，hybrid 退化为纯词法")
    deps = RecallDeps(index=index, embedding=embedding, rerank=rerank)

    details: list[dict] = []
    text_hits = {1: 0, 3: 0, 6: 0}
    source_hits = 0
    mrr_total = 0.0
    latency_total = 0.0

    for case in cases:
        patterns = [p for p in case.get("evidence_patterns") or [] if p]
        expected_sources = set(case["expected"]["source_ids"])
        started = time.time()
        outcome = search(deps, case["query"], top_k=top_k)
        latency = (time.time() - started) * 1000
        latency_total += latency

        # 信息级判定：返回段落的正文里出现信息指纹即算命中。
        # hash 级在这里是错误度量——同一信息散布在多副本、切分边界浮动的
        # 多个 passage 里，hash 穷尽不了，而"文本里有没有这条信息"才是
        # 用户真正感知的东西。
        rank_info = None
        rank_source = None
        for position, item in enumerate(outcome.get("results") or [], start=1):
            texts = [str(item.get("text") or "")]
            node_sources = {str(item.get("source_id") or "")}
            for extra in item.get("also_in") or []:
                node = index.node(str(extra.get("id") or ""))
                if node is not None:
                    texts.append(str(node["text"] or ""))
                    node_sources.add(str(node["source_id"]))
            if rank_info is None and any(p in text for p in patterns for text in texts):
                rank_info = position
            if rank_source is None and node_sources & expected_sources:
                rank_source = position

        for k in text_hits:
            if rank_info is not None and rank_info <= k:
                text_hits[k] += 1
        if rank_source is not None and rank_source <= top_k:
            source_hits += 1
        if rank_info is not None:
            mrr_total += 1.0 / rank_info

        details.append(
            {
                "id": case["id"],
                "query": case["query"],
                "rank_info": rank_info,
                "rank_source": rank_source,
                "latency_ms": round(latency, 1),
                "top1_path": " / ".join((outcome.get("results") or [{}])[0].get("path") or [])[:60],
                "degraded": bool((outcome.get("retrieval") or {}).get("degraded")),
            }
        )

    count = len(cases)
    return {
        "details": details,
        "count": count,
        "hit@1": text_hits[1] / count,
        "hit@3": text_hits[3] / count,
        "hit@6": text_hits[6] / count,
        "source_hit@topk": source_hits / count,
        "mrr": mrr_total / count,
        "avg_latency_ms": latency_total / count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", default="u1", help="账户目录名（默认 u1）")
    parser.add_argument("--mode", choices=["bm25", "hybrid", "both"], default="both")
    parser.add_argument("--cases", default=str(CASES_PATH))
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    load_env_file(ROOT / ".env")

    db_path = ROOT / "meet_files" / "users" / args.user / "recall" / "recall.sqlite3"
    if not db_path.exists():
        print(f"找不到索引：{db_path}")
        return 1
    index = RecallIndex(db_path)

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))["cases"]
    expand_expected(index, cases)
    print(f"评测集：{len(cases)} 例 | 索引：{db_path.relative_to(ROOT)}\n")

    modes: list[tuple[str, bool]] = []
    if args.mode in {"bm25", "both"}:
        modes.append(("bm25", False))
    if args.mode in {"hybrid", "both"}:
        modes.append(("hybrid", True))

    summary: dict[str, dict] = {}
    for name, hybrid in modes:
        result = run_mode(index, cases, hybrid=hybrid, top_k=args.top_k)
        summary[name] = result
        print(f"== {name} ==")
        print(
            f"  信息级 hit@1={result['hit@1']:.2%}  hit@3={result['hit@3']:.2%}  "
            f"hit@6={result['hit@6']:.2%}  MRR={result['mrr']:.3f}"
        )
        print(f"  来源级 hit@{args.top_k}={result['source_hit@topk']:.2%}  平均延迟={result['avg_latency_ms']:.0f}ms")
        if args.verbose:
            for detail in result["details"]:
                mark = "√" if detail["rank_info"] is not None else "×"
                rank = detail["rank_info"] or detail["rank_source"] or "-"
                print(
                    f"  {mark} [{rank}] {detail['id']}  ({detail['latency_ms']:.0f}ms)  {detail['top1_path']}"
                )
        print()

    if "bm25" in summary and "hybrid" in summary:
        delta = summary["hybrid"]["mrr"] - summary["bm25"]["mrr"]
        print(f"hybrid 相对 bm25 的 MRR 变化：{delta:+.3f}")

    out_path = ROOT / "scripts" / "eval_recall_baseline.json"
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cases": len(cases),
        "summary": {name: {k: v for k, v in r.items() if k != "details"} for name, r in summary.items()},
        "details": {name: r["details"] for name, r in summary.items()},
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"基线已写入：{out_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
