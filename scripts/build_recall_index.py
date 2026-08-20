#!/usr/bin/env python3
"""给存量语料建检索索引：工作区材料 + 历史会话。

日常维护不需要它——文件写入和轮末都会自动增量入索引。这个脚本只用于第一次建库，
或者索引被删掉之后重建（索引是投影，删了随时可以重建，这正是它该有的性质）。

    .venv/bin/python scripts/build_recall_index.py --data-root <账户数据目录>
    .venv/bin/python scripts/build_recall_index.py --files-only --limit 50
"""

from __future__ import annotations

from pathlib import Path
import argparse
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from work_agent_core.recall.sync import RecallSync  # noqa: E402
from work_agent_core.recall.tools import (  # noqa: E402
    backfill_vectors_once,
    embedding_backend,
    recall_index_for,
    recall_status,
)
from work_agent_core.session_log_store import SessionLogStore  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=".", help="账户数据目录，索引写在它下面的 recall/")
    parser.add_argument("--materials", default="meet_files", help="要索引的材料目录，相对 data-root")
    parser.add_argument("--workspace-root", default="", help="来源标识相对它计算；留空用 data-root")
    parser.add_argument("--session-db", default="", help="会话事件日志数据库路径；留空则跳过会话")
    parser.add_argument("--limit", type=int, default=0, help="最多处理多少个文件，0 表示不限")
    parser.add_argument("--files-only", action="store_true")
    parser.add_argument("--chats-only", action="store_true")
    parser.add_argument("--vectors", action="store_true", help="建完后顺带补算向量")
    parser.add_argument("--rebuild", action="store_true", help="先清空索引再建")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_root = Path(args.data_root).resolve()
    index = recall_index_for(data_root)
    sync = RecallSync(index, workspace_root=Path(args.workspace_root or data_root))
    started = time.monotonic()

    if args.rebuild:
        if index.database_path.exists():
            index.database_path.unlink()
        index = recall_index_for.__wrapped__(data_root) if hasattr(recall_index_for, "__wrapped__") else None
        print("已清空索引，请重新运行（不带 --rebuild）")
        return 0

    if not args.chats_only:
        materials = data_root / args.materials
        if materials.is_dir():
            print(f"扫描材料目录 {materials} …")
            report = sync.index_directory(materials, limit=args.limit)
            print(
                f"  来源 {report.sources}，新增节点 {report.added}，更新 {report.updated}，"
                f"未变 {report.unchanged}，跳过 {report.skipped}"
            )
            for failure in report.failures[:10]:
                print(f"  失败：{failure}")
            if len(report.failures) > 10:
                print(f"  （另有 {len(report.failures) - 10} 个失败未列出）")
        else:
            print(f"材料目录不存在，跳过：{materials}")

    if not args.files_only and args.session_db:
        store = SessionLogStore(args.session_db)
        conversations = store.list_sessions()
        print(f"索引会话 {len(conversations)} 个 …")
        added = 0
        for conversation_id in conversations:
            try:
                report = sync.index_conversation(conversation_id, store.load(conversation_id))
                added += report.added
            except Exception as error:
                print(f"  {conversation_id} 失败：{type(error).__name__}: {error}")
        print(f"  新增节点 {added}")

    if args.vectors:
        backend = embedding_backend()
        if backend is None:
            print("未配置 SILICONFLOW_API_KEY，跳过向量补算")
        else:
            total = 0
            while True:
                outcome = backfill_vectors_once(data_root, budget=64)
                if outcome.get("error"):
                    print(f"  向量补算中止：{outcome['error']}")
                    break
                total += int(outcome.get("written") or 0)
                if not outcome.get("more"):
                    break
            print(f"  写入向量 {total}")

    print(f"\n用时 {time.monotonic() - started:.1f}s")
    for key, value in recall_status(data_root).items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
