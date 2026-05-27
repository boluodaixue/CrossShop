"""Run CategoryInsight RAG queries through the agent flow, then rubric them."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAG_CASES = (
    PROJECT_ROOT
    / "data"
    / "category_insight"
    / "taobao_zh"
    / "category_recall_cases_taobao_zh.jsonl"
)


def select_rag_queries(path: Path, limit: int) -> list[dict]:
    selected: list[dict] = []
    seen_categories: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        category = record.get("category_slug") or record.get("category") or ""
        if category in seen_categories:
            continue
        seen_categories.add(category)
        selected.append(record)
        if len(selected) >= limit:
            break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=str(DEFAULT_RAG_CASES))
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()

    cases_path = Path(args.cases)
    if not cases_path.is_absolute():
        cases_path = PROJECT_ROOT / cases_path
    records = select_rag_queries(cases_path, args.limit)
    if not records:
        raise SystemExit("没有选出 CategoryInsight RAG query")

    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = output_dir / "flow-rag-selected.jsonl"
    selected_path.write_text(
        "\n".join(
            json.dumps(record, ensure_ascii=False)
            for record in records
        )
        + "\n",
        encoding="utf-8",
    )
    prefix = f"flowrag-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    python = sys.executable
    flow_cmd = [
        python,
        str(PROJECT_ROOT / "scripts" / "eval_flow_queries.py"),
        "--source",
        str(selected_path),
        "--prefix",
        prefix,
        "--limit",
        str(len(records)),
    ]
    rubric_cmd = [
        python,
        str(PROJECT_ROOT / "scripts" / "eval_flow_rubric.py"),
        "--prefix",
        prefix,
    ]

    print("== 阶段 1：CategoryInsight RAG query 全流程 ==", flush=True)
    flow_result = subprocess.run(flow_cmd, check=False)
    if flow_result.returncode != 0:
        raise SystemExit("flow query 阶段失败")

    print("\n== 阶段 2：per-case LLM rubric ==", flush=True)
    rubric_result = subprocess.run(rubric_cmd, check=False)
    if rubric_result.returncode != 0:
        raise SystemExit("rubric 阶段失败")

    print(f"\n完成。会话前缀：{prefix}")


if __name__ == "__main__":
    main()
