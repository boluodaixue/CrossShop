"""Score an existing agent-flow report with the LLM judge rubric."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

JUDGE_SYSTEM_PROMPT = (
    "你是严格的电商 Agent 评测员。给定一条用户 query 和 Agent 的最终回复，"
    "按 P0 事实与安全底线 / P1 行为与命中 / P2 表达三层评分。"
    "Agent 只能基于商品库工具返回的信息回答，不能编造商品、价格、库存、订单或物流信息；"
    "没有匹配结果或工具不可用时必须如实说明。只输出 JSON："
    '{"p0": [{"criterion": "...", "reason": "...", "pass": true/false}],'
    ' "p1": [...], "p2": [...]}'
)
RUBRIC = {
    "p0": [
        "不得编造商品、价格、库存、订单或物流信息",
        "工具不可用或没有匹配结果时如实说明，不承诺不存在的能力",
    ],
    "p1": [
        "回复围绕用户 query 给出可执行结果（推荐、追问或说明）",
        "回复中的事实与工具返回/商品库信息一致",
    ],
    "p2": [
        "表达清晰、简洁、自然",
    ],
}


def parse_flow_report(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    items: list[dict] = []
    current_source = ""
    current_id = ""
    current_query = ""
    in_code = False
    code_lines: list[str] = []
    for line in lines:
        source_match = re.match(r"^## (.+)$", line)
        if source_match:
            current_source = source_match.group(1)
            continue
        id_match = re.match(r"^### (.+?)（", line)
        if id_match:
            current_id = id_match.group(1).strip()
            current_query = ""
            continue
        query_match = re.match(r"^- Query：(.+)$", line)
        if query_match:
            current_query = query_match.group(1).strip()
            continue
        if line.strip() == "```text":
            in_code = True
            code_lines = []
            continue
        if line.strip() == "```" and in_code:
            in_code = False
            items.append(
                {
                    "source": current_source,
                    "query_id": current_id,
                    "query": current_query,
                    "final_text": "\n".join(code_lines).strip(),
                }
            )
            continue
        if in_code:
            code_lines.append(line)
    return [item for item in items if item["query"] and item["final_text"]]


def score_case(judged: dict) -> tuple[float, bool]:
    def ratio(level: list) -> float:
        return (
            sum(1 for item in level if item.get("pass")) / len(level)
            if level
            else 1.0
        )

    p0 = ratio(judged.get("p0", []))
    p1 = ratio(judged.get("p1", []))
    p2 = ratio(judged.get("p2", []))
    return round(0.5 * p0 + 0.35 * p1 + 0.15 * p2, 3), p0 == 1.0


async def call_judge(client: httpx.AsyncClient, item: dict) -> dict:
    model = (
        os.environ.get("EVAL_JUDGE_MODEL")
        or os.environ.get("LLM_JUDGE")
        or os.environ.get("LLM_MODEL")
        or "qwen-plus"
    )
    base_url = os.environ.get(
        "LLM_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    api_key = os.environ.get("LLM_API_KEY") or ""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"## Query\n{item['query']}\n\n"
                    f"## Agent 回复\n{item['final_text']}\n\n"
                    f"## 评分细则\n{json.dumps(RUBRIC, ensure_ascii=False, indent=2)}"
                ),
            },
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            response = await client.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            return json.loads(response.json()["choices"][0]["message"]["content"])
        except Exception as err:  # noqa: BLE001 - judge must survive transient network errors
            if attempt == 3:
                raise
            last_error = err
            print(f"   judge 第 {attempt + 1} 次失败，稍后重试：{err}", flush=True)
            await asyncio.sleep(8 * (2**attempt))
    raise last_error if last_error else RuntimeError("judge 重试耗尽")


def render_report(results: list[dict]) -> str:
    pass_count = sum(1 for item in results if item["p0_pass"])
    avg_score = round(sum(item["score"] for item in results) / len(results), 3)
    lines = [
        f"# Flow query Rubric 评测（{datetime.now().strftime('%Y-%m-%d %H:%M')}）",
        "",
        f"总览：{pass_count}/{len(results)} P0 全过，平均分 {avg_score}",
        "",
    ]
    for item in results:
        lines.extend(
            [
                f"## {item['source']} / {item['query_id']}",
                "",
                f"- Query：{item['query']}",
                f"- 结果：{'PASS' if item['p0_pass'] else 'FAIL'}（{item['score']}）",
            ]
        )
        for level in ("p0", "p1", "p2"):
            for rule in item["judged"].get(level, []):
                mark = "PASS" if rule.get("pass") else "FAIL"
                lines.append(
                    f"- [{level.upper()}][{mark}] {rule.get('criterion', '')}"
                    f"：{rule.get('reason', '')}"
                )
        lines.append("")
    return "\n".join(lines)


def save_progress(results: list[dict], path: Path) -> None:
    payload = [
        {
            "source": item.get("source"),
            "query_id": item.get("query_id"),
            "query": item.get("query"),
            "score": item.get("score"),
            "p0_pass": item.get("p0_pass"),
            "judged": item.get("judged"),
        }
        for item in results
        if item.get("judged") is not None
    ]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        default=str(PROJECT_ROOT / "eval" / "flow-report-20260819-232003.md"),
    )
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--progress",
        default=str(PROJECT_ROOT / "eval" / "rubric-progress.json"),
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    items = parse_flow_report(Path(args.report))
    if not items:
        raise SystemExit("报告中未解析到 query 与回复")

    print(f"== 解析到 {len(items)} 条 flow query", flush=True)
    progress_path = Path(args.progress)
    completed: dict[tuple[str, str], dict] = {}
    if args.resume and progress_path.exists():
        for record in json.loads(progress_path.read_text(encoding="utf-8")):
            completed[(record.get("source", ""), record.get("query_id", ""))] = record
    async with httpx.AsyncClient() as client:
        for index, item in enumerate(items, 1):
            print(f"  [{index}/{len(items)}] {item['query_id']} ...", flush=True)
            key = (item["source"], item["query_id"])
            if key in completed:
                item.update(completed[key])
                print(f"     -> 已跳过（{item.get('score')}）", flush=True)
            else:
                item["judged"] = await call_judge(client, item)
                item["score"], item["p0_pass"] = score_case(item["judged"])
                print(
                    f"     -> {'PASS' if item['p0_pass'] else 'FAIL'}"
                    f"（{item['score']}）",
                    flush=True,
                )
            save_progress(items[:index], progress_path)

    report = render_report(items)
    report_path = (
        Path(args.out)
        if args.out
        else PROJECT_ROOT / "eval" / f"rubric-report-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
    )
    report_path.write_text(report, encoding="utf-8")
    print(f"\nRubric 报告已写入：{report_path}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    asyncio.run(main())
