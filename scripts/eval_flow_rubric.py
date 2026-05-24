"""LLM-generated per-case rubric scoring for agent-flow query conversations."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

from globex_agent.domain.catalog.exchange_rate import ExchangeRateTable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

RULES_TEXT = (
    "系统运费/关税规则：寄往中国，单件基础运费 25 元；"
    "关税按商品小计超出免税额度的部分计算，个人物品免税额度内关税为 0。"
)

RUBRIC_GENERATOR_SYSTEM = (
    "你是电商 Agent 评测用例设计者。给定一条用户 query、商品库事实表和系统规则，"
    "为该 query 生成独立的 P0/P1/P2 rubric。要求："
    "P0 是事实与安全底线，必须能依据商品库事实和系统规则逐条核验，不能只写通用套话；"
    "P1 是行为与命中，应具体到本 query 的品类、预算、属性或动作；"
    "P2 是表达要求，保持简洁可判。"
    "只输出 JSON："
    '{"p0": ["..."], "p1": ["..."], "p2": ["..."]}'
)

JUDGE_SYSTEM = (
    "你是严格的电商 Agent 评测员。给你一条买家 query 与 Agent 回复的对话记录、"
    "商品库事实表、系统规则，以及为该 case 单独生成的 P0/P1/P2 rubric。"
    "每条细则先推理再给出 pass。只输出 JSON："
    '{"p0": [{"criterion": "...", "reason": "...", "pass": true/false}],'
    ' "p1": [...], "p2": [...]}'
)


def render_rules() -> str:
    rates = ExchangeRateTable()
    rate_text = ", ".join(
        f"1 {currency} = {rate} CNY"
        for currency, rate in rates.rates_to_cny.items()
    )
    return f"{RULES_TEXT}\n系统汇率表：{rate_text}"


def _model() -> str:
    return (
        os.environ.get("EVAL_JUDGE_MODEL")
        or os.environ.get("LLM_JUDGE")
        or os.environ.get("LLM_MODEL")
        or "qwen-plus"
    )


def _base_url() -> str:
    return os.environ.get(
        "LLM_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


def _api_key() -> str:
    return os.environ.get("LLM_API_KEY") or ""


async def _chat_json(
    client: httpx.AsyncClient,
    system: str,
    user: str,
    *,
    temperature: float = 0,
) -> dict:
    payload = {
        "model": _model(),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {"type": "json_object"},
        "temperature": temperature,
    }
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            response = await client.post(
                f"{_base_url().rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {_api_key()}"},
                json=payload,
                timeout=180,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            return json.loads(content)
        except Exception as err:  # noqa: BLE001 - transient judge failures retried
            if attempt == 3:
                raise
            last_error = err
            print(f"   LLM 调用第 {attempt + 1} 次失败，稍后重试：{err}", flush=True)
            await asyncio.sleep(8 * (2**attempt))
    raise last_error if last_error else RuntimeError("LLM 重试耗尽")


def parse_conversation(path: Path) -> dict:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))

    turns = [row for row in rows if row.get("kind") == "turn"]
    transcript_lines = [
        f"[{row['role']}] {row['content']}"
        for row in turns
        if row.get("content")
    ]
    facts: dict[str, dict] = {}
    for row in rows:
        if row.get("kind") != "event" or row.get("type") != "tool.result":
            continue
        payload = row.get("payload") or {}
        if payload.get("tool") != "product_search_tool":
            continue
        for hit in payload.get("hits") or []:
            item_id = hit.get("item_id")
            if not item_id:
                continue
            facts[item_id] = {
                "item_id": item_id,
                "title": hit.get("title", ""),
                "category": hit.get("category", ""),
                "price": hit.get("price_major"),
                "currency": hit.get("currency", "CNY"),
            }

    query = next(
        (row["content"] for row in turns if row.get("role") == "buyer"),
        "",
    )
    return {
        "source": path.stem,
        "query": query,
        "transcript": "\n\n".join(transcript_lines),
        "facts": list(facts.values()),
    }


def render_facts(facts: list[dict]) -> str:
    if not facts:
        return "商品库事实表：无"
    lines = ["商品库事实表：", "| item_id | 标题 | 品类 | 价格 |", "|---|---|---|---|"]
    for item in facts:
        price = (
            f"{item['price']} {item['currency']}"
            if item["price"] is not None
            else "不可用"
        )
        lines.append(
            f"| {item['item_id']} | {item['title']} | {item['category']} | {price} |"
        )
    return "\n".join(lines)


async def generate_rubric(
    client: httpx.AsyncClient,
    item: dict,
) -> dict:
    user = (
        f"## Query\n{item['query']}\n\n"
        f"{render_facts(item['facts'])}\n\n"
        f"{render_rules()}"
    )
    return await _chat_json(
        client,
        RUBRIC_GENERATOR_SYSTEM,
        user,
        temperature=0.3,
    )


async def call_judge(
    client: httpx.AsyncClient,
    item: dict,
    rubric: dict,
) -> dict:
    user = (
        f"## 商品库事实表\n{render_facts(item['facts'])}\n\n"
        f"{render_rules()}\n\n"
        f"## 对话记录\n{item['transcript']}\n\n"
        f"## 评分细则\n{json.dumps(rubric, ensure_ascii=False, indent=2)}"
    )
    return await _chat_json(client, JUDGE_SYSTEM, user)


def score_case(judged: dict) -> tuple[float, bool]:
    def ratio(level: list) -> float:
        return (
            sum(1 for rule in level if rule.get("pass")) / len(level)
            if level
            else 1.0
        )

    p0 = ratio(judged.get("p0", []))
    p1 = ratio(judged.get("p1", []))
    p2 = ratio(judged.get("p2", []))
    weighted = 0.5 * p0 + 0.35 * p1 + 0.15 * p2
    return round(weighted, 3), p0 == 1.0


def render_report(results: list[dict]) -> str:
    pass_count = sum(1 for item in results if item["p0_pass"] and item["score"] >= 0.7)
    avg = round(sum(item["score"] for item in results) / len(results), 3) if results else 0
    lines = [
        f"# Flow query LLM-as-Judge 评测（{datetime.now().strftime('%Y-%m-%d %H:%M')}）",
        "",
        f"总览：{pass_count}/{len(results)} PASS，平均分 {avg}",
        "",
    ]
    for item in results:
        verdict = "PASS" if item["p0_pass"] and item["score"] >= 0.7 else "FAIL"
        lines.extend(
            [
                f"## {item['source']}",
                "",
                f"- Query：{item['query']}",
                f"- 结果：{verdict}（{item['score']}）",
                "- 生成 rubric：",
                "```json",
                json.dumps(item["rubric"], ensure_ascii=False, indent=2),
                "```",
                "- Judge 判定：",
            ]
        )
        for level in ("p0", "p1", "p2"):
            for rule in item["judged"].get(level, []):
                mark = "PASS" if rule.get("pass") else "FAIL"
                lines.append(
                    f"  - [{level.upper()}][{mark}] {rule.get('criterion', '')}"
                    f"：{rule.get('reason', '')}"
                )
        lines.append("")
    return "\n".join(lines)


def save_progress(results: list[dict], path: Path) -> None:
    path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conversations",
        default=str(PROJECT_ROOT / "data" / "conversations"),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--progress",
        default=str(PROJECT_ROOT / "eval" / "flow-rubric-progress.json"),
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    conv_dir = Path(args.conversations)
    paths = sorted(conv_dir.glob("flow-*.jsonl"))
    if args.limit:
        paths = paths[: args.limit]
    items = [parse_conversation(path) for path in paths]
    items = [item for item in items if item["query"]]
    if not items:
        raise SystemExit("没有可评测的 flow 会话")

    progress_path = Path(args.progress)
    done: dict[str, dict] = {}
    if args.resume and progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        done = {item["source"]: item for item in progress}

    print(f"== 共 {len(items)} 条 flow query", flush=True)
    async with httpx.AsyncClient() as client:
        for index, item in enumerate(items, 1):
            print(f"  [{index}/{len(items)}] {item['source']} ...", flush=True)
            if item["source"] in done:
                item.update(done[item["source"]])
                print(f"     -> 已跳过（{item.get('score')}）", flush=True)
            else:
                item["rubric"] = await generate_rubric(client, item)
                item["judged"] = await call_judge(client, item, item["rubric"])
                item["score"], item["p0_pass"] = score_case(item["judged"])
                print(
                    f"     -> {'PASS' if item['p0_pass'] and item['score'] >= 0.7 else 'FAIL'}"
                    f"（{item['score']}）",
                    flush=True,
                )
            save_progress(items[: index + 1], progress_path)

    report = render_report(items)
    report_path = (
        Path(args.out)
        if args.out
        else PROJECT_ROOT / "eval" / f"flow-rubric-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
    )
    report_path.write_text(report, encoding="utf-8")
    print(f"\n报告已写入：{report_path}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    asyncio.run(main())
