"""Run existing offline query sets through the full real-model agent flow."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

from globex_agent.infrastructure.transient import is_transient_error

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
load_dotenv(PROJECT_ROOT / ".env")

JUDGE_SYSTEM_PROMPT = (
    "你是严格的电商 Agent 评测员。给定一条用户 query 和 Agent 的最终回复，"
    "按 P0 事实与安全底线 / P1 行为与命中 / P2 表达三层评分。"
    "Agent 只能基于商品库工具返回的信息回答，不能编造商品、价格、库存、订单或物流信息；"
    "没有匹配结果或工具不可用时必须如实说明。只输出 JSON："
    '{"p0": [{"criterion": "...", "reason": "...", "pass": true/false}],'
    ' "p1": [...], "p2": [...]}'
)
DEFAULT_RUBRIC = {
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


def score_case(judged: dict) -> tuple[float, bool]:
    def ratio(items: list) -> float:
        return (
            sum(1 for item in items if item.get("pass")) / len(items)
            if items
            else 1.0
        )

    p0_ratio = ratio(judged.get("p0", []))
    p1_ratio = ratio(judged.get("p1", []))
    p2_ratio = ratio(judged.get("p2", []))
    weighted = 0.5 * p0_ratio + 0.35 * p1_ratio + 0.15 * p2_ratio
    return round(weighted, 3), p0_ratio == 1.0


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
                    f"## 评分细则\n{json.dumps(DEFAULT_RUBRIC, ensure_ascii=False, indent=2)}"
                ),
            },
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = await client.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            return json.loads(response.json()["choices"][0]["message"]["content"])
        except Exception as err:  # noqa: BLE001
            if not is_transient_error(err) or attempt == 2:
                raise
            last_error = err
            await asyncio.sleep(8 * (2**attempt))
    raise last_error if last_error else RuntimeError("judge 重试耗尽")


def _load_json_lines(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def _load_queries(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        rows = _load_json_lines(path)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else payload.get("cases", [])

    source_name = path.stem
    queries: list[dict] = []
    for record in rows:
        query = record.get("query") or record.get("raw_query") or ""
        if not query:
            continue
        query_id = (
            record.get("query_id")
            or record.get("id")
            or record.get("case_id")
            or f"{source_name}-{len(queries) + 1}"
        )
        split = record.get("split") or record.get("source_split") or ""
        language = record.get("language") or ""
        locale = record.get("locale") or ""
        if locale in {"cn", "zh"} or language == "zh":
            api_locale, currency = "zh-CN", "CNY"
        elif locale == "jp":
            api_locale, currency = "ja-JP", "JPY"
        else:
            api_locale, currency = "en-US", "USD"
        queries.append(
            {
                "source": source_name,
                "source_path": str(path),
                "query_id": str(query_id),
                "split": str(split),
                "language": language,
                "locale": locale,
                "api_locale": api_locale,
                "currency": currency,
                "query": query,
            }
        )
    return queries


async def run_one(client: httpx.AsyncClient, base_url: str, item: dict) -> dict:
    session_id = (
        f"flow-{item['source']}-{item['query_id']}-{uuid.uuid4().hex[:6]}"
    )
    started = time.monotonic()
    try:
        response = await client.post(
            f"{base_url}/commerce/intents",
            json={
                "shopping_session_id": session_id,
                "buyer_id": f"flow-buyer-{item['source']}",
                "locale": item["api_locale"],
                "currency": item["currency"],
                "raw_query": item["query"],
            },
            timeout=600,
        )
        response.raise_for_status()
        body = response.json()
        final_text = str(body.get("final_text", ""))
        error: str | None = None
        ok = not final_text.startswith("[error]")
    except Exception as err:  # noqa: BLE001 - report should never crash
        final_text = f"执行异常：{err}"
        error = str(err)
        ok = False
    elapsed_ms = round((time.monotonic() - started) * 1000)
    return {
        **item,
        "session_id": session_id,
        "final_text": final_text,
        "elapsed_ms": elapsed_ms,
        "ok": ok,
        "error": error,
    }


def render_report(results: list[dict]) -> str:
    ok_count = sum(1 for item in results if item["ok"])
    judged = [item for item in results if item.get("judged") is not None]
    pass_count = sum(1 for item in judged if item.get("p0_pass"))
    avg_ms = round(sum(item["elapsed_ms"] for item in results) / len(results)) if results else 0
    lines = [
        f"# Agent 全流程 query 测试（{datetime.now().strftime('%Y-%m-%d %H:%M')}）",
        "",
        f"总览：{ok_count}/{len(results)} 正常完成，平均耗时 {avg_ms} ms",
        "",
    ]
    if judged:
        avg_score = round(
            sum(item["score"] for item in judged) / len(judged),
            3,
        )
        lines.append(
            f"Rubric：{pass_count}/{len(judged)} P0 全过，平均分 {avg_score}"
        )
        lines.append("")
    current_source = None
    for item in results:
        if item["source"] != current_source:
            current_source = item["source"]
            lines.extend([f"## {current_source}", ""])
        lines.extend(
            [
                f"### {item['query_id']}（{item['split'] or 'unsplit'}）",
                "",
                f"- Query：{item['query']}",
                f"- Locale：{item['api_locale']} / {item['currency']}",
                f"- 耗时：{item['elapsed_ms']} ms",
                f"- 结果：{'OK' if item['ok'] else 'ERROR'}",
            ]
        )
        if item["error"]:
            lines.append(f"- 错误：{item['error']}")
        if item.get("judged") is not None:
            lines.append(
                f"- Rubric：{'PASS' if item['p0_pass'] else 'FAIL'}"
                f"（{item['score']}）"
            )
            for level in ("p0", "p1", "p2"):
                for rule in item["judged"].get(level, []):
                    mark = "PASS" if rule.get("pass") else "FAIL"
                    lines.append(
                        f"  - [{level.upper()}][{mark}] {rule.get('criterion', '')}"
                    )
        lines.extend(["", "```text", item["final_text"][:5000], "```", ""])
    return "\n".join(lines)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only-split", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--judge", action="store_true")
    args = parser.parse_args()

    all_items: list[dict] = []
    for raw_path in args.source:
        path = Path(raw_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        items = _load_queries(path)
        if args.only_split:
            items = [item for item in items if item["split"] == args.only_split]
        if args.limit:
            items = items[: args.limit]
        all_items.extend(items)

    if not all_items:
        raise SystemExit("没有可运行的 query")

    print(f"== 共 {len(all_items)} 条 query", flush=True)
    results: list[dict] = []
    async with httpx.AsyncClient() as client:
        for index, item in enumerate(all_items, 1):
            print(
                f"  [{index}/{len(all_items)}] "
                f"{item['source']} {item['query_id']} ...",
                flush=True,
            )
            result = await run_one(client, args.base_url, item)
            if args.judge:
                result["judged"] = await call_judge(client, result)
                result["score"], result["p0_pass"] = score_case(
                    result["judged"]
                )
            print(
                f"     -> {'OK' if result['ok'] else 'ERROR'}"
                f"（{result['elapsed_ms']} ms）"
                + (
                    f" rubric={result['score']}"
                    if result.get("judged") is not None
                    else ""
                ),
                flush=True,
            )
            results.append(result)

    report = render_report(results)
    report_path = (
        Path(args.out)
        if args.out
        else PROJECT_ROOT / "eval" / f"flow-report-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
    )
    report_path.write_text(report, encoding="utf-8")
    print(f"\n报告已写入：{report_path}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    asyncio.run(main())
