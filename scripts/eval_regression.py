"""LLM-judged regression for the 13 application-level end-to-end cases."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

import httpx
import yaml
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from globex_agent.catalog import LocalCatalog  # noqa: E402
from globex_agent.domain.catalog.exchange_rate import ExchangeRateTable  # noqa: E402
from globex_agent.infrastructure.transient import is_transient_error  # noqa: E402

BASE_URL = "http://127.0.0.1:8000"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

_JUDGE_MAX_RETRIES = 3
_JUDGE_RETRY_BASE_SECONDS = 8.0

JUDGE_SYSTEM_PROMPT = (
    "你是严格的电商 Agent 评测员。给你一段买家多轮提问与 Agent 回复的对话记录、"
    "商品库事实表，以及分级评分细则（P0 数字事实与安全底线 / P1 行为与命中 / P2 表达）。"
    "部分 case 会额外给出会话前置事实：这些事实真实有效，Agent 引用或应用它不算编造。"
    "每条细则先推理再给出 pass。只输出 JSON："
    '{"p0": [{"criterion": "...", "reason": "...", "pass": true/false}],'
    ' "p1": [...], "p2": [...]}'
)


def build_ground_truth() -> str:
    catalog = LocalCatalog.from_jsonl(
        PROJECT_ROOT / "data" / "demo" / "products.jsonl",
        strict=True,
    ).catalog
    lines = ["| item_id | 标题 | 品类 | 价格 |", "|---|---|---|---|"]
    for item in catalog.items:
        lines.append(
            f"| {item.item_id} | {item.title} | {' / '.join(item.category_path)} "
            f"| {item.price_cny} CNY |"
        )
    rates = ExchangeRateTable()
    rate_text = ", ".join(
        f"1 {currency} = {rate} CNY"
        for currency, rate in rates.rates_to_cny.items()
    )
    lines.append("")
    lines.append(f"系统汇率表：{rate_text}")
    return "\n".join(lines)


async def call_judge(
    client: httpx.AsyncClient,
    transcript: str,
    rubric: dict,
    ground_truth: str,
    prior_context: str = "",
) -> dict:
    model = (
        os.environ.get("EVAL_JUDGE_MODEL")
        or os.environ.get("LLM_JUDGE")
        or os.environ.get("LLM_MODEL")
        or os.environ.get("LLM_MAIN")
        or "qwen-plus"
    )
    base_url = (
        os.environ.get("LLM_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    prior_block = f"## 会话前置事实\n{prior_context}\n\n" if prior_context else ""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"## 商品库事实表\n{ground_truth}\n\n"
                    f"{prior_block}"
                    f"## 对话记录\n{transcript}\n\n"
                    f"## 评分细则\n{json.dumps(rubric, ensure_ascii=False, indent=2)}"
                ),
            },
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }
    last_error: Exception | None = None
    for attempt in range(_JUDGE_MAX_RETRIES):
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
            if not is_transient_error(err) or attempt == _JUDGE_MAX_RETRIES - 1:
                raise
            last_error = err
            delay = _JUDGE_RETRY_BASE_SECONDS * (2**attempt)
            print(f"   judge 遇限流，{delay:.0f}s 后重试：{err}", flush=True)
            await asyncio.sleep(delay)
    raise last_error if last_error else RuntimeError("judge 重试耗尽")


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


async def run_case(client: httpx.AsyncClient, case: dict, ground_truth: str) -> dict:
    session_id = f"eval-{case['id']}-{uuid.uuid4().hex[:6]}"
    buyer_id = case.get("buyer_id") or f"eval-buyer-{case['id']}"
    transcript_lines: list[str] = []
    for query in case["queries"]:
        response = await client.post(
            f"{BASE_URL}/commerce/intents",
            json={
                "shopping_session_id": session_id,
                "buyer_id": buyer_id,
                "locale": "zh-CN",
                "currency": "CNY",
                "raw_query": query,
            },
            timeout=600,
        )
        response.raise_for_status()
        final_text = response.json()["final_text"]
        transcript_lines.append(f"[买家] {query}\n[Agent] {final_text}")

    transcript = "\n\n".join(transcript_lines)
    judged = await call_judge(
        client,
        transcript,
        case["rubric"],
        ground_truth,
        case.get("prior_context", ""),
    )
    score, p0_all_pass = score_case(judged)
    return {
        "id": case["id"],
        "description": case["description"],
        "score": score,
        "p0_pass": p0_all_pass,
        "verdict": "PASS" if p0_all_pass and score >= 0.7 else "FAIL",
        "judged": judged,
        "transcript": transcript,
    }


def render_report(results: list[dict]) -> str:
    lines = [
        f"# Globex 评测回归报告（{datetime.now().strftime('%Y-%m-%d %H:%M')}）",
        "",
        f"总览：{sum(1 for r in results if r['verdict'] == 'PASS')}/{len(results)} PASS，"
        f"平均分 {sum(r['score'] for r in results) / len(results):.3f}",
        "",
        "| case | 描述 | 得分 | P0 | 结果 |",
        "|------|------|------|-----|------|",
    ]
    for result in results:
        lines.append(
            f"| {result['id']} | {result['description']} | {result['score']} | "
            f"{'通过' if result['p0_pass'] else '不通过'} | {result['verdict']} |"
        )
    lines.append("")
    for result in results:
        lines.append(f"## {result['id']}（{result['verdict']}，{result['score']}）")
        for level in ("p0", "p1", "p2"):
            for item in result["judged"].get(level, []):
                mark = "PASS" if item.get("pass") else "FAIL"
                lines.append(
                    f"- [{level.upper()}][{mark}] {item['criterion']}：{item.get('reason', '')}"
                )
        lines.append("")
        lines.append("<details><summary>对话记录</summary>\n")
        lines.append(result["transcript"])
        lines.append("\n</details>\n")
    return "\n".join(lines)


async def _guard_semantic_cache(allow: bool) -> None:
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            health = (await client.get(f"{BASE_URL}/health")).json()
    except Exception as err:  # noqa: BLE001
        print(f"警告：无法读取 /health（{err}），跳过缓存检查", flush=True)
        return
    if health.get("semantic_cache") and not allow:
        raise SystemExit(
            "拒绝跑回归：服务端语义缓存处于开启状态。"
            "请用 SEMANTIC_CACHE_ENABLED=0 重启服务，或加 --allow-semantic-cache。"
        )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        default=str(PROJECT_ROOT / "eval" / "cases.yaml"),
    )
    parser.add_argument("--only", default=None)
    parser.add_argument("--allow-semantic-cache", action="store_true")
    args = parser.parse_args()

    await _guard_semantic_cache(args.allow_semantic_cache)
    with open(args.cases, encoding="utf-8") as source:
        cases = yaml.safe_load(source)["cases"]
    if args.only:
        cases = [case for case in cases if case["id"] == args.only]

    results: list[dict] = []
    ground_truth = build_ground_truth()
    async with httpx.AsyncClient() as client:
        for case in cases:
            print(f"== 评测 {case['id']} ...", flush=True)
            try:
                result = await run_case(client, case, ground_truth)
            except Exception as err:  # noqa: BLE001
                result = {
                    "id": case["id"],
                    "description": case["description"],
                    "score": 0.0,
                    "p0_pass": False,
                    "verdict": "ERROR",
                    "judged": {},
                    "transcript": f"执行异常：{err}",
                }
            print(f"   -> {result['verdict']}（{result['score']}）", flush=True)
            results.append(result)

    report = render_report(results)
    report_path = PROJECT_ROOT / "eval" / f"report-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n报告已写入：{report_path}")


if __name__ == "__main__":
    asyncio.run(main())
