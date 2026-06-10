"""Run existing offline query sets through the full real-model agent flow."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv
from websockets.asyncio.client import connect as websocket_connect

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
EVENT_CAPTURE_GRACE_SECONDS = 30.0
load_dotenv(PROJECT_ROOT / ".env")


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
        # Preserve every original regression contract field for offline
        # expected-vs-observed reporting.  Only the normalized runtime fields
        # below are used when making the REST request.
        normalized = dict(record)
        normalized.update(
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
        queries.append(normalized)
    return queries


async def run_one(
    client: httpx.AsyncClient,
    base_url: str,
    item: dict,
    prefix: str,
    request_timeout: float,
) -> dict:
    session_id = (
        f"{prefix}-{item['source']}-{item['query_id']}-{uuid.uuid4().hex[:6]}"
    )
    started = time.monotonic()
    capture_ready = asyncio.Event()
    capture_done = asyncio.Event()
    capture_errors: list[str] = []
    stop_capture = asyncio.Event()
    rest_done = asyncio.Event()
    ws_task = asyncio.create_task(
        _capture_events(
            base_url,
            session_id,
            capture_ready,
            capture_done,
            capture_errors,
            stop_capture,
            rest_done,
        )
    )
    final_text = ""
    recommended_cards: list[dict] = []
    verification_status = "unavailable"
    error: str | None = None
    event_capture_error: str | None = None
    rest_returned = False
    rest_status_code: int | None = None
    ready_wait = asyncio.create_task(capture_ready.wait())
    done_wait = asyncio.create_task(capture_done.wait())
    finished, pending = await asyncio.wait(
        {ready_wait, done_wait}, timeout=8, return_when=asyncio.FIRST_COMPLETED
    )
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    if not finished:
        event_capture_error = "TimeoutError: websocket subscription ready handshake timed out"
        error = event_capture_error
    elif not capture_ready.is_set():
        event_capture_error = capture_errors[0] if capture_errors else (
            "RuntimeError: websocket closed before subscription ready handshake"
        )
        error = event_capture_error

    # A REST request is only issued after the WS subscription has explicitly
    # acknowledged readiness.  This prevents the old race that produced empty
    # event arrays while the API was already completing the request.
    if event_capture_error is None:
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
                timeout=request_timeout,
            )
            rest_returned = True
            rest_status_code = response.status_code
            response.raise_for_status()
            body = response.json()
            final_text = str(body.get("text", body.get("final_text", "")))
            recommended_cards = body.get("recommended_cards", [])
            verification_status = body.get("verification_status", "unavailable")
            ok = not final_text.startswith("[error]")
            if verification_status == "unavailable":
                ok = False
                error = (
                    "API returned verification_status=unavailable; "
                    "conservative fallback preserved"
                )
        except Exception as err:  # noqa: BLE001 - report should never crash
            final_text = f"执行异常：{err}"
            recommended_cards = []
            verification_status = "unavailable"
            error = str(err)
            ok = False
    else:
        ok = False
    rest_done.set()
    elapsed_ms = round((time.monotonic() - started) * 1000)
    try:
        # Let the already-started WS subscriber receive the terminal event.
        # The REST response and final.result are produced by adjacent async
        # paths, so stopping immediately here would lose the last events.
        events, capture_error = await asyncio.wait_for(
            asyncio.shield(ws_task), timeout=EVENT_CAPTURE_GRACE_SECONDS
        )
    except TimeoutError:
        stop_capture.set()
        try:
            events, capture_error = await asyncio.wait_for(asyncio.shield(ws_task), timeout=2)
        except Exception:  # noqa: BLE001 - retain the original bounded timeout
            ws_task.cancel()
            await asyncio.gather(ws_task, return_exceptions=True)
            events = []
            capture_error = "TimeoutError: websocket event capture shutdown timed out"
    except Exception as err:  # noqa: BLE001 - preserve exact capture failure
        events = []
        capture_error = f"{type(err).__name__}: {err}"
    event_capture_error = event_capture_error or capture_error
    ws_final_seen = any(
        isinstance(event, dict) and event.get("type") == "final.result" for event in events
    )
    if event_capture_error:
        ok = False
        error = f"{error}; event_capture_error={event_capture_error}" if error else (
            f"event_capture_error={event_capture_error}"
        )
    if not events and not event_capture_error:
        event_capture_error = "RuntimeError: websocket capture returned no events"
        ok = False
        error = f"event_capture_error={event_capture_error}"
    if (
        events
        and not any(
            isinstance(event, dict) and event.get("type") == "final.result"
            for event in events
        )
        and not event_capture_error
    ):
        event_capture_error = "RuntimeError: websocket capture missed final.result"
        ok = False
        error = f"event_capture_error={event_capture_error}"
    final_result = {
        "text": final_text,
        "recommended_cards": recommended_cards,
        "verification_status": verification_status,
    }
    return {
        **item,
        "session_id": session_id,
        "final_text": final_text,
        "final_result": final_result,
        "recommended_cards": recommended_cards,
        "verification_status": verification_status,
        "events": events,
        "event_capture_error": event_capture_error,
        "elapsed_ms": elapsed_ms,
        "ok": ok,
        "error": error,
        "rest_returned": rest_returned,
        "rest_status_code": rest_status_code,
        "ws_final_seen": ws_final_seen,
    }


async def _capture_events(
    base_url: str,
    session_id: str,
    ready: asyncio.Event,
    done: asyncio.Event,
    errors: list[str],
    stop: asyncio.Event,
    rest_done: asyncio.Event | None = None,
) -> tuple[list[dict], str | None]:
    """Capture the exact WS timeline after an explicit subscription handshake."""

    ws_url = base_url.replace("https://", "wss://").replace("http://", "ws://")
    events: list[dict] = []
    error: str | None = None
    try:
        async with websocket_connect(
            f"{ws_url.rstrip('/')}/commerce/events",
            open_timeout=5,
            close_timeout=2,
            ping_interval=20,
        ) as connection:
            await connection.send(json.dumps({"shopping_session_id": session_id}))
            ready.set()
            saw_final = False
            terminal_deadline: float | None = None
            while not stop.is_set():
                recv_task = asyncio.create_task(connection.recv())
                stop_task = asyncio.create_task(stop.wait())
                rest_task = asyncio.create_task(rest_done.wait()) if rest_done is not None else None
                wait_tasks = {recv_task, stop_task}
                if rest_task is not None:
                    wait_tasks.add(rest_task)
                timeout = None
                if rest_done is None:
                    timeout = EVENT_CAPTURE_GRACE_SECONDS
                elif rest_done.is_set():
                    terminal_deadline = terminal_deadline or (
                        asyncio.get_running_loop().time() + EVENT_CAPTURE_GRACE_SECONDS
                    )
                    timeout = max(0.0, terminal_deadline - asyncio.get_running_loop().time())
                finished, pending = await asyncio.wait(
                    wait_tasks,
                    timeout=timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)
                if not finished:
                    error = (
                        "TimeoutError: websocket terminal grace expired"
                        if rest_done is not None
                        else "TimeoutError: websocket receive timed out"
                    )
                    break
                # If both complete together, prefer the received event: the
                # terminal events may already be buffered when REST returns.
                if recv_task in finished:
                    raw = recv_task.result()
                elif rest_task is not None and rest_task in finished:
                    terminal_deadline = (
                        asyncio.get_running_loop().time() + EVENT_CAPTURE_GRACE_SECONDS
                    )
                    continue
                elif stop_task in finished:
                    break
                else:  # pragma: no cover - asyncio.wait contract guard
                    break
                if not raw:
                    break
                event = json.loads(raw)
                if isinstance(event, dict):
                    events.append(event)
                    if event.get("type") == "final.result":
                        saw_final = True
                        break
            if not saw_final:
                error = error or "RuntimeError: websocket capture stopped before final.result"
    except Exception as err:  # noqa: BLE001 - exact error is part of the report
        error = f"{type(err).__name__}: {err}"
        errors.append(error)
    finally:
        done.set()
    return events, error


def render_report(results: list[dict]) -> str:
    ok_count = sum(1 for item in results if item["ok"])
    avg_ms = round(sum(item["elapsed_ms"] for item in results) / len(results)) if results else 0
    lines = [
        f"# Agent 全流程 query 测试（{datetime.now().strftime('%Y-%m-%d %H:%M')}）",
        "",
        f"总览：{ok_count}/{len(results)} 正常完成，平均耗时 {avg_ms} ms",
        "",
    ]
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
                f"- REST：{'returned' if item.get('rest_returned') else 'not_returned'}"
                f" / HTTP {item.get('rest_status_code') or 'N/A'}；"
                f"WS final.result：{'seen' if item.get('ws_final_seen') else 'missing'}",
            ]
        )
        if item["error"]:
            lines.append(f"- 错误：{item['error']}")
        if item.get("event_capture_error"):
            lines.append(f"- WS 事件捕获错误：{item['event_capture_error']}")
        event_types = [
            str(event.get("type"))
            for event in item.get("events", [])
            if isinstance(event, dict) and event.get("type")
        ]
        lines.append(f"- WS 事件类型：{', '.join(dict.fromkeys(event_types)) or '无'}")
        lines.extend(["", "```text", item["final_text"][:5000], "```", ""])
    return "\n".join(lines)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only-split", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--prefix", default="flow")
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=90.0,
        help="单条 REST Flow 的有界等待秒数",
    )
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
            result = await run_one(
                client,
                args.base_url,
                item,
                args.prefix,
                args.request_timeout,
            )
            print(
                f"     -> {'OK' if result['ok'] else 'ERROR'}"
                f"（{result['elapsed_ms']} ms）",
                flush=True,
            )
            results.append(result)

    report = render_report(results)
    report_path = (
        Path(args.out)
        if args.out
        else PROJECT_ROOT / "output" / "eval" /
        f"flow-report-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
    )
    report_path.write_text(report, encoding="utf-8")
    json_path = report_path.with_suffix(".json")
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已写入：{report_path}")
    print(f"逐项证据 JSON 已写入：{json_path}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    asyncio.run(main())
