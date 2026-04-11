"""Run the stage-three single AgentLoop with a scripted or real model."""

import argparse
import asyncio
import sys
from pathlib import Path

from globex_agent.agent import ScriptedShoppingModel, run_agent
from globex_agent.agent.llm import get_llm
from globex_agent.catalog import LocalCatalog
from globex_agent.domain import SearchRequest, UserProfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "demo"


def main() -> None:
    _configure_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--case", type=int, help="1-based prepared demo query")
    source.add_argument("--query", help="free-form shopping query; requires --real")
    parser.add_argument("--user-id", help="optional demo profile ID used with --query")
    parser.add_argument(
        "--real",
        action="store_true",
        help="use the OPENAI-compatible model configured in .env",
    )
    args = parser.parse_args()

    if args.query and not args.real:
        parser.error("--query requires --real because free-form Planner parsing uses the LLM")
    if args.user_id and not args.query:
        parser.error("--user-id is only used together with --query")

    catalog = LocalCatalog.from_jsonl(DATA_DIR / "products.jsonl", strict=True).catalog
    profiles = _load_profiles()
    if args.user_id and args.user_id not in profiles:
        parser.error(f"unknown --user-id: {args.user_id}")

    if args.query:
        request = SearchRequest(
            query=args.query,
            user_id=args.user_id,
            platforms=set(catalog.platforms),
            top_k=3,
        )
        parse_request_with_llm = True
    else:
        requests = _load_requests()
        case_number = args.case or 1
        if not 1 <= case_number <= len(requests):
            parser.error(f"--case must be between 1 and {len(requests)}")
        request = requests[case_number - 1]
        parse_request_with_llm = False

    profile = profiles.get(request.user_id) if request.user_id else None
    model = get_llm() if args.real else ScriptedShoppingModel()
    mode = "real model" if args.real else "scripted model"

    result = asyncio.run(
        run_agent(
            request.query,
            catalog,
            request,
            profile,
            model=model,
            parse_request_with_llm=parse_request_with_llm,
        )
    )

    print(f"mode: {mode}")
    print(f"thread_id: {result.thread_id}")
    print("AgentLoop messages:")
    for message in result.messages:
        if message.tool_calls:
            print(f"[ai] tool_call: {', '.join(message.tool_calls)}")
        elif message.message_type == "tool":
            print(f"[tool] {message.tool_name}")
            if args.query and message.tool_name == "planner":
                print(f"      output: {message.content}")
        elif message.message_type == "human":
            print(f"[human] {message.content}")
        elif message.content:
            print("[ai] final answer")
    print(f"terminal_tool: {result.terminal_tool}")
    print("Final:")
    print(result.final_text)


def _configure_stdout() -> None:
    """Keep the active Windows encoding but do not crash on model emoji."""

    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(errors="replace")


def _load_requests() -> list[SearchRequest]:
    return [
        SearchRequest.model_validate_json(line)
        for line in (DATA_DIR / "queries.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_profiles() -> dict[str, UserProfile]:
    profiles = [
        UserProfile.model_validate_json(line)
        for line in (DATA_DIR / "users.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {profile.user_id: profile for profile in profiles}


if __name__ == "__main__":
    main()
