"""Run the fixed-order shopping pipeline without an LLM."""

import argparse
from pathlib import Path

from globex_agent.catalog import LocalCatalog
from globex_agent.domain import SearchRequest, UserProfile
from globex_agent.pipeline import run_deterministic_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "demo"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=int, default=1, help="1-based query case, default: 1")
    args = parser.parse_args()

    requests = _load_requests()
    if not 1 <= args.case <= len(requests):
        parser.error(f"--case must be between 1 and {len(requests)}")
    request = requests[args.case - 1]
    profile = _load_profiles().get(request.user_id) if request.user_id else None
    catalog = LocalCatalog.from_jsonl(DATA_DIR / "products.jsonl", strict=True).catalog

    result = run_deterministic_pipeline(catalog, request, profile)
    compared_offers = sum(
        len(comparison.offers) for comparison in result.price_comparison.comparisons
    )

    print(f"query: {request.query}")
    print(
        "ItemSearch: "
        f"{len(result.search.candidates)} returned / "
        f"{result.search.matched_catalog_items} matched"
    )
    print(
        "PriceCompare: "
        f"{len(result.price_comparison.comparisons)} products / "
        f"{compared_offers} offers"
    )
    print(f"ShippingCalc: {len(result.shipping.quotes)} landed-cost quotes")
    print(
        f"ItemPicker: {len(result.selection.picks)} picked / "
        f"{len(result.selection.rejected)} rejected"
    )
    print("ShoppingSummary:")
    print(result.summary.final_text)


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
