"""The first complete shopping flow, deliberately without an LLM."""

from globex_agent.catalog import LocalCatalog
from globex_agent.domain import (
    Currency,
    DeterministicPipelineResult,
    SearchRequest,
    UserProfile,
)
from globex_agent.tools import (
    build_shopping_summary,
    calculate_shipping,
    compare_prices,
    pick_items,
    search_items,
)


def run_deterministic_pipeline(
    catalog: LocalCatalog,
    request: SearchRequest,
    profile: UserProfile | None = None,
    *,
    destination: str = "CN",
) -> DeterministicPipelineResult:
    """Run ItemSearch -> PriceCompare -> ShippingCalc -> ItemPicker -> summary."""

    search = [
        search_items(
            catalog,
            query=request.query,
            platform=platform,
            top_k=request.top_k,
            hard_constraints=request.hard_constraints,
        )
        for platform in sorted(request.platforms, key=lambda value: value.value)
    ]
    merged_candidates = [
        candidate for output in search for candidate in output.candidates
    ]
    price_comparison = compare_prices(
        merged_candidates,
        base_currency=Currency.CNY,
        top_n=12,
    )
    shipping = calculate_shipping(price_comparison.ranked, destination=destination)
    selection = pick_items(
        shipping.items,
        request,
        profile,
        top_n=min(request.top_k, 3),
    )
    summary = build_shopping_summary(
        selection.picks,
        request.query,
        issues=[
            *(issue for output in search for issue in output.issues),
            *price_comparison.issues,
            *shipping.issues,
            *selection.issues,
        ],
        rejected_brief=selection.rejected_brief,
    )
    return DeterministicPipelineResult(
        search=search,
        price_comparison=price_comparison,
        shipping=shipping,
        selection=selection,
        summary=summary,
    )
