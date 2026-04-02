"""The first complete shopping flow, deliberately without an LLM."""

from globex_agent.catalog import LocalCatalog
from globex_agent.domain import (
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

    search = search_items(catalog, request, profile)
    price_comparison = compare_prices(search, base_currency=request.currency)
    shipping = calculate_shipping(price_comparison, destination=destination)
    selection = pick_items(search, shipping, profile, top_n=min(request.top_k, 3))
    summary = build_shopping_summary(
        request,
        selection,
        issues=[*search.issues, *price_comparison.issues, *shipping.issues],
    )
    return DeterministicPipelineResult(
        search=search,
        price_comparison=price_comparison,
        shipping=shipping,
        selection=selection,
        summary=summary,
    )
