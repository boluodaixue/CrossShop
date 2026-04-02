from globex_agent.catalog import LocalCatalog
from globex_agent.domain import ResultStatus, SearchRequest, UserProfile
from globex_agent.tools.item_picker import pick_items
from globex_agent.tools.item_search import search_items
from globex_agent.tools.price_compare import compare_prices
from globex_agent.tools.shipping_calc import calculate_shipping


def _run_to_shipping(
    catalog: LocalCatalog,
    request: SearchRequest,
    profile: UserProfile | None = None,
):
    search = search_items(catalog, request, profile)
    shipping = calculate_shipping(compare_prices(search, base_currency=request.currency))
    return search, shipping


def test_picker_enforces_budget_and_material_constraints(
    demo_catalog: LocalCatalog,
    demo_requests: list[SearchRequest],
    demo_profiles: dict[str, UserProfile],
) -> None:
    request = demo_requests[1]
    search, shipping = _run_to_shipping(demo_catalog, request, demo_profiles["demo-user-002"])

    result = pick_items(search, shipping, demo_profiles["demo-user-002"])

    assert result.status is ResultStatus.OK
    assert [pick.canonical_product_id for pick in result.picks] == ["bp-city-roll"]
    assert result.picks[0].landed_price <= request.budget
    assert all(pick.canonical_product_id != "bp-minimal-18" for pick in result.picks)


def test_picker_never_returns_duplicate_canonical_products(
    demo_catalog: LocalCatalog,
    demo_requests: list[SearchRequest],
) -> None:
    search, shipping = _run_to_shipping(demo_catalog, demo_requests[0])

    result = pick_items(search, shipping)
    product_ids = [pick.canonical_product_id for pick in result.picks]

    assert len(product_ids) == len(set(product_ids))
    assert len(result.picks) <= 3


def test_unknown_hard_constraint_stops_selection(
    demo_catalog: LocalCatalog,
    demo_requests: list[SearchRequest],
) -> None:
    request = demo_requests[0].model_copy(
        update={"hard_constraints": {"unknown_constraint": "required"}}
    )
    search, shipping = _run_to_shipping(demo_catalog, request)

    result = pick_items(search, shipping)

    assert result.status is ResultStatus.INVALID_INPUT
    assert result.picks == []
    assert result.issues[0].code == "unsupported_hard_constraint"
