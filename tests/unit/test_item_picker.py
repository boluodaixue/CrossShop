from globex_agent.catalog import LocalCatalog
from globex_agent.domain import ResultStatus, SearchRequest, UserProfile
from globex_agent.tools.item_picker import pick_items
from globex_agent.tools.item_search import search_items
from globex_agent.tools.price_compare import compare_prices
from globex_agent.tools.shipping_calc import calculate_shipping


def _run_to_shipping(
    catalog: LocalCatalog,
    request: SearchRequest,
):
    candidates = [
        candidate
        for platform in sorted(request.platforms, key=lambda value: value.value)
        for candidate in search_items(
            catalog,
            request.query,
            platform,
            request.top_k,
            request.hard_constraints,
        ).candidates
    ]
    return calculate_shipping(compare_prices(candidates).ranked)


def test_picker_enforces_budget_and_material_constraints(
    demo_catalog: LocalCatalog,
    demo_requests: list[SearchRequest],
    demo_profiles: dict[str, UserProfile],
) -> None:
    request = demo_requests[1]
    profile = demo_profiles["demo-user-002"]
    shipping = _run_to_shipping(demo_catalog, request)

    result = pick_items(shipping.items, request, profile)

    assert result.status is ResultStatus.OK
    assert [pick.same_group_id for pick in result.picks] == ["bp-city-roll"]
    assert result.picks[0].landed_cny <= request.budget
    assert all(pick.same_group_id != "bp-minimal-18" for pick in result.picks)


def test_picker_never_returns_duplicate_same_product_groups(
    demo_catalog: LocalCatalog,
    demo_requests: list[SearchRequest],
) -> None:
    request = demo_requests[0]
    shipping = _run_to_shipping(demo_catalog, request)

    result = pick_items(shipping.items, request)
    group_ids = [pick.same_group_id for pick in result.picks]

    assert len(group_ids) == len(set(group_ids))
    assert len(result.picks) <= 3


def test_unknown_hard_constraint_stops_selection(
    demo_catalog: LocalCatalog,
    demo_requests: list[SearchRequest],
) -> None:
    request = demo_requests[0].model_copy(
        update={"hard_constraints": {"unknown_constraint": "required"}}
    )
    shipping = _run_to_shipping(demo_catalog, request)

    result = pick_items(shipping.items, request)

    assert result.status is ResultStatus.INVALID_INPUT
    assert result.picks == []
    assert result.issues[0].code == "unsupported_hard_constraint"
