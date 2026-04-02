"""Stable template-based terminal summary for deterministic recommendations."""

from collections.abc import Iterable

from globex_agent.domain import (
    ItemPickerResult,
    ResultStatus,
    SearchRequest,
    ShoppingRecommendation,
    ShoppingSummaryResult,
    ToolIssue,
)


def build_shopping_summary(
    request: SearchRequest,
    selection: ItemPickerResult,
    issues: Iterable[ToolIssue] = (),
) -> ShoppingSummaryResult:
    """Build a deterministic human-readable summary without model generation."""

    all_issues = [*issues, *selection.issues]
    warnings = list(dict.fromkeys(issue.message for issue in all_issues))
    assumptions = [
        "商品、价格、评分、库存和运费均为受控模拟数据",
        "税费使用 demo-cn-landed-cost-v1 演示规则估算，不代表真实应缴金额",
        "到手价等于换算后商品价、模拟运费与模拟税费之和",
    ]
    recommendation = ShoppingRecommendation(
        request=request,
        items=selection.picks,
        assumptions=assumptions,
        warnings=warnings,
    )

    if selection.picks:
        lines = [f"为“{request.query}”找到 {len(selection.picks)} 个合规推荐：", ""]
        for index, pick in enumerate(selection.picks, start=1):
            lines.extend(
                [
                    f"{index}. {pick.title}（{pick.platform.value}）",
                    f"   到手价：{pick.currency.value} {pick.landed_price}",
                    f"   理由：{'；'.join(pick.reasons)}",
                ]
            )
        lines.extend(["", "说明：以上价格和税费均为离线演示估算。"])
        status = ResultStatus.OK
    else:
        lines = [f"未找到满足“{request.query}”全部硬约束的商品。"]
        if selection.rejected:
            lines.append("排除原因：")
            lines.extend(
                f"- {rejected.canonical_product_id}：{rejected.reason}"
                for rejected in selection.rejected[:8]
            )
        if warnings:
            lines.append("提示：" + "；".join(warnings))
        invalid_input = selection.status is ResultStatus.INVALID_INPUT or any(
            issue.code in {"unsupported_destination", "unsupported_hard_constraint"}
            for issue in all_issues
        )
        status = ResultStatus.INVALID_INPUT if invalid_input else ResultStatus.NO_RESULTS
    return ShoppingSummaryResult(
        recommendation=recommendation,
        final_text="\n".join(lines),
        status=status,
    )
