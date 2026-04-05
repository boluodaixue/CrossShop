"""Template-based ShoppingSummary for the no-LLM deterministic chain."""

from collections.abc import Iterable

from globex_agent.domain import (
    PickedItem,
    ResultStatus,
    ShoppingSummaryOutput,
    ToolIssue,
)


def build_shopping_summary(
    picks: list[PickedItem],
    user_query: str,
    new_preferences: list[str] | None = None,
    *,
    issues: Iterable[ToolIssue] = (),
    rejected_brief: Iterable[str] = (),
) -> ShoppingSummaryOutput:
    """Return the chapter 14 output contract without invoking an LLM.

    The course implementation asks an LLM to write ``final_text``. Stage two is
    explicitly LLM-free, so only that prose-generation step is replaced by a
    stable template; ``final_text / picks / learned_preferences`` stay unchanged.
    """

    issue_list = list(issues)
    warnings = list(dict.fromkeys(issue.message for issue in issue_list))
    assumptions = [
        "商品、价格、评分和库存均为受控模拟数据",
        "运费和税费使用第 12 章的简化演示规则估算，不代表真实应缴金额",
        "到手价等于商品价、模拟运费与模拟税费之和",
    ]

    if picks:
        lines = [f"为“{user_query}”找到 {len(picks)} 个合规推荐：", ""]
        for index, pick in enumerate(picks, start=1):
            lines.extend(
                [
                    f"{index}. {pick.title}（{pick.platform.value}）",
                    f"   item_id：{pick.item_id}",
                    f"   到手价：{pick.currency.value} {pick.landed_cny}",
                    f"   理由：{'；'.join(pick.reasons)}",
                ]
            )
        lines.extend(["", "说明：以上价格、运费和税费均为离线演示估算。"])
        status = ResultStatus.OK
    else:
        lines = [f"未找到满足“{user_query}”全部硬约束的商品。"]
        rejected = list(rejected_brief)
        if rejected:
            lines.append("排除原因：")
            lines.extend(f"- {reason}" for reason in rejected[:8])
        if warnings:
            lines.append("提示：" + "；".join(warnings))
        invalid_input = any(
            issue.code in {"unsupported_destination", "unsupported_hard_constraint"}
            for issue in issue_list
        )
        status = ResultStatus.INVALID_INPUT if invalid_input else ResultStatus.NO_RESULTS

    return ShoppingSummaryOutput(
        final_text="\n".join(lines),
        picks=picks,
        learned_preferences=new_preferences or [],
        assumptions=assumptions,
        warnings=warnings,
        status=status,
    )
