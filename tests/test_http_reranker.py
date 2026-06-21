from __future__ import annotations

import math

import pytest

from app.infrastructure.rerank.http_reranker import HttpReranker


def test_parse_scores_restores_original_order_and_accepts_score_alias() -> None:
    body = {
        "results": [
            {"index": 2, "relevance_score": "2.5"},
            {"index": 0, "score": -1},
            {"index": 1, "relevance_score": 0.75},
        ]
    }

    assert HttpReranker._parse_scores(body, expected_count=3) == [-1.0, 0.75, 2.5]


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        ([], "body 不是对象"),
        ({}, "results 数量错误"),
        ({"results": {}}, "results 数量错误"),
        ({"results": []}, "results 数量错误"),
        ({"results": ["bad"]}, "result 不是对象"),
        ({"results": [{"index": True, "relevance_score": 1}]}, "index 非法"),
        ({"results": [{"index": 0.0, "relevance_score": 1}]}, "index 非法"),
        ({"results": [{"index": -1, "relevance_score": 1}]}, "index 非法"),
        ({"results": [{"index": 1, "relevance_score": 1}]}, "index 非法"),
        (
            {
                "results": [
                    {"index": 0, "relevance_score": 1},
                    {"index": 0, "relevance_score": 2},
                ]
            },
            "index 重复",
        ),
        ({"results": [{"relevance_score": 1}]}, "index 非法"),
        ({"results": [{"index": 0}]}, "缺少分数"),
        ({"results": [{"index": 0, "relevance_score": None}]}, "分数不是数值"),
        (
            {"results": [{"index": 0, "relevance_score": "not-a-number"}]},
            "分数不是数值",
        ),
        ({"results": [{"index": 0, "relevance_score": math.nan}]}, "分数非有限值"),
        ({"results": [{"index": 0, "relevance_score": math.inf}]}, "分数非有限值"),
        ({"results": [{"index": 0, "relevance_score": -math.inf}]}, "分数非有限值"),
        ({"results": [{"index": 0, "relevance_score": "NaN"}]}, "分数非有限值"),
    ],
)
def test_parse_scores_rejects_malformed_response(body: object, reason: str) -> None:
    expected_count = 2 if reason == "index 重复" else 1
    with pytest.raises(RuntimeError, match=reason):
        HttpReranker._parse_scores(body, expected_count=expected_count)
