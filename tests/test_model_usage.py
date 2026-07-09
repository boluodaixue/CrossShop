from types import SimpleNamespace

from agentscope.model import ChatUsage

from app.infrastructure.model_usage import capture_model_usage, record_model_usage


def test_capture_normalizes_agentscope_cache_usage() -> None:
    response = SimpleNamespace(
        usage=ChatUsage(
            input_tokens=1_200,
            output_tokens=80,
            time=1.25,
            cache_input_tokens=960,
        ),
    )

    with capture_model_usage() as capture:
        record_model_usage(response, requested_model="deepseek-v4-flash")

    assert capture.totals() == {
        "model_calls": 1,
        "input_tokens": 1_200,
        "output_tokens": 80,
        "total_tokens": 1_280,
        "cached_input_tokens": 960,
        "cache_hit_calls": 1,
        "token_weighted_cache_hit_ratio": 0.8,
    }
    assert capture.samples[0].requested_model == "deepseek-v4-flash"
    assert capture.samples[0].cache_hit is True


def test_capture_accepts_raw_ark_prompt_token_details() -> None:
    response = {
        "usage": {
            "prompt_tokens": 2_000,
            "completion_tokens": 100,
            "total_tokens": 2_100,
            "prompt_tokens_details": {"cached_tokens": 1_600},
        },
    }

    with capture_model_usage() as capture:
        record_model_usage(response, requested_model="deepseek-v4-flash")

    assert capture.samples[0].input_tokens == 2_000
    assert capture.samples[0].cached_input_tokens == 1_600
    assert capture.samples[0].total_tokens == 2_100


def test_capture_is_opt_in_and_task_scoped() -> None:
    response = SimpleNamespace(
        usage=ChatUsage(input_tokens=10, output_tokens=2, time=0.1),
    )

    record_model_usage(response, requested_model="outside")
    with capture_model_usage() as outer:
        record_model_usage(response, requested_model="outer")
        with capture_model_usage() as inner:
            record_model_usage(response, requested_model="inner")
        record_model_usage(response, requested_model="outer")

    assert [item.requested_model for item in outer.samples] == ["outer", "outer"]
    assert [item.requested_model for item in inner.samples] == ["inner"]
