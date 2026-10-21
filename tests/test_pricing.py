from datetime import UTC, datetime

import httpx
import respx
from openai import AsyncOpenAI

from agentmon.llm import LLMRouter, ModelSpec, SlotsConfig
from agentmon.llm.router import PeakWindow, Price, _cost

PEAK = PeakWindow(hours_utc=[(1, 4), (6, 10)])
MONDAY_PEAK = datetime(2026, 10, 19, 7, 30, tzinfo=UTC)
MONDAY_OFF = datetime(2026, 10, 19, 4, 0, tzinfo=UTC)  # window end is exclusive
SATURDAY = datetime(2026, 10, 24, 7, 30, tzinfo=UTC)


def router(**providers) -> LLMRouter:
    config = SlotsConfig.model_validate(
        {
            "providers": providers,
            "slots": {},
            "pricing": {"ds": {"input": 0.30, "cached_input": 0.006, "output": 1.20, "off_peak": {"input": 0.15, "cached_input": 0.003, "output": 0.60}}},
        }
    )
    r = LLMRouter(config)
    for name, spec in config.providers.items():
        # A plain httpx client, so respx can intercept the SDK's requests.
        r._clients[name] = AsyncOpenAI(base_url=spec.base_url, api_key="k", http_client=httpx.AsyncClient(), max_retries=0)
    return r


def test_peak_window_hours_and_weekdays():
    assert PEAK.contains(MONDAY_PEAK)
    assert not PEAK.contains(MONDAY_OFF)
    assert not PEAK.contains(SATURDAY)


async def test_config_price_switches_to_off_peak_outside_window():
    r = router(p={"base_url": "http://llm.test/v1", "api_key_env": "KEY", "peak": PEAK.model_dump()})
    spec = ModelSpec(provider="p", model="ds", family="deepseek")
    assert (await r.price(spec, MONDAY_PEAK)).input == 0.30
    assert (await r.price(spec, SATURDAY)).input == 0.15
    assert await r.price(ModelSpec(provider="p", model="other", family="x"), SATURDAY) is None


@respx.mock
async def test_live_prices_from_verbose_model_listing():
    route = respx.get("http://llm.test/v1/models", params={"verbose": "true"}).mock(
        return_value=httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {"id": "judge", "object": "model", "pricing": {"prompt": "0.0000002", "completion": "0.0000006", "input_cache_read": None}},
                    {"id": "cached", "object": "model", "pricing": {"prompt": "0.000001", "completion": "0.000002", "input_cache_read": "0.0000001"}},
                    {"id": "free-form", "object": "model"},
                ],
            },
        )
    )
    r = router(p={"base_url": "http://llm.test/v1", "api_key_env": "KEY", "live_pricing": True})
    judge = await r.price(ModelSpec(provider="p", model="judge", family="qwen"), MONDAY_PEAK)
    assert judge == Price(input=0.2, cached_input=0.2, output=0.6)  # no cache-read price -> full input rate
    cached = await r.price(ModelSpec(provider="p", model="cached", family="x"), MONDAY_PEAK)
    assert cached.cached_input == 0.1
    assert await r.price(ModelSpec(provider="p", model="free-form", family="x"), MONDAY_PEAK) is None
    assert route.call_count == 1  # listing is fetched once per router


@respx.mock
async def test_live_pricing_failure_falls_back_to_config():
    respx.get("http://llm.test/v1/models").mock(return_value=httpx.Response(500))
    r = router(p={"base_url": "http://llm.test/v1", "api_key_env": "KEY", "live_pricing": True})
    assert (await r.price(ModelSpec(provider="p", model="ds", family="deepseek"), MONDAY_PEAK)).input == 0.30


def test_cost_splits_cached_and_uncached_prompt_tokens():
    price = Price(input=0.2, cached_input=0.02, output=0.6)
    assert _cost(price, prompt=1_000_000, completion=500_000, cached=900_000) == (100_000 * 0.2 + 900_000 * 0.02 + 500_000 * 0.6) / 1e6
    assert _cost(None, 1, 1, 0) is None
