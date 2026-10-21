"""One provider interface for every OpenAI-compatible LLM backend, addressed by named slots."""

import asyncio
import logging
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import yaml
from dotenv import dotenv_values
from openai import AsyncOpenAI
from pydantic import BaseModel

from agentmon.settings import ROOT

log = logging.getLogger(__name__)


class PeakWindow(BaseModel):
    """Time-of-use pricing: a model's normal prices apply inside the window, its `off_peak` prices outside."""

    hours_utc: list[tuple[int, int]]  # [start, end) UTC hours
    weekdays: list[int] = [0, 1, 2, 3, 4]  # Monday = 0

    def contains(self, at: datetime) -> bool:
        at = at.astimezone(UTC)
        return at.weekday() in self.weekdays and any(start <= at.hour < end for start, end in self.hours_utc)


class ProviderSpec(BaseModel):
    base_url: str
    api_key_env: str
    live_pricing: bool = False  # read per-model prices from the provider's GET /models?verbose=true
    peak: PeakWindow | None = None


class ModelSpec(BaseModel):
    provider: str
    model: str
    family: str


class Price(BaseModel):
    """USD per 1M tokens."""

    input: float
    cached_input: float
    output: float
    off_peak: "Price | None" = None


class SlotsConfig(BaseModel):
    providers: dict[str, ProviderSpec]
    slots: dict[str, list[ModelSpec]]
    pricing: dict[str, Price] = {}

    def slot(self, name: str) -> list[ModelSpec]:
        if name not in self.slots:
            raise KeyError(f"slot '{name}' is not configured")
        return self.slots[name]


def load_slots(path: Path) -> SlotsConfig:
    return SlotsConfig.model_validate(yaml.safe_load(path.read_text()))


class LLMResult(BaseModel):
    text: str
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    cost_usd: float | None
    latency_ms: int


UsageHook = Callable[[str, LLMResult], None]


class LLMRouter:
    def __init__(self, config: SlotsConfig, on_usage: UsageHook | None = None):
        self.config = config
        self.on_usage = on_usage
        self._clients: dict[str, AsyncOpenAI] = {}
        self._env = {**dotenv_values(ROOT / ".env"), **os.environ}
        self._live_prices: dict[str, dict[str, Price]] = {}
        self._pricing_lock = asyncio.Lock()

    def client(self, provider: str) -> AsyncOpenAI:
        if provider not in self._clients:
            spec = self.config.providers[provider]
            key = self._env.get(spec.api_key_env)
            if not key:
                raise RuntimeError(f"{spec.api_key_env} is not set")
            self._clients[provider] = AsyncOpenAI(base_url=spec.base_url, api_key=key, max_retries=2, timeout=120)
        return self._clients[provider]

    async def list_models(self, provider: str) -> list[str]:
        page = await self.client(provider).models.list()
        return sorted(m.id for m in page.data)

    async def price(self, spec: ModelSpec, at: datetime) -> Price | None:
        """Price in effect for a call made at `at`: the provider's live listing first, then the config's `pricing`."""
        provider = self.config.providers[spec.provider]
        price = None
        if provider.live_pricing:
            price = (await self._live(spec.provider)).get(spec.model)
        price = price or self.config.pricing.get(spec.model)
        if price and price.off_peak and provider.peak and not provider.peak.contains(at):
            return price.off_peak
        return price

    async def _live(self, provider: str) -> dict[str, Price]:
        async with self._pricing_lock:  # one listing per router, not one per concurrent call
            if provider not in self._live_prices:
                try:
                    self._live_prices[provider] = await self._fetch_live_prices(provider)
                except Exception as e:
                    log.warning("live pricing for %s unavailable (%s); falling back to config pricing", provider, e)
                    self._live_prices[provider] = {}
        return self._live_prices[provider]

    async def _fetch_live_prices(self, provider: str) -> dict[str, Price]:
        page = await self.client(provider).models.list(extra_query={"verbose": "true"})
        prices: dict[str, Price] = {}
        for m in page.data:
            p = (m.model_extra or {}).get("pricing") or {}
            try:
                per_m = {k: round(float(p[k]) * 1_000_000, 6) for k in ("prompt", "completion")}
            except (KeyError, TypeError, ValueError):
                continue
            cache_read = p.get("input_cache_read")
            # No cache-read price listed: cached prompt tokens are billed at the normal input rate.
            cached = round(float(cache_read) * 1_000_000, 6) if cache_read is not None else per_m["prompt"]
            prices[m.id] = Price(input=per_m["prompt"], cached_input=cached, output=per_m["completion"])
        return prices

    async def chat(
        self,
        spec: ModelSpec,
        messages: list[dict],
        *,
        slot: str,
        temperature: float = 0.0,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> LLMResult:
        kwargs: dict = {"model": spec.model, "messages": messages, "temperature": temperature}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        called_at = datetime.now(UTC)
        started = time.perf_counter()
        resp = await self.client(spec.provider).chat.completions.create(**kwargs)
        latency_ms = int((time.perf_counter() - started) * 1000)

        usage = resp.usage
        prompt = usage.prompt_tokens if usage else 0
        completion = usage.completion_tokens if usage else 0
        cached = _cached_tokens(usage)
        result = LLMResult(
            text=resp.choices[0].message.content or "",
            provider=spec.provider,
            model=spec.model,
            prompt_tokens=prompt,
            completion_tokens=completion,
            cached_tokens=cached,
            cost_usd=_cost(await self.price(spec, called_at), prompt, completion, cached),
            latency_ms=latency_ms,
        )
        if self.on_usage:
            self.on_usage(slot, result)
        return result


def _cost(price: Price | None, prompt: int, completion: int, cached: int) -> float | None:
    if price is None:
        return None
    uncached = max(prompt - cached, 0)
    return (uncached * price.input + cached * price.cached_input + completion * price.output) / 1_000_000


def _cached_tokens(usage) -> int:
    if usage is None:
        return 0
    # DeepSeek reports prompt_cache_hit_tokens; OpenAI-style providers use prompt_tokens_details.cached_tokens.
    hit = getattr(usage, "prompt_cache_hit_tokens", None)
    if hit is not None:
        return hit
    details = getattr(usage, "prompt_tokens_details", None)
    return (getattr(details, "cached_tokens", None) or 0) if details else 0
