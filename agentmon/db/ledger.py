"""Writes every LLM call the monitor makes to the cost ledger, attributed to the run being processed."""

from contextvars import ContextVar

from agentmon.db.schema import LLMCallRow
from agentmon.db.session import session_scope
from agentmon.llm import LLMResult

current_run_id: ContextVar[str | None] = ContextVar("current_run_id", default=None)


def record_usage(slot: str, result: LLMResult) -> None:
    with session_scope() as s:
        s.add(
            LLMCallRow(
                slot=slot,
                run_id=current_run_id.get(),
                provider=result.provider,
                model=result.model,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                cached_tokens=result.cached_tokens,
                cost_usd=result.cost_usd,
                latency_ms=result.latency_ms,
            )
        )
