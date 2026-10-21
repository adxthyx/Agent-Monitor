"""One monitoring cycle: sync the suite, run every selected probe N times, grade, persist, then evaluate incidents."""

import asyncio
import time
from collections.abc import Callable
from datetime import UTC, datetime

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from agentmon.adapters import build_adapter
from agentmon.adapters.base import MONITOR_HEADER
from agentmon.db import session_scope
from agentmon.db.ledger import current_run_id
from agentmon.db.schema import CheckRow, CycleRow, ProbeRow, RunRow, TargetRow, VerdictRow, new_id
from agentmon.engine.pipeline import Graded, grade
from agentmon.engine.runner import execute_probe
from agentmon.incidents import Notification, evaluate_cycle, notify
from agentmon.judge import Judge
from agentmon.llm import LLMRouter
from agentmon.models import Probe, RunTrace, Suite

RunCallback = Callable[[Probe, int, RunTrace, Graded, str], None]


def sync_suite(session: Session, suite: Suite) -> dict[str, int]:
    """Upsert the target and one row per probe version. Returns probe id -> probe row id."""
    t = suite.target
    row = session.get(TargetRow, t.id)
    config = t.model_dump(mode="json")
    if row is None:
        session.add(TargetRow(id=t.id, name=t.name, model_family=t.model_family, config=config))
    else:
        row.name, row.model_family, row.config = t.name, t.model_family, config

    ids: dict[str, int] = {}
    for p in suite.probes:
        h = p.content_hash()
        existing = session.scalar(
            select(ProbeRow).where(ProbeRow.suite_id == suite.id, ProbeRow.probe_key == p.id, ProbeRow.content_hash == h)
        )
        if existing is None:
            existing = ProbeRow(
                suite_id=suite.id,
                target_id=t.id,
                probe_key=p.id,
                content_hash=h,
                cls=p.cls.value,
                axis=p.axis.value,
                severity=p.severity,
                body=p.model_dump(mode="json", by_alias=True),
            )
            session.add(existing)
            session.flush()
        ids[p.id] = existing.id
    return ids


async def classic_check(url: str | None, client: httpx.AsyncClient, cycle_id: str) -> dict | None:
    """What a traditional uptime monitor sees: status code and latency only."""
    if not url:
        return None
    started = time.perf_counter()
    try:
        resp = await client.get(url, headers={MONITOR_HEADER: f"uptime/{cycle_id}"}, timeout=10)
        return {"url": url, "status_code": resp.status_code, "ok": resp.status_code < 400, "latency_ms": _ms(started)}
    except httpx.HTTPError as e:
        return {"url": url, "status_code": None, "ok": False, "latency_ms": _ms(started), "error": str(e)}


def _ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def persist_run(
    session: Session, run_id: str, cycle_id: str, probe_row_id: int, attempt: int, trace: RunTrace, graded: Graded, started: datetime
) -> None:
    response_hash = trace.response_hash()
    run = RunRow(
        id=run_id,
        cycle_id=cycle_id,
        probe_id=probe_row_id,
        attempt=attempt,
        started_at=started,
        finished_at=datetime.now(UTC),
        transcript=[t.model_dump(mode="json") for t in trace.transcript],
        transport_error=trace.transport_error.model_dump() if trace.transport_error else None,
        retried_errors=[e.model_dump() for e in trace.retried_errors],
        tools_visible=trace.tools_visible,
        latency_ms=trace.total_latency_ms,
        response_hash=response_hash,
        status=graded.status.value,
        failure_class=graded.failure_class.value if graded.failure_class else None,
        decided_by=graded.decided_by,
    )
    run.checks = [
        CheckRow(
            name=c.name,
            passed=c.passed,
            hard=c.hard,
            failure_class=c.failure_class.value if c.failure_class else None,
            detail=c.detail,
        )
        for c in graded.checks
    ]
    if graded.judge:
        run.verdicts = [
            VerdictRow(
                judge_provider=v.provider,
                judge_model=v.model,
                rubric_version=v.rubric_version,
                response_hash=response_hash,
                attempt=v.attempt,
                valid=v.valid,
                invalid_reason=v.invalid_reason,
                verdict=v.verdict,
                confidence=v.confidence,
                failure_class=v.failure_class.value if v.failure_class else None,
                reasoning=v.reasoning,
                evidence=v.evidence,
                raw=v.raw,
            )
            for v in graded.judge.verdicts
        ]
    session.add(run)


async def run_cycle(
    suite: Suite,
    router: LLMRouter,
    *,
    attempts: int | None = None,
    probe_ids: list[str] | None = None,
    kind: str = "manual",
    note: str | None = None,
    use_judge: bool = True,
    on_run: RunCallback | None = None,
) -> tuple[str, list[Notification]]:
    probes = [suite.probe(pid) for pid in probe_ids] if probe_ids else suite.probes
    attempts = attempts or suite.runs_per_probe
    judge = Judge(router, suite.target.model_family) if use_judge else None

    with session_scope() as s:
        probe_rows = sync_suite(s, suite)
        cycle = CycleRow(id=new_id(), target_id=suite.target.id, suite_id=suite.id, kind=kind, note=note)
        s.add(cycle)
        cycle_id = cycle.id

    sem = asyncio.Semaphore(suite.target.max_concurrency)
    async with httpx.AsyncClient() as client, build_adapter(suite.target.adapter, client) as adapter:
        classic = await classic_check(suite.target.health_url, client, cycle_id)

        async def one(probe: Probe, attempt: int) -> None:
            async with sem:
                run_id = new_id()
                token = current_run_id.set(run_id)
                try:
                    started = datetime.now(UTC)
                    trace = await execute_probe(probe, adapter, run_id, retries=suite.target.transport_retries)
                    graded = await grade(trace, probe, suite, judge)
                    with session_scope() as s:
                        persist_run(s, run_id, cycle_id, probe_rows[probe.id], attempt, trace, graded, started)
                finally:
                    current_run_id.reset(token)
                if on_run:
                    on_run(probe, attempt, trace, graded, run_id)

        await asyncio.gather(*(one(p, a) for p in probes for a in range(1, attempts + 1)))

    with session_scope() as s:
        row = s.get(CycleRow, cycle_id)
        row.finished_at = datetime.now(UTC)
        row.classic_check = classic
        notifications = evaluate_cycle(s, cycle_id, suite)
    await notify(notifications)
    return cycle_id, notifications
