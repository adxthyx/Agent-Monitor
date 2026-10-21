import asyncio
import signal
from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy import func, select, text

from agentmon.db import get_engine, session_scope
from agentmon.db.ledger import record_usage
from agentmon.db.schema import CycleRow, IncidentRow, LLMCallRow, ProbeRow, RunRow
from agentmon.engine.cycle import run_cycle
from agentmon.engine.pipeline import Graded
from agentmon.llm import LLMRouter, load_slots
from agentmon.models import Probe, RunStatus, RunTrace, load_suite
from agentmon.settings import get_settings

app = typer.Typer(no_args_is_help=True, help="Semantic uptime monitor for AI agents.")
console = Console()

STATUS_STYLE = {"pass": "green", "fail": "bold red", "uncertain": "yellow", "judge_error": "magenta"}


def make_router(ledger: bool = True) -> LLMRouter:
    return LLMRouter(load_slots(get_settings().slots_path), on_usage=record_usage if ledger else None)


@app.command()
def doctor():
    """Check the database and that every configured LLM slot is reachable, answers and is priced. Exits 1 if not green."""
    failures = 0
    try:
        with get_engine().connect() as conn:
            conn.execute(text("select 1"))
        console.print("[green]database ok[/]")
    except Exception as e:
        failures += 1
        console.print(f"[red]database unreachable:[/] {e}")

    router = make_router(ledger=False)

    async def check() -> Table:
        nonlocal failures
        table = Table("slot", "provider", "model", "listed", "call", "latency", "$/1M in · cached · out (now)")
        listed: dict[str, list[str] | str] = {}
        for name in router.config.providers:
            try:
                listed[name] = await router.list_models(name)
            except Exception as e:
                listed[name] = f"error: {e}"
        for slot, specs in router.config.slots.items():
            for spec in specs:
                models = listed.get(spec.provider)
                on_list = "?" if isinstance(models, str) else ("yes" if spec.model in models else "[red]NO[/]")
                failures += on_list != "yes"
                try:
                    if slot == "embeddings":
                        await router.client(spec.provider).embeddings.create(model=spec.model, input="ping")
                        call, latency = "[green]ok[/]", "-"
                    else:
                        r = await router.chat(spec, [{"role": "user", "content": "Reply with the word OK."}], slot="doctor", max_tokens=200)
                        call, latency = f"[green]ok[/] {r.text.strip()[:20]!r}", f"{r.latency_ms} ms"
                except Exception as e:
                    failures += 1
                    call, latency = f"[red]{type(e).__name__}: {str(e)[:60]}[/]", "-"
                p = await router.price(spec, datetime.now(UTC))
                price = f"{p.input:g} · {p.cached_input:g} · {p.output:g}" if p else "[yellow]unpriced[/]"
                table.add_row(slot, spec.provider, spec.model, on_list, call, latency, price)
        return table

    console.print(asyncio.run(check()))
    if failures:
        console.print(f"[red]{failures} problem(s)[/]")
        raise typer.Exit(1)


@app.command()
def run(
    suite_path: Path = typer.Argument(..., exists=True, help="Suite YAML"),
    attempts: int = typer.Option(None, "--attempts", "-n", help="Runs per probe (default: suite setting)"),
    probe: list[str] = typer.Option(None, "--probe", "-p", help="Only these probe ids"),
    no_judge: bool = typer.Option(False, "--no-judge", help="Deterministic checks only"),
    note: str = typer.Option(None, "--note", help="Free-text note stored on the cycle"),
):
    """Run one monitoring cycle now."""
    suite = load_suite(suite_path)
    asyncio.run(_cycle(suite, attempts, probe, no_judge, note, kind="manual"))


async def _cycle(suite, attempts, probe_ids, no_judge, note, kind):
    results: list[tuple[Probe, int, RunTrace, Graded, str]] = []

    def on_run(probe, attempt, trace, graded, run_id):
        results.append((probe, attempt, trace, graded, run_id))
        style = STATUS_STYLE[graded.status.value]
        console.print(f"  [{style}]{graded.status.value:<11}[/] {probe.id} #{attempt}", highlight=False)

    console.print(f"[bold]cycle[/] {suite.id} -> {suite.target.name}")
    cycle_id, notes = await run_cycle(
        suite, make_router(), attempts=attempts, probe_ids=probe_ids, kind=kind, note=note, use_judge=not no_judge, on_run=on_run
    )

    order = {p.id: i for i, p in enumerate(suite.probes)}
    results.sort(key=lambda r: (order[r[0].id], r[1]))
    table = Table("probe", "class", "sev", "status", "failure", "decided by", "latency", "run")
    for probe, attempt, trace, graded, run_id in results:
        style = STATUS_STYLE[graded.status.value]
        table.add_row(
            f"{probe.id}" + (f" #{attempt}" if attempt > 1 else ""),
            probe.cls.value,
            probe.severity,
            f"[{style}]{graded.status.value}[/]",
            graded.failure_class.value if graded.failure_class else "",
            graded.decided_by,
            f"{trace.total_latency_ms} ms",
            run_id[:8],
        )
    console.print(table)

    with session_scope() as s:
        classic = s.get(CycleRow, cycle_id).classic_check
    passed = sum(1 for r in results if r[3].status == RunStatus.passed)
    if classic:
        colour = "green" if classic["ok"] else "red"
        console.print(f"Classic uptime check ({classic['url']}): [{colour}]HTTP {classic['status_code']}[/]")
    colour = "green" if passed == len(results) else "red"
    console.print(f"Semantic check: [{colour}]{passed}/{len(results)} runs passed[/]   cycle {cycle_id}")
    if notes:
        console.print(f"{sum(n.kind == 'opened' for n in notes)} incident(s) opened, {sum(n.kind == 'resolved' for n in notes)} resolved")


@app.command()
def watch(
    suite_paths: list[Path] = typer.Argument(..., exists=True, help="One or more suite YAMLs"),
    interval: int = typer.Option(None, "--interval", help="Minutes between cycles (default: each suite's interval_minutes)"),
    no_judge: bool = typer.Option(False, "--no-judge"),
):
    """Run cycles on a schedule until interrupted. A cycle still running when the next is due is skipped, not stacked."""
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.interval import IntervalTrigger

    async def main() -> None:
        scheduler = AsyncIOScheduler(timezone=UTC)

        async def job(path: Path, minutes: int) -> None:
            try:
                suite = load_suite(path)  # re-read so suite edits apply on the next cycle
                await _cycle(suite, None, None, no_judge, None, kind="scheduled")
            except Exception as e:  # one bad cycle must not stop the schedule
                console.print(f"[red]cycle for {path} failed:[/] {type(e).__name__}: {e}")
                return
            wanted = interval or suite.interval_minutes
            if wanted != minutes:
                scheduler.reschedule_job(str(path), trigger=IntervalTrigger(minutes=wanted))
                scheduler.modify_job(str(path), args=[path, wanted])
            console.print(f"next {suite.id} cycle at {scheduler.get_job(str(path)).next_run_time:%H:%M:%S %Z}")

        for path in suite_paths:
            minutes = interval or load_suite(path).interval_minutes
            scheduler.add_job(
                job,
                IntervalTrigger(minutes=minutes),
                args=[path, minutes],
                id=str(path),
                next_run_time=datetime.now(UTC),
                max_instances=1,
                coalesce=True,
                misfire_grace_time=60,
            )
        scheduler.start()
        stop = asyncio.Event()
        asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, stop.set)  # service managers stop with SIGTERM
        try:
            await stop.wait()
        finally:
            scheduler.shutdown(wait=False)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    console.print("stopped")


@app.command()
def show(run_id: str = typer.Argument(..., help="Run id or unique prefix")):
    """Show a run: transcript, checks and judge verdicts with evidence."""
    with session_scope() as s:
        runs = s.scalars(select(RunRow).where(RunRow.id.startswith(run_id))).all()
        if len(runs) != 1:
            console.print(f"[red]{len(runs)} runs match {run_id!r}[/]")
            raise typer.Exit(1)
        run = runs[0]
        probe = run.probe
        style = STATUS_STYLE[run.status]
        console.print(f"[bold]{probe.probe_key}[/] ({probe.cls}, {probe.severity}) [{style}]{run.status}[/] {run.failure_class or ''}")
        console.print(f"expected: {probe.body['expect']['outcome']}\n")
        for t in run.transcript:
            who = "[cyan]USER [/]" if t["role"] == "user" else "[magenta]AGENT[/]"
            console.print(f"{who} {t['content']}", highlight=False)
            for c in t.get("tool_calls", []):
                console.print(f"      [dim]tool: {c['name']}({c['arguments']})[/]", highlight=False)
        for e in run.retried_errors:
            console.print(f"[yellow]retried after transport error:[/] {e['kind']}: {e['detail'][:200]}", highlight=False)
        if run.transport_error:
            console.print(f"[red]transport error:[/] {run.transport_error}")
        console.print()
        for c in run.checks:
            mark = "[green]ok[/]" if c.passed else "[red]FAIL[/]"
            console.print(f"check {c.name:<16} {mark} {c.detail}", highlight=False)
        for v in run.verdicts:
            if not v.valid:
                console.print(f"[yellow]judge attempt {v.attempt} invalid:[/] {v.invalid_reason}")
                continue
            console.print(f"\n[bold]judge[/] {v.judge_model} (rubric {v.rubric_version}): {v.verdict} conf={v.confidence}")
            console.print(f"  {v.reasoning}", highlight=False)
            for q in v.evidence:
                console.print(f"  [dim]> {q}[/]", highlight=False)


@app.command()
def runs(limit: int = typer.Option(20, "--limit", "-l")):
    """Most recent runs."""
    with session_scope() as s:
        rows = s.execute(
            select(RunRow, ProbeRow.probe_key).join(ProbeRow).order_by(RunRow.started_at.desc()).limit(limit)
        ).all()
        table = Table("run", "at", "probe", "status", "failure")
        for run, key in rows:
            style = STATUS_STYLE[run.status]
            table.add_row(run.id[:8], f"{run.started_at:%m-%d %H:%M:%S}", key, f"[{style}]{run.status}[/]", run.failure_class or "")
        console.print(table)


@app.command()
def incidents(all_: bool = typer.Option(False, "--all", help="Include resolved")):
    """Open (or all) incidents."""
    with session_scope() as s:
        q = select(IncidentRow).order_by(IncidentRow.opened_at.desc())
        if not all_:
            q = q.where(IncidentRow.status == "open")
        table = Table("incident", "opened", "sev", "probe", "axis", "failure", "status", "first run")
        for i in s.scalars(q):
            table.add_row(
                i.id[:8], f"{i.opened_at:%m-%d %H:%M}", i.severity, i.probe_key, i.axis, i.failure_class or "", i.status, i.first_run_id[:8]
            )
        console.print(table)


@app.command()
def cost():
    """LLM spend from the cost ledger, by slot and model."""
    with session_scope() as s:
        rows = s.execute(
            select(
                LLMCallRow.slot,
                LLMCallRow.model,
                func.count(),
                func.sum(LLMCallRow.prompt_tokens),
                func.sum(LLMCallRow.cached_tokens),
                func.sum(LLMCallRow.completion_tokens),
                func.sum(LLMCallRow.cost_usd),
                func.count(LLMCallRow.cost_usd),
            ).group_by(LLMCallRow.slot, LLMCallRow.model)
        ).all()
        table = Table("slot", "model", "calls", "prompt tok", "cached", "completion tok", "cost USD")
        for slot, model, n, pt, ct, comp, usd, priced in rows:
            cost_str = f"{usd:.4f}" if priced == n else (f"{usd or 0:.4f} (+{n - priced} unpriced)" if priced else "unpriced")
            table.add_row(slot, model, str(n), str(pt), str(ct), str(comp), cost_str)
        console.print(table)


if __name__ == "__main__":
    app()
