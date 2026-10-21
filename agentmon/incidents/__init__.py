"""Phase 1 alerting: open an incident when a probe fails, dedupe while it stays open, resolve when it passes.

Phase 2 replaces the trigger with pass-rate intervals over a rolling window, baselines and persistence.
"""

import asyncio
import smtplib
from datetime import UTC, datetime
from email.message import EmailMessage

import httpx
from pydantic import BaseModel
from rich.console import Console
from sqlalchemy import select
from sqlalchemy.orm import Session

from agentmon.db.schema import CycleRow, IncidentRow, ProbeRow, RunRow
from agentmon.models import RunStatus, Suite
from agentmon.settings import Settings, get_settings

console = Console(stderr=True)


class Notification(BaseModel):
    kind: str  # opened | resolved
    incident_id: str
    severity: str
    probe_key: str
    text: str
    payload: dict


def _run_summary(run: RunRow) -> dict:
    valid = [v for v in run.verdicts if v.valid]
    v = valid[-1] if valid else None
    return {
        "run_id": run.id,
        "at": run.started_at.isoformat(),
        "status": run.status,
        "failure_class": run.failure_class,
        "decided_by": run.decided_by,
        "transcript": [{"role": t["role"], "content": t["content"], "tool_calls": t.get("tool_calls", [])} for t in run.transcript],
        "failed_checks": [{"name": c.name, "detail": c.detail} for c in run.checks if not c.passed],
        "judge": {"model": v.judge_model, "reasoning": v.reasoning, "evidence": v.evidence, "confidence": v.confidence} if v else None,
    }


def _last_pass(session: Session, suite_id: str, probe_key: str, before_cycle: str) -> RunRow | None:
    return session.scalar(
        select(RunRow)
        .join(ProbeRow, RunRow.probe_id == ProbeRow.id)
        .where(ProbeRow.suite_id == suite_id, ProbeRow.probe_key == probe_key, RunRow.status == RunStatus.passed.value)
        .where(RunRow.cycle_id != before_cycle)
        .order_by(RunRow.started_at.desc())
        .limit(1)
    )


def evaluate_cycle(session: Session, cycle_id: str, suite: Suite) -> list[Notification]:
    cycle = session.get(CycleRow, cycle_id)
    by_probe: dict[str, list[RunRow]] = {}
    for run in cycle.runs:
        by_probe.setdefault(run.probe.probe_key, []).append(run)

    notes: list[Notification] = []
    for key, runs in by_probe.items():
        probe = suite.probe(key)
        failed = [r for r in runs if r.status == RunStatus.failed.value]
        all_passed = all(r.status == RunStatus.passed.value for r in runs)
        incident = session.scalar(
            select(IncidentRow).where(
                IncidentRow.target_id == suite.target.id, IncidentRow.probe_key == key, IncidentRow.status == "open"
            )
        )
        if failed and incident is None:
            last_pass = _last_pass(session, suite.id, key, cycle_id)
            payload = {
                "target": suite.target.name,
                "probe": key,
                "class": probe.cls.value,
                "axis": probe.axis.value,
                "severity": probe.severity,
                "expected": probe.expect.outcome,
                "failure_class": failed[0].failure_class,
                "failed_runs": f"{len(failed)}/{len(runs)}",
                "classic_uptime_check": cycle.classic_check,
                "failing": _run_summary(failed[0]),
                "last_passing": _run_summary(last_pass) if last_pass else None,
            }
            incident = IncidentRow(
                target_id=suite.target.id,
                probe_key=key,
                axis=probe.axis.value,
                failure_class=failed[0].failure_class,
                severity=probe.severity,
                first_run_id=failed[0].id,
                last_run_id=failed[0].id,
                payload=payload,
            )
            session.add(incident)
            session.flush()
            notes.append(_opened(incident, payload))
        elif failed and incident is not None:
            incident.last_run_id = failed[0].id
        elif all_passed and incident is not None:
            incident.status = "resolved"
            incident.resolved_at = datetime.now(UTC)
            notes.append(
                Notification(
                    kind="resolved",
                    incident_id=incident.id,
                    severity=incident.severity,
                    probe_key=key,
                    text=f"[RESOLVED] {suite.target.name}: probe `{key}` passing again.",
                    payload={"incident_id": incident.id},
                )
            )
    return notes


def _opened(incident: IncidentRow, p: dict) -> Notification:
    failing = p["failing"]
    lines = [
        f"[{p['severity']}] {p['target']}: `{p['probe']}` failed ({p['failure_class']}, axis {p['axis']}, {p['failed_runs']} runs)",
        f"Expected: {p['expected']}",
    ]
    classic = p.get("classic_uptime_check")
    if classic:
        lines.append(f"Classic uptime check: HTTP {classic.get('status_code')} ({'green' if classic.get('ok') else 'red'})")
    agent_turns = [t for t in failing["transcript"] if t["role"] == "agent"]
    if agent_turns:
        lines.append(f"Agent said: {agent_turns[-1]['content'][:400]}")
    for c in failing["failed_checks"]:
        lines.append(f"Check `{c['name']}`: {c['detail']}")
    if failing["judge"]:
        lines.append(f"Judge ({failing['judge']['model']}): {failing['judge']['reasoning']}")
        lines += [f"  > {q}" for q in failing["judge"]["evidence"][:3]]
    return Notification(
        kind="opened", incident_id=incident.id, severity=incident.severity, probe_key=p["probe"], text="\n".join(lines), payload=p
    )


def _transcript_text(summary: dict) -> str:
    lines = []
    for t in summary["transcript"]:
        for c in t.get("tool_calls", []):
            lines.append(f"  [tool] {c['name']}({c['arguments']})")
        lines.append(f"  {'USER ' if t['role'] == 'user' else 'AGENT'}: {t['content']}")
    return "\n".join(lines)


def email_message(n: Notification, settings: Settings) -> EmailMessage:
    """Plain-text alert email: the summary line as subject, then the full failing and last passing transcripts."""
    msg = EmailMessage()
    msg["Subject"] = n.text.splitlines()[0].replace("`", "")
    msg["From"] = settings.alert_email_from
    msg["To"] = settings.alert_email_to
    body = [n.text]
    if n.kind == "opened":
        failing = n.payload["failing"]
        body += ["", f"Failing run {failing['run_id']} at {failing['at']}:", _transcript_text(failing)]
        if failing["judge"]:
            body += ["", "Judge evidence:", *[f"  > {q}" for q in failing["judge"]["evidence"]]]
        if last := n.payload.get("last_passing"):
            body += ["", f"Last passing run {last['run_id']} at {last['at']}:", _transcript_text(last)]
        body += ["", f"Replay: agentmon show {failing['run_id'][:8]}"]
    msg.set_content("\n".join(body))
    return msg


def _send_email(messages: list[EmailMessage], s: Settings) -> None:
    smtp_cls = smtplib.SMTP_SSL if s.smtp_security == "ssl" else smtplib.SMTP
    with smtp_cls(s.smtp_host, s.smtp_port, timeout=20) as smtp:
        if s.smtp_security == "starttls":
            smtp.starttls()
        if s.smtp_user:
            smtp.login(s.smtp_user, s.smtp_password or "")
        for m in messages:
            smtp.send_message(m)


async def notify(notes: list[Notification]) -> None:
    if not notes:
        return
    s = get_settings()
    for n in notes:
        console.print(f"[bold {'red' if n.kind == 'opened' else 'green'}]ALERT[/] {n.text}\n")
    if s.alert_webhook_url:
        async with httpx.AsyncClient(timeout=10) as client:
            for n in notes:
                try:
                    await client.post(s.alert_webhook_url, json={"text": n.text, "incident": n.payload | {"kind": n.kind}})
                except httpx.HTTPError as e:
                    console.print(f"[yellow]webhook delivery failed: {e}[/]")
    if s.smtp_host and s.alert_email_to:
        try:
            await asyncio.to_thread(_send_email, [email_message(n, s) for n in notes], s)
        except (smtplib.SMTPException, OSError) as e:
            console.print(f"[yellow]email delivery failed: {e}[/]")
