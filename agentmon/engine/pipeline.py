"""Grading pipeline for one run: deterministic checks first, the LLM judge last, latency recorded alongside."""

from pydantic import BaseModel

from agentmon.checks import CheckResult, run_deterministic_checks
from agentmon.checks.deterministic import check_latency
from agentmon.judge import Judge, JudgeOutcome
from agentmon.models import FailureClass, Probe, RunStatus, RunTrace, Suite


class Graded(BaseModel):
    status: RunStatus
    failure_class: FailureClass | None
    decided_by: str
    checks: list[CheckResult]
    judge: JudgeOutcome | None = None


async def grade(trace: RunTrace, probe: Probe, suite: Suite, judge: Judge | None) -> Graded:
    checks = run_deterministic_checks(trace, probe, suite.target, suite.profile)
    hard_fail = next((c for c in checks if c.hard and not c.passed), None)
    if hard_fail:
        return Graded(status=RunStatus.failed, failure_class=hard_fail.failure_class, decided_by=hard_fail.name, checks=checks)

    status, failure, decided_by, outcome = RunStatus.passed, None, "checks", None
    if judge and probe.expect.judge:
        outcome = await judge.grade(probe, trace, suite.profile)
        status, failure, decided_by = outcome.status, outcome.failure_class, "judge"

    latency = check_latency(trace, probe, suite.target)
    checks.append(latency)
    if status == RunStatus.passed and not latency.passed:
        status, failure, decided_by = RunStatus.failed, FailureClass.latency, "latency"
    return Graded(status=status, failure_class=failure, decided_by=decided_by, checks=checks, judge=outcome)
