"""Results store. Every transcript, check, verdict and LLM call is kept so any alert can be replayed."""

import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

JSONType = JSON().with_variant(JSONB(), "postgresql")


def utcnow() -> datetime:
    return datetime.now(UTC)


def new_id() -> str:
    return uuid.uuid4().hex


class Base(DeclarativeBase):
    pass


class TargetRow(Base):
    __tablename__ = "targets"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    model_family: Mapped[str | None] = mapped_column(String(64))
    config: Mapped[dict] = mapped_column(JSONType)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class ProbeRow(Base):
    """One version of a probe. Editing a probe changes content_hash, which creates a new row and resets its baseline."""

    __tablename__ = "probes"
    __table_args__ = (UniqueConstraint("suite_id", "probe_key", "content_hash"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    suite_id: Mapped[str] = mapped_column(String(64), index=True)
    target_id: Mapped[str] = mapped_column(ForeignKey("targets.id"))
    probe_key: Mapped[str] = mapped_column(String(128))
    content_hash: Mapped[str] = mapped_column(String(32))
    cls: Mapped[str] = mapped_column(String(32))
    axis: Mapped[str] = mapped_column(String(32))
    severity: Mapped[str] = mapped_column(String(4))
    status: Mapped[str] = mapped_column(String(16), default="approved")  # draft | approved | rejected
    body: Mapped[dict] = mapped_column(JSONType)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class CycleRow(Base):
    __tablename__ = "cycles"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    target_id: Mapped[str] = mapped_column(ForeignKey("targets.id"), index=True)
    suite_id: Mapped[str] = mapped_column(String(64))
    kind: Mapped[str] = mapped_column(String(16), default="manual")  # scheduled | manual | ci
    note: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    classic_check: Mapped[dict | None] = mapped_column(JSONType)  # plain HTTP uptime result, for comparison

    runs: Mapped[list["RunRow"]] = relationship(back_populates="cycle")


class RunRow(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    cycle_id: Mapped[str] = mapped_column(ForeignKey("cycles.id"), index=True)
    probe_id: Mapped[int] = mapped_column(ForeignKey("probes.id"), index=True)
    attempt: Mapped[int] = mapped_column(Integer, default=1)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    transcript: Mapped[list] = mapped_column(JSONType)
    transport_error: Mapped[dict | None] = mapped_column(JSONType)
    retried_errors: Mapped[list] = mapped_column(JSONType, default=list)  # transient errors the runner retried past
    tools_visible: Mapped[bool] = mapped_column(Boolean, default=False)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    response_hash: Mapped[str | None] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16))  # pass | fail | uncertain | judge_error
    failure_class: Mapped[str | None] = mapped_column(String(32))
    decided_by: Mapped[str | None] = mapped_column(String(32))  # which stage set the status

    cycle: Mapped[CycleRow] = relationship(back_populates="runs")
    probe: Mapped[ProbeRow] = relationship()
    checks: Mapped[list["CheckRow"]] = relationship(back_populates="run", cascade="all, delete-orphan")
    verdicts: Mapped[list["VerdictRow"]] = relationship(back_populates="run", cascade="all, delete-orphan")


class CheckRow(Base):
    __tablename__ = "check_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), index=True)
    name: Mapped[str] = mapped_column(String(32))
    passed: Mapped[bool] = mapped_column(Boolean)
    hard: Mapped[bool] = mapped_column(Boolean)
    failure_class: Mapped[str | None] = mapped_column(String(32))
    detail: Mapped[str] = mapped_column(Text, default="")

    run: Mapped[RunRow] = relationship(back_populates="checks")


class VerdictRow(Base):
    __tablename__ = "verdicts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), index=True)
    judge_provider: Mapped[str] = mapped_column(String(32))
    judge_model: Mapped[str] = mapped_column(String(128))
    rubric_version: Mapped[str] = mapped_column(String(32))
    response_hash: Mapped[str | None] = mapped_column(String(32), index=True)  # transcript graded; judge-cache key
    attempt: Mapped[int] = mapped_column(Integer)
    valid: Mapped[bool] = mapped_column(Boolean)
    invalid_reason: Mapped[str | None] = mapped_column(Text)
    verdict: Mapped[str | None] = mapped_column(String(8))
    confidence: Mapped[float | None] = mapped_column(Float)
    failure_class: Mapped[str | None] = mapped_column(String(32))
    reasoning: Mapped[str | None] = mapped_column(Text)
    evidence: Mapped[list] = mapped_column(JSONType, default=list)
    raw: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    run: Mapped[RunRow] = relationship(back_populates="verdicts")


class LLMCallRow(Base):
    """Cost ledger: one row per LLM call the monitor makes."""

    __tablename__ = "llm_calls"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    slot: Mapped[str] = mapped_column(String(32), index=True)
    run_id: Mapped[str | None] = mapped_column(String(32), index=True)  # set when the call grades or drives a run
    provider: Mapped[str] = mapped_column(String(32))
    model: Mapped[str] = mapped_column(String(128))
    prompt_tokens: Mapped[int] = mapped_column(Integer)
    completion_tokens: Mapped[int] = mapped_column(Integer)
    cached_tokens: Mapped[int] = mapped_column(Integer)
    cost_usd: Mapped[float | None] = mapped_column(Float)
    latency_ms: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class IncidentRow(Base):
    __tablename__ = "incidents"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    target_id: Mapped[str] = mapped_column(ForeignKey("targets.id"), index=True)
    probe_key: Mapped[str] = mapped_column(String(128))
    axis: Mapped[str] = mapped_column(String(32))
    failure_class: Mapped[str | None] = mapped_column(String(32))
    severity: Mapped[str] = mapped_column(String(4))
    status: Mapped[str] = mapped_column(String(16), default="open")  # open | resolved
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    first_run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"))
    last_run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"))
    payload: Mapped[dict] = mapped_column(JSONType)
