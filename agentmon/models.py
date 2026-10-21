"""Domain objects: Target, AgentProfile, Probe, Suite, and the in-memory shape of a Run.

Targets, profiles and suites are config-as-code (YAML in git); runs and verdicts live in the database.
"""

import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class ProbeClass(StrEnum):
    fact_recall = "fact_recall"
    unanswerable = "unanswerable"
    tool_required = "tool_required"
    tool_forbidden = "tool_forbidden"
    should_refuse = "should_refuse"
    should_not_refuse = "should_not_refuse"
    multi_turn_memory = "multi_turn_memory"
    format_contract = "format_contract"
    policy = "policy"
    canary = "canary"


class FailureClass(StrEnum):
    transport = "transport"
    format_break = "format_break"
    pii_leak = "pii_leak"
    hallucination = "hallucination"
    ungrounded_claim = "ungrounded_claim"
    wrong_tool = "wrong_tool"
    over_refusal = "over_refusal"
    under_refusal = "under_refusal"
    policy_violation = "policy_violation"
    memory_loss = "memory_loss"
    latency = "latency"
    model_swap = "model_swap"


class Axis(StrEnum):
    correctness = "correctness"
    groundedness = "groundedness"
    tool_accuracy = "tool_accuracy"
    refusal_balance = "refusal_balance"
    policy = "policy"
    latency = "latency"
    format = "format"


PROBE_AXIS: dict[ProbeClass, Axis] = {
    ProbeClass.fact_recall: Axis.correctness,
    ProbeClass.canary: Axis.correctness,
    ProbeClass.multi_turn_memory: Axis.correctness,
    ProbeClass.unanswerable: Axis.groundedness,
    ProbeClass.tool_required: Axis.tool_accuracy,
    ProbeClass.tool_forbidden: Axis.tool_accuracy,
    ProbeClass.should_refuse: Axis.refusal_balance,
    ProbeClass.should_not_refuse: Axis.refusal_balance,
    ProbeClass.policy: Axis.policy,
    ProbeClass.format_contract: Axis.format,
}


class RunStatus(StrEnum):
    passed = "pass"
    failed = "fail"
    uncertain = "uncertain"
    judge_error = "judge_error"


# --- Target & adapter ---------------------------------------------------------------------------


class ResponseMapping(BaseModel):
    reply: str = Field(description="JSONPath to the reply text")
    tool_calls: str | None = Field(None, description="JSONPath to a list of tool calls, if the target exposes them")
    tool_name: str = "name"
    tool_args: str = "arguments"


class StreamMapping(BaseModel):
    """Where the reply lives in a server-sent-event stream. Defaults match OpenAI-style chat completion chunks."""

    delta: str | None = Field("$.choices[0].delta.content", description="JSONPath to the text fragment in each event's data; null = the raw data is the fragment")
    tool_calls: str | None = Field(None, description="JSONPath to complete tool calls in an event, collected across the stream")
    tool_name: str = "name"
    tool_args: str = "arguments"
    done: str | None = Field("[DONE]", description="Data value that ends the stream; null = the stream ends when the connection closes")
    event: str | None = Field(None, description="Only events of this `event:` type carry reply fragments")


class CallbackSpec(BaseModel):
    """webhook: how the agent's asynchronous reply is matched to the turn that is waiting for it."""

    correlate: str | None = Field(
        None,
        description="JSONPath in the callback body holding the correlation_id or session_id that was sent. Needed only when the "
        "agent posts to the fixed /callbacks URL instead of the per-turn {{callback_url}}",
    )


class AdapterSpec(BaseModel):
    kind: Literal["openai_chat", "json_post", "sse", "webhook"]
    url: str
    headers: dict[str, str] = {}
    # Request body template (Jinja2, sandboxed). Variables: session_id, user_turn, history, run_id; webhook adds
    # callback_url and correlation_id. A string that is a single {{ expression }} keeps its type. ${ENV} in any string.
    body: dict[str, Any] | None = None
    # openai_chat / sse: model name sent to the target, if it needs one.
    model: str | None = None
    # Where the reply lives: the JSON response (json_post, openai_chat) or the callback body (webhook).
    response: ResponseMapping | None = None
    stream: StreamMapping | None = None  # sse
    callback: CallbackSpec | None = None  # webhook
    state: Literal["server_session", "client_history"] = "client_history"
    timeout_s: float = 60.0

    @model_validator(mode="after")
    def _check(self):
        if self.kind in ("json_post", "webhook") and (self.body is None or self.response is None):
            raise ValueError(f"{self.kind} adapter needs `body` and `response`")
        return self


class Target(BaseModel):
    id: str
    name: str
    adapter: AdapterSpec
    model_family: str | None = Field(None, description="Agent's LLM family; judges of the same family are excluded")
    max_concurrency: int = 4
    transport_retries: int = Field(2, ge=0, description="Re-runs of a probe after a transient transport error")
    latency_budget_ms: int = 15_000
    health_url: str | None = Field(None, description="Classic uptime endpoint, checked alongside for comparison")


# --- Agent profile ------------------------------------------------------------------------------


class AgentProfile(BaseModel):
    purpose: str
    allowed_tools: list[str] = []
    forbidden_actions: list[str] = []
    refusal_policy: str = ""
    rules: list[str] = []
    ground_truth: list[str] = Field([], description="Paths to ground-truth documents, relative to the profile file")
    ground_truth_text: str = ""
    pii_allow: list[str] = Field([], description="PII the agent may legitimately disclose (e.g. the caller's own email)")

    def summary(self) -> str:
        parts = [f"Purpose: {self.purpose}"]
        if self.allowed_tools:
            parts.append("Allowed tools: " + ", ".join(self.allowed_tools))
        if self.forbidden_actions:
            parts.append("Forbidden actions:\n" + "\n".join(f"- {a}" for a in self.forbidden_actions))
        if self.refusal_policy:
            parts.append(f"Refusal policy: {self.refusal_policy}")
        if self.rules:
            parts.append("Rules:\n" + "\n".join(f"- {r}" for r in self.rules))
        return "\n".join(parts)


# --- Probe --------------------------------------------------------------------------------------


class ContractExpect(BaseModel):
    json_schema: dict[str, Any] | None = None
    max_chars: int | None = None
    banned_strings: list[str] = []
    must_contain_any: list[str] = Field([], description="Deterministic: reply must contain at least one of these")
    pii_check: bool = True


class ToolExpect(BaseModel):
    required: list[str] = []
    forbidden: list[str] = []
    args: dict[str, dict[str, str]] = Field({}, description="tool -> arg -> expected substring")


class Expectation(BaseModel):
    outcome: str = Field(description="Plain-language expected outcome, shown to the judge and in alerts")
    facts: list[str] = Field([], description="Ground-truth facts this probe tests, quoted to the judge")
    contract: ContractExpect = ContractExpect()
    tools: ToolExpect = ToolExpect()
    latency_ms: int | None = None
    judge: bool = True


class Probe(BaseModel):
    id: str
    cls: ProbeClass = Field(alias="class")
    severity: Literal["P0", "P1", "P2"] = "P1"
    turns: list[str] = Field(min_length=1, description="Scripted user turns (the simulator replaces this in Phase 3)")
    expect: Expectation
    request_overrides: dict[str, Any] = Field({}, description="Merged into the json_post body for this probe")
    tags: list[str] = []

    model_config = {"populate_by_name": True}

    @property
    def axis(self) -> Axis:
        return PROBE_AXIS[self.cls]

    def content_hash(self) -> str:
        canon = json.dumps(self.model_dump(mode="json", by_alias=True), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canon.encode()).hexdigest()[:16]


# --- Suite --------------------------------------------------------------------------------------


class Suite(BaseModel):
    id: str
    target: Target
    profile: AgentProfile
    probes: list[Probe]
    interval_minutes: int = 15
    runs_per_probe: int = 1
    base_dir: Path

    def probe(self, probe_id: str) -> Probe:
        for p in self.probes:
            if p.id == probe_id:
                return p
        raise KeyError(probe_id)


def load_suite(path: Path) -> Suite:
    """Load a suite file; `target`, `profile` and `probes` may be inline or paths relative to the suite file."""
    path = path.resolve()
    base = path.parent
    raw = yaml.safe_load(path.read_text())

    def resolve(value):
        if isinstance(value, str):
            return yaml.safe_load((base / value).read_text())
        return value

    target = Target.model_validate(resolve(raw["target"]))
    profile = AgentProfile.model_validate(resolve(raw["profile"]))
    profile_dir = (base / raw["profile"]).parent if isinstance(raw["profile"], str) else base
    if profile.ground_truth and not profile.ground_truth_text:
        profile.ground_truth_text = "\n\n".join((profile_dir / p).read_text() for p in profile.ground_truth)

    probe_items: list = []
    for entry in raw["probes"]:
        loaded = resolve(entry)
        probe_items.extend(loaded if isinstance(loaded, list) else [loaded])
    probes = [Probe.model_validate(p) for p in probe_items]
    ids = [p.id for p in probes]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate probe ids in {path}")

    return Suite(
        id=raw["id"],
        target=target,
        profile=profile,
        probes=probes,
        interval_minutes=raw.get("interval_minutes", 15),
        runs_per_probe=raw.get("runs_per_probe", 1),
        base_dir=base,
    )


# --- Run (in-memory) ----------------------------------------------------------------------------


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] | str = {}


class Turn(BaseModel):
    role: Literal["user", "agent"]
    content: str
    latency_ms: int | None = None
    tool_calls: list[ToolCall] = []
    status_code: int | None = None
    raw: Any = None


class TransportError(BaseModel):
    kind: Literal["timeout", "connection", "http_status", "empty", "malformed"]
    detail: str
    status_code: int | None = None


class RunTrace(BaseModel):
    """What the runner observed: the transcript plus anything the adapter could see."""

    transcript: list[Turn] = []
    transport_error: TransportError | None = None
    retried_errors: list[TransportError] = Field([], description="Transient errors that made the runner start the probe over")
    tools_visible: bool = False

    @property
    def agent_turns(self) -> list[Turn]:
        return [t for t in self.transcript if t.role == "agent"]

    @property
    def final_reply(self) -> str:
        turns = self.agent_turns
        return turns[-1].content if turns else ""

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [c for t in self.agent_turns for c in t.tool_calls]

    @property
    def total_latency_ms(self) -> int:
        return sum(t.latency_ms or 0 for t in self.agent_turns)

    def render(self) -> str:
        lines = []
        for t in self.transcript:
            # Tool calls happen while the agent works on its reply, so they are rendered before the reply text.
            for c in t.tool_calls:
                args = json.dumps(c.arguments) if isinstance(c.arguments, dict) else c.arguments
                lines.append(f"[AGENT TOOL CALL, before replying] {c.name}({args})")
            who = "USER" if t.role == "user" else "AGENT"
            lines.append(f"[{who}] {t.content}")
        return "\n".join(lines)

    def response_hash(self) -> str:
        return hashlib.sha256(self.render().encode()).hexdigest()[:16]
