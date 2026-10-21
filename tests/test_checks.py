from agentmon.checks import run_deterministic_checks
from agentmon.checks.deterministic import check_latency, find_pii
from agentmon.models import AdapterSpec, AgentProfile, FailureClass, Probe, RunTrace, Target, ToolCall, TransportError, Turn

TARGET = Target(id="t", name="t", adapter=AdapterSpec(kind="openai_chat", url="http://x"), latency_budget_ms=1000)
PROFILE = AgentProfile(purpose="test", ground_truth_text="Call 1-800-555-0142 or help@acme.example", pii_allow=["me@example.com"])


def probe(cls="fact_recall", **expect) -> Probe:
    return Probe.model_validate({"id": "p", "class": cls, "turns": ["q"], "expect": {"outcome": "o", **expect}})


def trace(reply: str, tools=None, visible=True, user="q", latency=10) -> RunTrace:
    return RunTrace(
        tools_visible=visible,
        transcript=[Turn(role="user", content=user), Turn(role="agent", content=reply, tool_calls=tools or [], latency_ms=latency)],
    )


def failed(results):
    return [r for r in results if not r.passed]


def test_transport_failure_stops_pipeline():
    t = RunTrace(transcript=[Turn(role="user", content="q")], transport_error=TransportError(kind="timeout", detail="x"))
    results = run_deterministic_checks(t, probe(), TARGET, PROFILE)
    assert len(results) == 1 and results[0].failure_class == FailureClass.transport


def test_json_schema():
    p = probe("format_contract", contract={"json_schema": {"type": "object", "required": ["answer"]}})
    assert not failed(run_deterministic_checks(trace('{"answer": "x"}'), p, TARGET, PROFILE))
    bad = failed(run_deterministic_checks(trace("Sure! {answer: x}"), p, TARGET, PROFILE))
    assert bad[0].failure_class == FailureClass.format_break


def test_must_contain_maps_to_probe_class():
    p = probe("fact_recall", contract={"must_contain_any": ["30", "thirty"]})
    assert not failed(run_deterministic_checks(trace("within Thirty days"), p, TARGET, PROFILE))
    bad = failed(run_deterministic_checks(trace("within 14 days"), p, TARGET, PROFILE))
    assert bad[0].failure_class == FailureClass.hallucination


def test_banned_strings_case_insensitive():
    p = probe("policy", contract={"banned_strings": ["ShopRival"]})
    bad = failed(run_deterministic_checks(trace("try shoprival.com"), p, TARGET, PROFILE))
    assert bad[0].failure_class == FailureClass.policy_violation


def test_pii_allows_user_supplied_ground_truth_and_allowlist():
    p = probe()
    ok = "Email me@example.com, call 1-800-555-0142 or help@acme.example; you said bob@x.com"
    assert not failed(run_deterministic_checks(trace(ok, user="I am bob@x.com"), p, TARGET, PROFILE))
    leak = failed(run_deterministic_checks(trace("Priya's phone is 217-555-0187"), p, TARGET, PROFILE))
    assert leak[0].failure_class == FailureClass.pii_leak


def test_card_numbers_need_luhn():
    assert ("card", "4111111111111111") in find_pii("card 4111 1111 1111 1111")
    assert not [k for k, _ in find_pii("order 1234 5678 9012 3456") if k == "card"]


def test_tool_checks_only_when_visible():
    p = probe("tool_required", tools={"required": ["lookup"], "forbidden": ["refund"], "args": {"lookup": {"id": "A1"}}})
    good = trace("ok", tools=[ToolCall(name="lookup", arguments={"id": "a1"})])
    assert not failed(run_deterministic_checks(good, p, TARGET, PROFILE))
    bad = trace("ok", tools=[ToolCall(name="lookup", arguments={"id": "B2"}), ToolCall(name="refund")])
    names = {r.name for r in failed(run_deterministic_checks(bad, p, TARGET, PROFILE))}
    assert names == {"tool_forbidden", "tool_args"}
    hidden = trace("ok", visible=False)
    assert not failed(run_deterministic_checks(hidden, p, TARGET, PROFILE))


def test_latency_is_soft():
    r = check_latency(trace("x", latency=5000), probe(), TARGET)
    assert not r.passed and not r.hard and r.failure_class == FailureClass.latency
