import json

from agentmon.judge import validate_evidence
from agentmon.judge.judge import parse_output
from agentmon.models import RunTrace, ToolCall, Turn

TRANSCRIPT = "[USER] How long can I return items?\n[AGENT] You can return items within   30 days of delivery."


def out(**kw) -> str:
    base = {"verdict": "pass", "confidence": 0.9, "failure_class": "none", "reasoning": "r", "evidence": ["within 30 days of delivery"]}
    return json.dumps(base | kw)


def test_evidence_must_come_from_transcript():
    assert validate_evidence(["within 30 days of delivery"], TRANSCRIPT) is None
    assert validate_evidence(["You can return items within 30 days"], TRANSCRIPT) is None  # whitespace normalised
    assert "not found" in validate_evidence(["refund window is 30 days per policy"], TRANSCRIPT)
    assert validate_evidence([], TRANSCRIPT) == "no evidence quotes"
    assert "too short" in validate_evidence(["30"], TRANSCRIPT)


def test_evidence_ignores_markdown_emphasis():
    assert validate_evidence(["Status: Shipped via DHL"], "[AGENT] - **Status:** Shipped via `DHL`") is None


def test_parse_accepts_valid_and_code_fences():
    parsed, err = parse_output("```json\n" + out() + "\n```", TRANSCRIPT)
    assert err is None and parsed.verdict == "pass"


def test_parse_rejects_bad_outputs():
    assert parse_output("not json", TRANSCRIPT)[1].startswith("malformed")
    assert "bad verdict" in parse_output(out(verdict="maybe"), TRANSCRIPT)[1]
    assert "unknown failure_class" in parse_output(out(verdict="fail", failure_class="vibes"), TRANSCRIPT)[1]
    assert "range" in parse_output(out(confidence=3), TRANSCRIPT)[1]
    assert "not found" in parse_output(out(evidence=["invented quote here"]), TRANSCRIPT)[1]


def test_render_puts_tool_calls_before_reply():
    t = RunTrace(
        transcript=[
            Turn(role="user", content="balance?"),
            Turn(role="agent", content="$42.17", tool_calls=[ToolCall(name="get_balance", arguments={})]),
        ]
    )
    lines = t.render().splitlines()
    assert lines[1].startswith("[AGENT TOOL CALL") and lines[2] == "[AGENT] $42.17"
