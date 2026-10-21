import asyncio
import json
import socket

import httpx
import pytest
import respx
from aiosmtpd.controller import Controller

import agentmon.adapters.webhook as webhook
from agentmon.adapters import AdapterError, Conversation, build_adapter
from agentmon.engine.runner import execute_probe
from agentmon.incidents import Notification, notify
from agentmon.models import AdapterSpec, Probe
from agentmon.settings import Settings


def sse_body(*events: dict | str) -> bytes:
    return "".join(f"data: {e if isinstance(e, str) else json.dumps(e)}\n\n" for e in events).encode()


def stream_response(*events, status=200) -> httpx.Response:
    return httpx.Response(status, headers={"content-type": "text/event-stream"}, content=sse_body(*events))


# --- sse ------------------------------------------------------------------------------------------


@respx.mock
async def test_sse_openai_preset_joins_deltas():
    route = respx.post("http://agent.test/v1/chat/completions").mock(
        return_value=stream_response(
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"content": "Ships in "}}]},
            {"choices": [{"delta": {"content": "3 days"}}]},
            "[DONE]",
        )
    )
    spec = AdapterSpec(kind="sse", url="http://agent.test/v1/chat/completions", model="m")
    async with httpx.AsyncClient() as client:
        adapter = build_adapter(spec, client, env={})
        conv = Conversation(run_id="r")
        turn = await adapter.send(conv, "when?")
    assert turn.content == "Ships in 3 days"
    assert json.loads(route.calls[0].request.content) == {"messages": [{"role": "user", "content": "when?"}], "model": "m", "stream": True}
    assert conv.history[-1] == {"role": "assistant", "content": "Ships in 3 days"}


@respx.mock
async def test_sse_custom_mapping_with_tool_events():
    respx.post("http://agent.test/chat/stream").mock(
        return_value=stream_response({"tool": {"name": "order_lookup", "arguments": {"order_id": "A1"}}}, {"text": "Shipped"}, "[DONE]")
    )
    spec = AdapterSpec(
        kind="sse",
        url="http://agent.test/chat/stream",
        body={"session_id": "{{ session_id }}", "message": "{{ user_turn }}"},
        stream={"delta": "$.text", "tool_calls": "$.tool"},
    )
    async with httpx.AsyncClient() as client:
        adapter = build_adapter(spec, client, env={})
        assert adapter.tools_visible
        turn = await adapter.send(Conversation(run_id="r"), "where is A1?")
    assert turn.content == "Shipped"
    assert [(c.name, c.arguments) for c in turn.tool_calls] == [("order_lookup", {"order_id": "A1"})]


@pytest.mark.parametrize(
    "response,kind",
    [
        (stream_response({"choices": [{"delta": {"content": "half a rep"}}]}), "malformed"),  # no [DONE]
        (stream_response("not json", "[DONE]"), "malformed"),
        (httpx.Response(200, json={"choices": []}), "malformed"),  # not an event stream
        (stream_response({"choices": [{"delta": {}}]}, "[DONE]"), "empty"),
        (httpx.Response(503, text="overloaded"), "http_status"),
    ],
)
@respx.mock
async def test_sse_transport_errors(response, kind):
    respx.post("http://agent.test/v1/chat/completions").mock(return_value=response)
    async with httpx.AsyncClient() as client:
        adapter = build_adapter(AdapterSpec(kind="sse", url="http://agent.test/v1/chat/completions"), client, env={})
        with pytest.raises(AdapterError) as e:
            await adapter.send(Conversation(run_id="r"), "hi")
    assert e.value.error.kind == kind


# --- webhook --------------------------------------------------------------------------------------


@pytest.fixture
def receiver_on_free_port(monkeypatch):
    monkeypatch.setattr(webhook, "get_settings", lambda: Settings(callback_host="127.0.0.1", callback_port=0))


def webhook_spec(**kw) -> AdapterSpec:
    base = {
        "kind": "webhook",
        "url": "http://agent.test/chat/async",
        "body": {"session_id": "{{ session_id }}", "message": "{{ user_turn }}", "callback_url": "{{ callback_url }}", "correlation_id": "{{ correlation_id }}"},
        "response": {"reply": "$.output.text", "tool_calls": "$.output.tools"},
        "timeout_s": 5,
    }
    return AdapterSpec.model_validate(base | kw)


def fake_async_agent(reply_to: str, payload_for) -> list[asyncio.Task]:
    """Mock agent: answers 202 at once, then POSTs the reply to the callback like a real webhook bot."""
    tasks = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)

        async def later():
            await asyncio.sleep(0.05)
            async with httpx.AsyncClient() as c:
                await c.post(body["callback_url"] if reply_to == "callback_url" else reply_to.format(**body), json=payload_for(body))

        tasks.append(asyncio.create_task(later()))
        return httpx.Response(202, json={"accepted": True})

    respx.post("http://agent.test/chat/async").mock(side_effect=handler)
    respx.route(host="127.0.0.1").pass_through()  # the callback really goes to the receiver
    return tasks


@respx.mock
async def test_webhook_round_trip_via_per_turn_callback_url(receiver_on_free_port):
    fake_async_agent("callback_url", lambda b: {"output": {"text": f"echo {b['message']}", "tools": [{"name": "t", "arguments": {}}]}})
    async with httpx.AsyncClient() as client, build_adapter(webhook_spec(), client, env={}) as adapter:
        conv = Conversation(run_id="run1")
        first = await adapter.send(conv, "one")
        second = await adapter.send(conv, "two")
    assert (first.content, second.content) == ("echo one", "echo two")
    assert first.tool_calls[0].name == "t" and first.status_code == 202
    assert webhook._receiver is None  # stopped once the last user released it


@respx.mock
async def test_webhook_fixed_url_correlates_by_echoed_session(receiver_on_free_port):
    def payload(b):
        return {"conversation": {"id": b["session_id"]}, "output": {"text": "ok"}}

    async with httpx.AsyncClient() as client, build_adapter(webhook_spec(callback={"correlate": "$.conversation.id"}), client, env={}) as adapter:
        fake_async_agent(f"http://127.0.0.1:{adapter.receiver.port}/callbacks", payload)
        turn = await adapter.send(Conversation(run_id="run2"), "hi")
    assert turn.content == "ok"


@respx.mock
async def test_webhook_times_out_without_callback(receiver_on_free_port):
    respx.post("http://agent.test/chat/async").mock(return_value=httpx.Response(202))
    async with httpx.AsyncClient() as client, build_adapter(webhook_spec(timeout_s=0.2), client, env={}) as adapter:
        with pytest.raises(AdapterError) as e:
            await adapter.send(Conversation(run_id="r"), "hi")
        assert adapter.receiver._by_key == {}  # nothing left waiting
    assert e.value.error.kind == "timeout"


# --- runner retries -------------------------------------------------------------------------------


def probe(turns=("q",)) -> Probe:
    return Probe.model_validate({"id": "p", "class": "fact_recall", "turns": list(turns), "expect": {"outcome": "o"}})


def json_post() -> AdapterSpec:
    return AdapterSpec(
        kind="json_post",
        url="http://agent.test/chat",
        body={"session_id": "{{ session_id }}", "message": "{{ user_turn }}"},
        response={"reply": "$.text"},
    )


@respx.mock
async def test_runner_retries_transient_errors_in_a_fresh_session(monkeypatch):
    monkeypatch.setattr("agentmon.engine.runner.RETRY_BACKOFF_S", 0)
    route = respx.post("http://agent.test/chat").mock(
        side_effect=[httpx.Response(200, json={"text": "a"}), httpx.Response(502, text="bad gateway"), httpx.Response(200, json={"text": "a"}), httpx.Response(200, json={"text": "b"})]
    )
    async with httpx.AsyncClient() as client:
        trace = await execute_probe(probe(("q1", "q2")), build_adapter(json_post(), client, env={}), "r", retries=2)
    assert trace.transport_error is None and [t.content for t in trace.agent_turns] == ["a", "b"]
    assert [e.status_code for e in trace.retried_errors] == [502]
    sessions = [json.loads(c.request.content)["session_id"] for c in route.calls]
    assert sessions[0] == sessions[1] != sessions[2] == sessions[3]  # the conversation restarted


@pytest.mark.parametrize("response", [httpx.Response(400, text="bad request"), httpx.Response(200, json={"text": ""})])
@respx.mock
async def test_runner_does_not_retry_agent_answers(response):
    route = respx.post("http://agent.test/chat").mock(return_value=response)
    async with httpx.AsyncClient() as client:
        trace = await execute_probe(probe(), build_adapter(json_post(), client, env={}), "r", retries=2)
    assert trace.transport_error is not None and trace.retried_errors == [] and route.call_count == 1


@respx.mock
async def test_runner_gives_up_after_retries(monkeypatch):
    monkeypatch.setattr("agentmon.engine.runner.RETRY_BACKOFF_S", 0)
    route = respx.post("http://agent.test/chat").mock(side_effect=httpx.ConnectError("refused"))
    async with httpx.AsyncClient() as client:
        trace = await execute_probe(probe(), build_adapter(json_post(), client, env={}), "r", retries=2)
    assert trace.transport_error.kind == "connection" and len(trace.retried_errors) == 2 and route.call_count == 3


# --- email ----------------------------------------------------------------------------------------


class Inbox:
    def __init__(self):
        self.messages = []

    async def handle_DATA(self, server, session, envelope):
        self.messages.append(envelope)
        return "250 OK"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def test_email_alert_carries_transcript_and_evidence(monkeypatch):
    inbox = Inbox()
    port = free_port()
    controller = Controller(inbox, hostname="127.0.0.1", port=port)
    controller.start()
    try:
        settings = Settings(smtp_host="127.0.0.1", smtp_port=port, smtp_security="none", alert_email_to="ops@example.com, oncall@example.com")
        monkeypatch.setattr("agentmon.incidents.get_settings", lambda: settings)
        run = {
            "run_id": "abcdef123456",
            "at": "2026-10-21T06:00:00",
            "transcript": [{"role": "user", "content": "Refund window?"}, {"role": "agent", "content": "14 days.", "tool_calls": []}],
            "judge": {"evidence": ["14 days."]},
        }
        await notify([Notification(kind="opened", incident_id="i1", severity="P0", probe_key="fact", text="[P0] Acme: `fact` failed\nExpected: 30 days", payload={"failing": run, "last_passing": None})])
    finally:
        controller.stop()
    assert len(inbox.messages) == 1
    env = inbox.messages[0]
    assert env.rcpt_tos == ["ops@example.com", "oncall@example.com"]
    body = env.content.decode()
    assert "Subject: [P0] Acme: fact failed" in body and "AGENT: 14 days." in body and "> 14 days." in body and "agentmon show abcdef12" in body
