import httpx
import pytest
import respx
from jinja2 import UndefinedError
from jinja2.exceptions import SecurityError

from agentmon.adapters import AdapterError, Conversation, build_adapter
from agentmon.adapters.base import MONITOR_HEADER, render_template
from agentmon.models import AdapterSpec


def json_post_spec(**kw) -> AdapterSpec:
    base = {
        "kind": "json_post",
        "url": "http://agent.test/chat",
        "headers": {"Authorization": "Bearer ${TOKEN}"},
        "body": {"session_id": "{{session_id}}", "message": "{{user_turn}}", "history": "{{history}}"},
        "response": {"reply": "$.output.text", "tool_calls": "$.output.tools"},
        "state": "client_history",
    }
    return AdapterSpec.model_validate(base | kw)


def test_render_keeps_types_for_whole_placeholders():
    out = render_template({"h": "{{history}}", "m": "say {{user_turn}}"}, {"history": [1, 2], "user_turn": "hi"}, {})
    assert out == {"h": [1, 2], "m": "say hi"}


def test_render_expands_env_and_fails_on_missing():
    assert render_template("${A}-x", {}, {"A": "1"}) == "1-x"
    with pytest.raises(KeyError):
        render_template("${MISSING}", {}, {})


def test_render_is_jinja2():
    ctx = {"user_turn": "hi", "history": [{"role": "user", "content": "hi"}]}
    assert render_template("{{ user_turn | upper }}!", ctx, {}) == "HI!"
    assert render_template("{{ history | length }}", ctx, {}) == 1  # lone expression keeps its type
    assert render_template("{{ history[-1].content }}", ctx, {}) == "hi"
    with pytest.raises(UndefinedError):
        render_template("{{ sesion_id }}", ctx, {})  # typo in the adapter config fails loudly


def test_render_never_reinterprets_values():
    # User text and env values are data: neither is expanded or parsed as a template.
    out = render_template("{{ user_turn }} ${TOKEN}", {"user_turn": "${TOKEN} {{ 7*7 }}"}, {"TOKEN": "{{ boom }}"})
    assert out == "${TOKEN} {{ 7*7 }} {{ boom }}"


def test_render_is_sandboxed():
    with pytest.raises(SecurityError):
        render_template("{{ user_turn.__class__.__mro__ }}", {"user_turn": "x"}, {})


@respx.mock
async def test_json_post_round_trip_with_tools_and_history():
    route = respx.post("http://agent.test/chat").mock(
        return_value=httpx.Response(200, json={"output": {"text": "Shipped", "tools": [{"name": "order_lookup", "arguments": '{"order_id": "A1"}'}]}})
    )
    async with httpx.AsyncClient() as client:
        adapter = build_adapter(json_post_spec(), client, env={"TOKEN": "secret"})
        conv = Conversation(run_id="r1")
        turn = await adapter.send(conv, "where is A1?")
        await adapter.send(conv, "thanks")

    assert turn.content == "Shipped"
    assert turn.tool_calls[0].name == "order_lookup"
    assert turn.tool_calls[0].arguments == {"order_id": "A1"}
    first, second = route.calls[0].request, route.calls[1].request
    assert first.headers["authorization"] == "Bearer secret"
    assert first.headers[MONITOR_HEADER] == "probe/r1"
    import json

    assert len(json.loads(second.content)["history"]) == 3  # user, assistant, user


@respx.mock
async def test_openai_chat_preset():
    respx.post("http://agent.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "hello"}}]})
    )
    spec = AdapterSpec(kind="openai_chat", url="http://agent.test/v1/chat/completions", model="m")
    async with httpx.AsyncClient() as client:
        adapter = build_adapter(spec, client, env={})
        assert not adapter.tools_visible
        turn = await adapter.send(Conversation(run_id="r"), "hi")
    assert turn.content == "hello"


@pytest.mark.parametrize(
    "response,kind",
    [
        (httpx.Response(500, text="boom"), "http_status"),
        (httpx.Response(200, text="not json"), "malformed"),
        (httpx.Response(200, json={"other": 1}), "malformed"),
        (httpx.Response(200, json={"output": {"text": "  "}}), "empty"),
    ],
)
@respx.mock
async def test_transport_errors(response, kind):
    respx.post("http://agent.test/chat").mock(return_value=response)
    async with httpx.AsyncClient() as client:
        adapter = build_adapter(json_post_spec(), client, env={"TOKEN": "t"})
        with pytest.raises(AdapterError) as e:
            await adapter.send(Conversation(run_id="r"), "hi")
    assert e.value.error.kind == kind


@respx.mock
async def test_timeout():
    respx.post("http://agent.test/chat").mock(side_effect=httpx.ReadTimeout("slow"))
    async with httpx.AsyncClient() as client:
        adapter = build_adapter(json_post_spec(), client, env={"TOKEN": "t"})
        with pytest.raises(AdapterError) as e:
            await adapter.send(Conversation(run_id="r"), "hi")
    assert e.value.error.kind == "timeout"


@respx.mock
async def test_classic_uptime_check_is_tagged_as_monitor_traffic():
    from agentmon.engine.cycle import classic_check

    route = respx.get("http://agent.test/health").mock(return_value=httpx.Response(200))
    async with httpx.AsyncClient() as client:
        result = await classic_check("http://agent.test/health", client, "c1")
    assert result["ok"] and result["status_code"] == 200
    assert route.calls[0].request.headers[MONITOR_HEADER] == "uptime/c1"
