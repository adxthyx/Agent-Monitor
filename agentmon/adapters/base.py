"""Declarative adapters: how to send a user turn to a black-box agent and where the reply lives.

A custom adapter is config (AdapterSpec), not code. Presets differ only in defaults.
"""

import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import httpx
from jinja2 import StrictUndefined, Template, Undefined, UndefinedError
from jinja2.sandbox import SandboxedEnvironment
from jsonpath_ng.ext import parse as jsonpath

from agentmon.models import AdapterSpec, ResponseMapping, ToolCall, TransportError, Turn

MONITOR_HEADER = "X-Agent-Monitor"
_EXPRESSION = re.compile(r"\{\{\s*([^{}]+?)\s*\}\}")
_ENV = re.compile(r"\$\{(\w+)\}")
_ENV_VAR = "_env_"  # template variable that carries ${ENV} values, so they are never parsed as template source
_SANDBOX = SandboxedEnvironment(undefined=StrictUndefined, autoescape=False, keep_trailing_newline=True)

OPENAI_CHAT_RESPONSE = ResponseMapping(reply="$.choices[0].message.content")


class AdapterError(Exception):
    def __init__(self, error: TransportError, latency_ms: int | None = None):
        super().__init__(f"{error.kind}: {error.detail}")
        self.error = error
        self.latency_ms = latency_ms


@dataclass
class Conversation:
    run_id: str
    session_id: str = field(default_factory=lambda: f"am-{uuid.uuid4().hex[:12]}")
    history: list[dict] = field(default_factory=list)


def expand_env(value: str, env: dict[str, str]) -> str:
    def sub(m: re.Match) -> str:
        if m.group(1) not in env:
            raise KeyError(f"environment variable {m.group(1)} is not set")
        return env[m.group(1)]

    return _ENV.sub(sub, value)


@lru_cache(maxsize=512)
def _template(source: str) -> Template:
    return _SANDBOX.from_string(source)


@lru_cache(maxsize=512)
def _expression(source: str):
    return _SANDBOX.compile_expression(source, undefined_to_none=False)


def render_template(node: Any, ctx: dict[str, Any], env: dict[str, str]) -> Any:
    """Render Jinja2 (sandboxed) in every string. A string that is exactly one {{ expression }} keeps its value's type."""
    if isinstance(node, dict):
        return {k: render_template(v, ctx, env) for k, v in node.items()}
    if isinstance(node, list):
        return [render_template(v, ctx, env) for v in node]
    if not isinstance(node, str):
        return node
    whole = _EXPRESSION.fullmatch(node.strip())
    if whole and not _ENV.search(node):
        value = _expression(whole.group(1))(**ctx)
        if isinstance(value, Undefined):
            raise UndefinedError(f"undefined template variable in {node!r}")
        return value

    def env_ref(m: re.Match) -> str:
        if m.group(1) not in env:
            raise KeyError(f"environment variable {m.group(1)} is not set")
        return "{{ %s[%r] }}" % (_ENV_VAR, m.group(1))

    return _template(_ENV.sub(env_ref, node)).render({**ctx, _ENV_VAR: env})


def deep_merge(base: dict, extra: dict) -> dict:
    out = dict(base)
    for k, v in extra.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _dig(obj: Any, dotted: str) -> Any:
    for part in dotted.split("."):
        if not isinstance(obj, dict):
            return None
        obj = obj.get(part)
    return obj


def extract_tool_calls(data: Any, path: str | None, name_field: str = "name", args_field: str = "arguments") -> list[ToolCall]:
    if not path:
        return []
    matches = [m.value for m in jsonpath(path).find(data)]
    if len(matches) == 1 and isinstance(matches[0], list):
        matches = matches[0]
    calls = []
    for item in matches:
        if not isinstance(item, dict):
            continue
        args = _dig(item, args_field)
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                pass
        calls.append(ToolCall(name=str(_dig(item, name_field)), arguments=args if args is not None else {}))
    return calls


class Adapter:
    """json_post and openai_chat: one HTTP POST per user turn, reply in the JSON response. Subclasses override `_exchange`."""

    def __init__(self, spec: AdapterSpec, client: httpx.AsyncClient, env: dict[str, str] | None = None):
        self.spec = spec
        self.client = client
        self.env = env if env is not None else dict(os.environ)
        self.mapping = spec.response or OPENAI_CHAT_RESPONSE

    async def __aenter__(self) -> "Adapter":
        return self

    async def __aexit__(self, *exc) -> None:
        return None

    @property
    def tools_visible(self) -> bool:
        return bool(self.mapping.tool_calls)

    def default_body(self, history: list[dict]) -> dict:
        body: dict = {"messages": history}
        if self.spec.model:
            body["model"] = self.spec.model
        return body

    def build_request(
        self, conv: Conversation, user_turn: str, overrides: dict | None = None, extra: dict | None = None
    ) -> tuple[str, dict, dict]:
        history = conv.history + [{"role": "user", "content": user_turn}]
        ctx = {
            "session_id": conv.session_id,
            "run_id": conv.run_id,
            "user_turn": user_turn,
            "history": history if self.spec.state == "client_history" else [],
            **(extra or {}),
        }
        url = render_template(self.spec.url, ctx, self.env)
        headers = {k: render_template(v, ctx, self.env) for k, v in self.spec.headers.items()}
        headers[MONITOR_HEADER] = f"probe/{conv.run_id}"
        body = render_template(self.spec.body, ctx, self.env) if self.spec.body is not None else self.default_body(history)
        if overrides:
            body = deep_merge(body, overrides)
        return url, headers, body

    async def send(self, conv: Conversation, user_turn: str, overrides: dict | None = None) -> Turn:
        turn = await self._exchange(conv, user_turn, overrides)
        conv.history = conv.history + [
            {"role": "user", "content": user_turn},
            {"role": "assistant", "content": turn.content},
        ]
        return turn

    async def _exchange(self, conv: Conversation, user_turn: str, overrides: dict | None) -> Turn:
        url, headers, body = self.build_request(conv, user_turn, overrides)
        started = time.perf_counter()
        resp = await self._post(url, headers, body, started)
        try:
            data = resp.json()
        except ValueError:
            raise AdapterError(TransportError(kind="malformed", detail=f"non-JSON body: {resp.text[:200]}"), _ms(started))
        return self.turn_from_json(data, resp.status_code, _ms(started))

    async def _post(self, url: str, headers: dict, body: dict, started: float) -> httpx.Response:
        try:
            resp = await self.client.post(url, json=body, headers=headers, timeout=self.spec.timeout_s)
        except httpx.TimeoutException as e:
            raise AdapterError(TransportError(kind="timeout", detail=str(e) or "timed out"), _ms(started))
        except httpx.HTTPError as e:
            raise AdapterError(TransportError(kind="connection", detail=f"{type(e).__name__}: {e}"), _ms(started))
        if resp.status_code >= 400:
            raise AdapterError(
                TransportError(kind="http_status", detail=resp.text[:500], status_code=resp.status_code), _ms(started)
            )
        return resp

    def turn_from_json(self, data: Any, status_code: int | None, latency: int) -> Turn:
        found = jsonpath(self.mapping.reply).find(data)
        if not found:
            raise AdapterError(TransportError(kind="malformed", detail=f"reply path {self.mapping.reply} not found"), latency)
        reply = found[0].value
        if not isinstance(reply, str):
            reply = json.dumps(reply)
        if not reply.strip():
            raise AdapterError(TransportError(kind="empty", detail="empty reply"), latency)
        return Turn(
            role="agent",
            content=reply,
            latency_ms=latency,
            tool_calls=extract_tool_calls(data, self.mapping.tool_calls, self.mapping.tool_name, self.mapping.tool_args),
            status_code=status_code,
            raw=data,
        )


def _ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)

