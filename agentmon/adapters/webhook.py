"""webhook: the agent acknowledges a turn, then POSTs its reply later to a callback endpoint the monitor runs.

Each turn gets a unique correlation id (it contains the run id). The callback URL sent to the agent embeds it, so
nothing has to be echoed back. Agents that can only post to one fixed URL use /callbacks and echo the correlation_id
or session_id in the body; `callback.correlate` says where.
"""

import asyncio
import secrets
import socket
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx
import uvicorn
from jsonpath_ng.ext import parse as jsonpath
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from agentmon.adapters.base import Adapter, AdapterError, Conversation, _ms
from agentmon.models import TransportError, Turn
from agentmon.settings import get_settings


@dataclass
class _Pending:
    future: asyncio.Future
    keys: set[str]
    correlate: str | None


class _EmbeddedServer(uvicorn.Server):
    @contextmanager
    def capture_signals(self):  # the host process keeps its own Ctrl-C handling
        yield


@dataclass
class CallbackReceiver:
    host: str
    port: int  # 0 = any free port
    public_url: str | None = None  # default http://host:port
    _by_key: dict[str, _Pending] = field(default_factory=dict)
    _server: _EmbeddedServer | None = None
    _task: asyncio.Task | None = None

    def url(self, token: str) -> str:
        base = self.public_url or f"http://{self.host}:{self.port}"
        return f"{base.rstrip('/')}/callbacks/{token}"

    def expect(self, token: str, session_id: str, correlate: str | None) -> _Pending:
        pending = _Pending(asyncio.get_running_loop().create_future(), {token, session_id}, correlate)
        for key in pending.keys:
            self._by_key[key] = pending
        return pending

    def forget(self, pending: _Pending) -> None:
        for key in pending.keys:
            if self._by_key.get(key) is pending:
                del self._by_key[key]

    def _match(self, body: Any) -> _Pending | None:
        for path in {p.correlate for p in self._by_key.values() if p.correlate}:
            for m in jsonpath(path).find(body):
                if (hit := self._by_key.get(str(m.value))) is not None:
                    return hit
        return None

    async def _receive(self, request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except ValueError:
            return JSONResponse({"error": "body must be JSON"}, status_code=400)
        token = request.path_params.get("token")
        pending = self._by_key.get(token) if token else self._match(body)
        if pending is None or pending.future.done():
            return JSONResponse({"error": "unknown or expired callback"}, status_code=404)
        pending.future.set_result(body)
        return JSONResponse({"ok": True})

    async def start(self) -> None:
        # Bind here rather than in uvicorn, which calls sys.exit() when the port is taken.
        sock = socket.socket(socket.AF_INET6 if ":" in self.host else socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((self.host, self.port))
        except OSError as e:
            sock.close()
            raise RuntimeError(f"callback receiver cannot listen on {self.host}:{self.port}: {e}") from e
        self.port = sock.getsockname()[1]
        routes = [Route("/callbacks/{token}", self._receive, methods=["POST"]), Route("/callbacks", self._receive, methods=["POST"])]
        config = uvicorn.Config(Starlette(routes=routes), log_level="warning", lifespan="off")
        self._server = _EmbeddedServer(config)
        self._task = asyncio.create_task(self._server.serve(sockets=[sock]))
        while not self._server.started:
            if self._task.done():
                await self._task
                raise RuntimeError(f"callback receiver stopped during startup on {self.host}:{self.port}")
            await asyncio.sleep(0.02)

    async def stop(self) -> None:
        if self._server and self._task:
            self._server.should_exit = True
            await self._task


_receiver: CallbackReceiver | None = None
_users = 0
_lock: tuple[asyncio.AbstractEventLoop, asyncio.Lock] | None = None


def _loop_lock() -> asyncio.Lock:
    global _lock
    loop = asyncio.get_running_loop()
    if _lock is None or _lock[0] is not loop:  # each `asyncio.run` gets its own loop
        _lock = (loop, asyncio.Lock())
    return _lock[1]


async def acquire_receiver() -> CallbackReceiver:
    """One receiver per process, shared by every webhook target; it runs while any of them is in use."""
    global _receiver, _users
    async with _loop_lock():
        if _receiver is None:
            s = get_settings()
            receiver = CallbackReceiver(s.callback_host, s.callback_port, s.callback_public_url)
            await receiver.start()
            _receiver = receiver
        _users += 1
        return _receiver


async def release_receiver() -> None:
    global _receiver, _users
    async with _loop_lock():
        _users -= 1
        if _users == 0 and _receiver is not None:
            await _receiver.stop()
            _receiver = None


class WebhookAdapter(Adapter):
    receiver: CallbackReceiver | None = None

    async def __aenter__(self) -> "WebhookAdapter":
        self.receiver = await acquire_receiver()
        return self

    async def __aexit__(self, *exc) -> None:
        await release_receiver()
        self.receiver = None

    async def _exchange(self, conv: Conversation, user_turn: str, overrides: dict | None) -> Turn:
        if self.receiver is None:
            raise RuntimeError("webhook adapter must be used as `async with adapter:` so its callback receiver runs")
        token = f"{conv.run_id}-{secrets.token_hex(6)}"
        correlate = self.spec.callback.correlate if self.spec.callback else None
        pending = self.receiver.expect(token, conv.session_id, correlate)
        extra = {"callback_url": self.receiver.url(token), "correlation_id": token}
        started = time.perf_counter()
        try:
            url, headers, body = self.build_request(conv, user_turn, overrides, extra)
            async with asyncio.timeout(self.spec.timeout_s):
                ack = await self._post(url, headers, body, started)
                data = await pending.future
        except TimeoutError:
            raise AdapterError(TransportError(kind="timeout", detail=f"no callback within {self.spec.timeout_s:g} s"), _ms(started))
        finally:
            self.receiver.forget(pending)
        return self.turn_from_json(data, ack.status_code, _ms(started))
