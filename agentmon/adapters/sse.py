"""sse: the agent streams its reply as server-sent events; fragments are joined into one turn."""

import asyncio
import json
import time

import httpx
from httpx_sse import SSEError, aconnect_sse
from jsonpath_ng.ext import parse as jsonpath

from agentmon.adapters.base import Adapter, AdapterError, Conversation, _ms, extract_tool_calls
from agentmon.models import StreamMapping, ToolCall, TransportError, Turn


class SSEAdapter(Adapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stream = self.spec.stream or StreamMapping()

    @property
    def tools_visible(self) -> bool:
        return bool(self.stream.tool_calls)

    def default_body(self, history: list[dict]) -> dict:
        return super().default_body(history) | {"stream": True}

    async def _exchange(self, conv: Conversation, user_turn: str, overrides: dict | None) -> Turn:
        url, headers, body = self.build_request(conv, user_turn, overrides)
        s = self.stream
        fragments: list[str] = []
        tools: list[ToolCall] = []
        events: list[dict] = []
        finished = False
        status: int | None = None
        started = time.perf_counter()

        def fail(kind: str, detail: str, code: int | None = None) -> AdapterError:
            return AdapterError(TransportError(kind=kind, detail=detail, status_code=code), _ms(started))

        try:
            # The read timeout only bounds the gap between events; this bounds the whole reply.
            async with asyncio.timeout(self.spec.timeout_s):
                async with aconnect_sse(self.client, "POST", url, json=body, headers=headers, timeout=self.spec.timeout_s) as source:
                    status = source.response.status_code
                    if status >= 400:
                        await source.response.aread()
                        raise fail("http_status", source.response.text[:500], status)
                    async for ev in source.aiter_sse():
                        if s.done is not None and ev.data.strip() == s.done:
                            finished = True
                            break
                        events.append({"event": ev.event, "data": ev.data})
                        if s.event and ev.event != s.event:
                            continue
                        if s.delta is None:
                            fragments.append(ev.data)
                            continue
                        try:
                            data = json.loads(ev.data)
                        except json.JSONDecodeError:
                            raise fail("malformed", f"event data is not JSON: {ev.data[:200]}")
                        fragments += [m.value for m in jsonpath(s.delta).find(data) if isinstance(m.value, str)]
                        tools += extract_tool_calls(data, s.tool_calls, s.tool_name, s.tool_args)
        except TimeoutError:
            raise fail("timeout", f"stream not finished after {self.spec.timeout_s:g} s")
        except httpx.TimeoutException as e:
            raise fail("timeout", str(e) or "timed out")
        except SSEError as e:
            raise fail("malformed", f"not an event stream: {e}", status)
        except httpx.HTTPError as e:
            raise fail("connection", f"{type(e).__name__}: {e}")

        if s.done is not None and not finished:
            raise fail("malformed", f"stream ended before {s.done!r} after {len(events)} events")
        reply = "".join(fragments)
        if not reply.strip():
            raise fail("empty", f"no reply text in {len(events)} events")
        return Turn(role="agent", content=reply, latency_ms=_ms(started), tool_calls=tools, status_code=status, raw={"events": events})
