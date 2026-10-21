"""Executes one probe against a target through its adapter and records what was observed."""

import asyncio

from agentmon.adapters import Adapter, AdapterError, Conversation
from agentmon.models import Probe, RunTrace, TransportError, Turn

RETRY_BACKOFF_S = 1.0


def is_transient(error: TransportError) -> bool:
    """Worth retrying: the request may succeed unchanged. Malformed or empty replies are the agent's answer, not noise."""
    if error.kind in ("timeout", "connection"):
        return True
    return error.kind == "http_status" and (error.status_code == 429 or (error.status_code or 0) >= 500)


async def _converse(probe: Probe, adapter: Adapter, run_id: str) -> RunTrace:
    trace = RunTrace(tools_visible=adapter.tools_visible)
    conv = Conversation(run_id=run_id)
    for user_turn in probe.turns:
        trace.transcript.append(Turn(role="user", content=user_turn))
        try:
            trace.transcript.append(await adapter.send(conv, user_turn, probe.request_overrides))
        except AdapterError as e:
            trace.transport_error = e.error
            break
    return trace


async def execute_probe(probe: Probe, adapter: Adapter, run_id: str, retries: int = 0) -> RunTrace:
    """Runs the probe; on a transient transport error starts it over in a new session, up to `retries` times.

    The whole conversation restarts because a failed turn may already have reached the agent's server-side session.
    """
    retried: list[TransportError] = []
    while True:
        trace = await _converse(probe, adapter, run_id)
        err = trace.transport_error
        if err is None or not is_transient(err) or len(retried) >= retries:
            trace.retried_errors = retried
            return trace
        retried.append(err)
        await asyncio.sleep(RETRY_BACKOFF_S * 2 ** (len(retried) - 1))
