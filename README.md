# agentmon — semantic uptime monitor for AI agents

Outside-in monitor that holds synthetic conversations with an agent endpoint, grades them (deterministic checks first, LLM judge last) and opens incidents when behaviour breaks — even when HTTP says 200. See `PLAN.md` for the full build plan.

## Run it

```bash
uv sync                                   # Python deps
docker compose up -d db                   # Postgres on :5433
uv run alembic upgrade head               # schema
uv run agentmon doctor                    # DB + every LLM slot reachable?

# demo target (needs DEEPSEEK_API_KEY; model_swap fault needs NEBIUS_API_KEY)
uv run uvicorn demo_agent.app:app --port 8100

uv run agentmon run suites/acme/suite.yaml          # one cycle
uv run agentmon run suites/acme/suite.yaml -n 3     # 3 runs per probe
uv run agentmon watch suites/acme/suite*.yaml       # APScheduler, each suite every interval_minutes; Ctrl-C / SIGTERM stops
uv run agentmon show <run-id-prefix>                # transcript, checks, judge evidence
uv run agentmon runs | incidents | cost
uv run pytest                                       # offline unit tests
```

Inject a fault into the demo agent, then run a cycle:

```bash
curl -XPOST localhost:8100/admin/faults -H 'content-type: application/json' -d '{"stale_kb": true}'
curl -XPOST localhost:8100/admin/faults/reset
```

Faults: `stale_kb`, `kb_gap_fill`, `broken_tool`, `strict_prompt`, `lax_prompt`, `policy_leak`, `drop_history`, `model_swap`, `format_break`, `slow` (+ `slow_seconds`), `gradual_decay` (+ `decay_per_hour`).

`gradual_decay` silently corrupts a growing share of the knowledge base: `floor(hours since switched on × decay_per_hour × 18)` facts, in the order listed under `decayed` in `demo_agent/kb.yaml` (canary fact last). Default 0.05/h, all facts after 20 h. `GET /admin/faults` shows `decayed_facts`. For a quick hand test use `{"gradual_decay": true, "decay_per_hour": 3}` (one more fact every ~67 s).

`agentmon doctor` exits 1 unless the DB and every slot are green. It also shows each slot's current price. Nebius prices come live from `GET /models?verbose=true`; DeepSeek has no pricing API, so its peak and off-peak rates sit in `config/slots.yaml`.

## Transports (adapters)

A target's adapter is config. Request strings are sandboxed Jinja2 templates. The variables are `session_id`, `user_turn`, `history`, `run_id`, and for `webhook` also `callback_url` and `correlation_id`. A string that is a single `{{ expression }}` keeps its type, so `"{{ history }}"` sends a list. `${ENV_VAR}` works in any string. User text and env values are never re-read as templates.

| kind | How a turn travels | Demo target |
|---|---|---|
| `json_post` | POST a JSON body; reply and tool calls are JSONPaths into the response | `suites/acme/suite.yaml` |
| `openai_chat` | OpenAI chat-completions body; reply at `choices[0].message.content` | (preset) |
| `sse` | POST, reply streamed as server-sent events. `stream.delta` / `stream.tool_calls` are JSONPaths per event, `stream.done` ends it (missing = malformed stream). Defaults match OpenAI streaming | `suite-sse.yaml`, `suite-openai-stream.yaml` |
| `webhook` | POST; the agent acks, then POSTs its reply to the monitor's callback receiver (`CALLBACK_HOST/PORT`, default 127.0.0.1:8200; `CALLBACK_PUBLIC_URL` if reached through a tunnel). Per-turn `{{ callback_url }}`, or a fixed `/callbacks` URL plus `callback.correlate` | `suite-webhook.yaml` |

Every request carries `X-Agent-Monitor: probe/<run_id>` (the classic check sends `uptime/<cycle_id>`). On a timeout, connection error, 429 or 5xx, the runner starts the probe over in a new session, up to `transport_retries` times (default 2, backoff 1 s then 2 s). Malformed or empty replies count as the agent's answer and are not retried.

## Alerts

Alerts print to the console. They also go to a Slack-compatible webhook if `ALERT_WEBHOOK_URL` is set, and by email if `SMTP_HOST` + `ALERT_EMAIL_TO` are set (`SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_SECURITY` = starttls | ssl | none). The email carries the expected outcome, the judge's reasoning and evidence, the full failing transcript, the last passing transcript, and the `agentmon show` command to replay it.

## Demo recording

`docs/demo/stale_kb.cast` shows the Phase 1 exit: with `stale_kb`, the classic check stays at HTTP 200 while agentmon goes red in one cycle, with transcript, evidence and an email alert. Play it with `uvx --from asciinema asciinema play docs/demo/stale_kb.cast`. Re-record it with the command in `docs/demo/record.sh`.

## Layout

| Path | What |
|---|---|
| `agentmon/models.py` | Target, AgentProfile, Probe, Suite, RunTrace |
| `agentmon/adapters/` | Declarative adapters (config, not code): `json_post`, `openai_chat`, `sse`, `webhook` + callback receiver |
| `agentmon/checks/` | Transport, contract (schema, banned strings, PII), tools, latency |
| `agentmon/judge/` | Evidence-gated LLM judge + per-class rubrics (`rubrics/*.md`, version = content hash) |
| `agentmon/engine/` | Runner, grading pipeline, cycle orchestration |
| `agentmon/incidents/` | Open / dedupe / resolve incidents; console, webhook and email notify |
| `agentmon/llm/` | OpenAI-compatible router with named slots (`config/slots.yaml`) and cost ledger |
| `agentmon/db/` | SQLAlchemy schema; migrations in `migrations/` |
| `demo_agent/` | Failure-injectable Acme support bot (DeepSeek, real tool calling; JSON, SSE, OpenAI and async-callback endpoints) |
| `suites/acme/` | Targets (one per transport), profile, ground truth, 27 probes |
| `docs/demo/` | Recorded Phase 1 demo and the script that records it |
