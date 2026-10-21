# Handoff

## Goal
Build the Semantic Uptime Monitor for AI Agents (outside-in, zero-instrumentation behavioural monitor). The spec is the design doc pasted in chat on 2026-10-20 and 2026-10-21; it is not saved in the repo. Build plan: `PLAN.md`. How to run: `README.md`.

## Status
- **Phase 0: done** (2026-10-21). `gradual_decay` fault; cost ledger priced, with live Nebius prices and DeepSeek peak/off-peak; `verdicts.response_hash`; tagged uptime check; `doctor` exit code.
- **Phase 1: done** (2026-10-21). Every PLAN.md Phase 1 item is built and verified on the real demo agent. The exit demo is recorded at `docs/demo/stale_kb.cast`.
- Phases 2–4: not started.

## Just completed: Phase 1 gaps
- **Jinja2 sandboxed templates** (`adapters/base.py`). StrictUndefined, so a typo fails loudly. A lone `{{ expr }}` keeps its type. `${ENV}` is injected as a variable, so user text and secrets are never re-read as templates. URL and headers are templated too.
- **`sse` adapter** (`adapters/sse.py`, httpx-sse). Per-event JSONPaths for delta and tool calls; done sentinel; a stream that ends without it is "malformed"; total-time timeout. Defaults match OpenAI streaming.
- **`webhook` adapter** (`adapters/webhook.py`). The agent acks, then POSTs its reply. A per-process, ref-counted callback receiver (Starlette on embedded uvicorn) runs on `CALLBACK_HOST/PORT` (default 127.0.0.1:8200). It binds the socket itself, because uvicorn calls `sys.exit` when a port is taken, and it doesn't hijack Ctrl-C. Correlation is by a per-turn URL token (contains the run id), or a fixed `/callbacks` URL + `callback.correlate` JSONPath.
- **Runner retries** (`engine/runner.py`). On timeout, connection error, 429 or 5xx, the whole probe restarts in a fresh session, up to `target.transport_retries` (default 2) with 1 s/2 s backoff. New `runs.retried_errors` column (migration `8dd6893b677f`). Malformed or empty replies are not retried.
- **Email notifier** (`incidents/`, smtplib in a thread). `SMTP_*` + `ALERT_EMAIL_TO` settings. The email carries the summary, the full failing and last-passing transcripts, the evidence, and the replay command.
- **`watch` on APScheduler** (AsyncIOScheduler). Several suites per process; first cycle runs immediately; overlapping cycles are skipped (`max_instances=1`, coalesce). It picks up interval edits from the YAML, keeps going after a failed cycle, and stops gracefully on Ctrl-C or SIGTERM.
- **Probes 16 → 27**; every class now has ≥ 2 except canary (1).
- **Demo agent** gained real token streaming (`/chat/stream` with tool events; OpenAI `stream: true`), passes thinking-mode `reasoning_content` back on tool rounds, has `/chat/async` with callback delivery, and its OpenAI endpoint honours `response_format` / `format`.
- Suite variants per transport: `suites/acme/suite-{sse,openai-stream,webhook}.yaml`.
- Fixed a loader bug: `ground_truth` paths now resolve against the profile file, as documented.
- Tests: 30 → 49 (`test_transports.py`, `test_models.py`, template tests).

## Verified on the real target (2026-10-21)
- Healthy, json_post, n=3: 81/81 pass. SSE 27/27, webhook 27/27, OpenAI-stream 27/27 (after the format fix). Healthy again after the matrix: 27/27.
- Fault matrix (json_post, 27 probes, cycles `622e95…` to `b9422b…`). Every toggle was caught while the classic check stayed at HTTP 200:
  stale_kb 6 fails · kb_gap_fill 3 (all unanswerable) · broken_tool 3 · strict_prompt 8 (over_refusal) · lax_prompt 3 (pii_leak on refuse-other-phone) · policy_leak 2 · drop_history 2 (both memory) · format_break 2 · slow 27/27 latency · gradual_decay 12 (4→7 facts decayed during the cycle) · model_swap 20, but **only through latency/timeouts**: Kimi-K2.6 on Nebius took 20–80 s per turn. That is not style detection, which is still Phase 3.
- Retries, live against a closed port: 2 retries, 4 s total, then a transport failure "(after 2 retries)".
- `watch`: cycles at +0 and +1 min; rescheduled to 2 min after the YAML edit; three suites (two webhook, sharing the receiver) ran concurrently and the receiver was released at exit; SIGTERM gives a clean "stopped".
- Email went over real SMTP to a local aiosmtpd sink (in the recording too). **Not verified: delivery through a real mail provider** (needs the user's SMTP credentials).

## Current state
- Nothing committed yet (user hasn't asked). `.env` is gitignored.
- Running: demo agent on :8100 (latest code, faults reset) and Postgres `agentmonitor-db-1` on :5433, migrated to head.
- The DB contains test targets from today's checks: `retry-check` (one open P0 incident, closed-port canary), `acme-demo-watch`, `multi-wh1/wh2/sse`. They are harmless and can be deleted.
- Port 8000 belongs to an unrelated process of the user's. Don't touch it.

## Known issues / dead ends
- **Single-judge inconsistency:** under gradual_decay, run `2bb75148` was correctly a fail (the agent said "21 days"), but the judge's reasoning argued pass while its verdict said `fail`/`wrong_tool`. A good golden-set item; Phase 2's panel and agreement scoring target exactly this.
- Probe `request_overrides` are transport-specific (`format: json` only works where the target understands it). Fine for the demo; real multi-transport suites would need per-target overrides.
- Phase 1 incident dedupe is per probe and ignores the fault class (a new fault on an already-open probe sends no new alert). Phase 2 fixes it.
- Nebius lists no cache-read price, so cached tokens are billed at the full input rate. DeepSeek's Chinese-holiday off-peak pricing isn't modelled. 506 old ledger rows are unpriced.
- `slow` defaults to +12 s (the 20 s budget needs ~25 s to trip until Phase 2 latency baselines exist). KB has 18 facts. No pydantic HealthVector model yet.
- The demo agent keeps only text in session history, so `memory-name-and-order` can flake (a Phase 2 flaky-probe case).
- `--json` CLI output (listed in an earlier audit) is not in PLAN Phase 1 and is not built.

## Next steps
1. Ask the user: commit? Save the design doc to `docs/design.md`? Delete today's test targets from the DB? Publish the demo recording as a shareable page?
2. Phase 2: N-run rolling-window Wilson state machine (+ insufficient_data); baselines; flaky detection; golden set from stored runs (include `2bb75148`) + `judge-eval` (accuracy, κ); 3-judge panel; judge cache on (probe, response_hash, rubric_version, judge).
3. Decide before Phase 2: golden-set labellers (the plan recommends the user + one other person on a 30-item overlap).
4. Phase 3: canary style fingerprint, so `model_swap` is caught by style rather than latency; CUSUM validated with `gradual_decay` at its default rate.
