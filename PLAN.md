# Build Plan — Semantic Uptime Monitor for AI Agents

Source: design doc "Semantic Uptime Monitor for AI Agents" (20 Oct 2026). Estimates assume ~full-time solo; scale up for part-time.

## Stack

- **Python 3.12 + uv.** Eval/stats ecosystem lives here (scipy, numpy, pydantic).
- **FastAPI** (API, webhook receiver), **SQLAlchemy 2 + Alembic**, **Postgres 16** via docker-compose from day 1 (JSONB transcripts, no SQLite→PG migration later).
- **httpx async** for target calls. **OpenAI SDK** for both LLM providers (both OpenAI-compatible).
- **Scheduler:** APScheduler in-process (P1–P3) → Postgres-backed queue (procrastinate) + worker pool in P4.
- **Config-as-code:** targets / profiles / suites as YAML in git → versioning and CI reuse for free. DB holds runtime data only.
- **Libs:** jsonpath-ng (adapters), Jinja2 sandboxed env (request templates), rapidfuzz (evidence-quote check), scipy (Wilson, Fisher, CMH), pydantic (judge output schema).
- **Dashboard (P3):** Next.js + Recharts.

## Repo layout

```
agent-monitor/
  pyproject.toml  docker-compose.yml  .env.example
  agentmon/
    models/       # pydantic: Target, AgentProfile, Probe, Suite, Run, Verdict, HealthVector, Incident
    db/           # SQLAlchemy tables + Alembic migrations
    llm/          # provider interface, named slots, cost ledger, `doctor`
    adapters/     # openai_chat, json_post, sse, webhook + declarative mapping
    engine/       # runner: probe -> Run (transcript, latency, raw, tool calls)
    checks/       # transport, contract, tool, latency (deterministic)
    judge/        # rubrics/ (versioned prompts), single, panel, evidence validator, cache
    scoring/      # wilson, baselines, flaky, cusum, fingerprint, health vector
    incidents/    # alert rules, dedupe, cause hints, notifiers
    generator/    # P3: profile -> draft probes
    simulator/    # P3: multi-turn persona
    sampler/      # P3: stratified rotation, cost caps
    ci/           # P4: preview vs control
    api/          # FastAPI routes
    cli.py
  demo_agent/     # failure-injectable target
  suites/         # YAML targets, profiles, probes, fixtures
  golden/         # hand-labelled transcripts
  web/            # Next.js dashboard (P3)
  tests/
```

## Phase 0 — Foundation + demo agent (~1 wk)

Demo agent comes first: it is the test fixture for every later phase.

- `demo_agent/`: FastAPI "Acme Store support bot" on DeepSeek v4-flash.
  - KB of ~20 atomic facts (refund window 30 days, shipping regions, ...).
  - Tools: `order_lookup`, `get_balance`, `issue_refund`.
  - Seeded fixtures: order `A123` = "shipped, DHL, arrives Oct 24"; test account balance `$42.17`.
  - Fault toggles via `POST /admin/faults`:

| Toggle | Effect | Class exercised |
|---|---|---|
| `stale_kb` | refund window 30 → 14 days | hallucination |
| `kb_gap_fill` | prompt: "never say you don't know" | unanswerable / ungrounded |
| `broken_tool` | `order_lookup` returns 500, agent answers anyway | wrong tool |
| `strict_prompt` | over-cautious system prompt | over-refusal |
| `lax_prompt` | safety rules stripped | under-refusal |
| `policy_leak` | recommends a competitor | policy / tone |
| `drop_history` | only last turn sent to model | memory loss |
| `model_swap` | switch to another family (e.g. Mistral on Nebius) | silent model swap |
| `format_break` | JSON mode off | format contract |
| `slow` | +4 s sleep | latency |
| `gradual_decay` | X% of facts corrupted, X grows per hour | drift (CUSUM) |

- Monitor skeleton: pydantic domain models, DB schema, LLM provider + slots + cost ledger, `agentmon doctor` (lists reachable models per slot via `/models`).
- Plain HTTP status check built alongside, for the "green vs red" demo.

**Schema decisions to lock now (expensive to retrofit):**
- Run has `cycle_id` + `attempt` → N-runs in P2 need no migration.
- Probe has stable `probe_id` + `content_hash`; editing creates a new version and resets its baseline.
- Verdict stores `rubric_version`, `judge_provider`, `judge_model`, `response_hash` → replay, cache, judge-drift audit.
- Verdict status enum: `pass | fail | uncertain | judge_error` — judge errors never count as agent failures.
- Cost ledger row per LLM call (slot, model, tokens in/out/cached, $).
- All synthetic traffic tagged with header `X-Agent-Monitor: probe/<run_id>` so owners can filter it from their analytics and billing.

**Exit:** every toggle reproducible by hand; `agentmon doctor` green.

## Phase 1 — Core loop MVP (~2 wk)

1. **Adapters** — declarative YAML; presets `openai_chat`, `json_post`, `sse`, `webhook` (monitor exposes a callback endpoint, correlates by run id). State modes: `server_session` | `client_history`.
   ```yaml
   adapter:
     kind: json_post
     request:
       url: ${TARGET_URL}/chat
       headers: {Authorization: "Bearer ${TARGET_TOKEN}"}
       body: {session_id: "{{session_id}}", message: "{{user_turn}}"}
     response:
       reply: $.output.text
       tool_calls: $.output.tools[*]   # optional
     state: server_session
   ```
2. **Runner** — async; probe → Run (transcript, per-turn latency, raw responses, tool calls if mapped). Retries on transport errors only.
3. **Deterministic pipeline** (stop at first hard fail): transport → contract (JSON schema, required fields, max length, banned strings, PII regex) → tool (required / forbidden / arg match) → judge → latency (static threshold for now).
4. **Judge v1** — single Nebius model from a non-DeepSeek family. One rubric file per failure class, versioned. JSON output validated by pydantic. **Every evidence quote is checked against the transcript** (normalised substring or rapidfuzz ≥ 90); a fabricated quote makes the verdict invalid. One retry, then `judge_error`.
5. **~25 hand-written probes** for the demo agent covering all classes.
6. **Scheduler** — one cycle every X min, every probe once.
7. **Alerts** — Slack-compatible webhook + email. Payload: probe, transcript, expected outcome, judge reasoning, evidence.
8. **CLI** — `agentmon run --suite suites/acme.yaml`, `agentmon show run <id>`.

**Exit:** toggle `stale_kb` → HTTP check green, monitor red with transcript + evidence within one cycle. Record it — first demo asset.

## Phase 2 — Trust (~3 wk)

1. **N runs per probe per cycle** (default 3).
2. **Statistics fix vs the doc.** Wilson at N=3 is too wide to act on per cycle: 3/3 → 95% CI [0.44, 1.00]; 1/3 → [0.06, 0.79]. "Healthy = lower bound above threshold" would almost never be true. Instead:
   - Probe state from a **rolling window** (last W cycles, e.g. W=5 → 15 runs; 15/15 → lower bound 0.80). States: `healthy | flaky | failing | insufficient_data`.
   - **Fast path** only for canaries / P0: any failure on a probe with baseline ≥ 0.95 → immediate.
   - Incident when window upper bound < baseline threshold AND it persists M cycles AND no open incident covers it.
3. **Baselines** — first K≈20 cycles after approval + owner "known good" marks; per-probe threshold = baseline pass rate − margin.
4. **Flaky detection** — window pass rate in a mid band and cross-cycle variance above binomial expectation → "unstable probe" list, no alert.
5. **Golden set** — ~100 transcripts sampled from P1 runs across all toggles, balanced pass/fail per class. Small labelling CLI/page (pass/fail + class + note). You + one other labeller on a 30-item overlap → human–human κ = ceiling for judge κ.
6. **Judge eval harness** — `agentmon judge-eval --config judges.yaml` → accuracy, Cohen's κ, per-class confusion. Deploy gate e.g. κ ≥ 0.7 and per-class recall ≥ 0.8 (tune). Results stored per judge config.
7. **Panel** — 2–3 Nebius models from different families, auto-excluding the agent's family (Profile field `model_family`). Majority vote; split → `uncertain`, excluded from alerting, queued for review.
8. **Cascade + cache** — cheap judge first; full panel when confidence < τ or probe is P0. Cache key `(probe_id, response_hash, rubric_version, judge_config)`. Shared system prompt + rubric first in every call for provider prompt caching.
9. **Latency baseline** — per-probe p50/p95 from the baseline window.
10. **Judge drift canary** — weekly golden-set regrade; accuracy shift flags that week's alerts.

**Exit:** published judge-vs-human κ; 7 days on the stable demo agent (no toggles) with 0 incidents; every toggle still caught.

## Phase 3 — Breadth (~4 wk)

1. **Agent Profile** schema + ground-truth ingestion (md / txt / PDF / URL).
2. **Generator** (DeepSeek v4-pro) — extract atomic facts with source spans → probes per class template → dedupe by embedding cosine (> ~0.9 dropped) → `draft`.
3. **Approval flow** — approve / edit / reject; only approved probes run. Suite version = hash of approved probe set.
4. **Simulator** (DeepSeek v4-flash, temp 0) — persona + goal + turn budget; stop on goal / failure / budget. It never sees the expected answer: enforced structurally, the prompt is built from `persona` + `goal` only. Cache simulator turns keyed by `(probe, transcript-so-far hash)` so identical agent replies get identical next turns → variance comes from the agent, not the simulator.
5. **Answer-verifiable tool probes** — `fixtures` section in the profile; correct fixture value in reply ⇒ tool used; plausible wrong value ⇒ not used.
6. **Sampler** — canaries + all P0 every cycle; stratified rotation for the rest with every class each cycle; full coverage per 24 h; monthly cost cap enforced.
7. **Health Vector** — per-axis severity-weighted mean, time series with baseline band.
8. **Drift** — CUSUM per axis (k = 0.5σ, h ≈ 4–5σ from baseline window). Validate with `gradual_decay`: must fire before a static threshold would.
9. **Model-swap fingerprint** — canary features (length, type-token ratio, refusal phrasing, markdown usage, embedding centroid) vs baseline; two-sample test (KS / energy distance). No judge needed.
10. **Cause hints** — rule table from the doc (all axes + fingerprint → model change; one KB source → stale KB; tool probes only → downstream API; refusal only → prompt/safety change).
11. **Incidents** — dedupe by axis + probe cluster; payload with before/after scores, 3 failing transcripts beside the last passing ones, evidence, cause hint.
12. **Dashboard** — targets, health vector, incidents, transcript viewer with evidence highlighting, probe approval, flaky list.

**Exit:** all 8 failure classes detected on demo-agent toggles, each with the correct cause hint.

## Phase 4 — SaaS-ready (~3–4 wk)

1. **CI gate** — `agentmon ci --preview URL --control URL --subset ci` + GitHub Action wrapper; preview and control run in parallel.
   - **Statistics caveat:** per-probe Fisher at N=3 can't reach significance (3/3 vs 0/3 → one-sided p = 0.05 at best). Use N ≥ 10 for P0 probes in CI, and a pooled axis-level test (Cochran–Mantel–Haenszel across probes) for "axis dropped".
   - Position-swap double judging for comparative grading.
   - PR comment: axis diff table, failing transcripts, report link.
   - Per-PR token/cost ceiling; sampler drops P2 first.
   - Preview scores become a baseline candidate on merge; promoted after M healthy prod cycles.
2. **Multi-tenant** — `org_id` on every table, API keys, Postgres RLS, encrypted target secrets; move to job queue + workers.
3. Per-target monthly cost caps.
4. Public status page per target (per-axis uptime %), shareable SLA report (HTML → PDF).
5. Onboard one external agent (OSS agent or a friend's).

**Exit:** external user live; CI blocks a real bad-prompt PR on the demo agent repo.

## Initial LLM slots (verified against provider `/models`, 2026-10-20)

| Slot | Provider | Model |
|---|---|---|
| generator | DeepSeek | `deepseek-v4-pro` |
| simulator | DeepSeek | `deepseek-flash` (the API ID; the doc's "v4-flash") |
| judge panel | Nebius Token Factory | `Qwen/Qwen3-235B-A22B-Instruct-2507`, `openai/gpt-oss-120b`, `zai-org/GLM-5.3` |
| safety | — | **Llama Guard is not in this account's Nebius catalogue.** Options: a judge-model safety rubric, or another provider |
| embeddings | Nebius | `Qwen/Qwen3-Embedding-8B` |
| demo agent | DeepSeek | `deepseek-flash` → panel auto-excludes DeepSeek; `model_swap` uses `moonshotai/Kimi-K2.6` (outside the panel) |

No Llama 3.x 70B on Nebius for this account either. Families available there: Qwen, DeepSeek, gpt-oss, GLM, Kimi, Gemma, Nemotron, MiniMax, Hermes.
Base URLs: DeepSeek `https://api.deepseek.com`, Nebius `https://api.tokenfactory.nebius.com/v1/` (both verified live). `agentmon doctor` re-checks every slot.

## Volume and cost

runs/day = probes per cycle × N × cycles/day. Example: 18 × 3 × 96 (15-min cycles) ≈ 5.2k runs/day; deterministic checks + cascade put roughly half through the judge. Cycle interval is the main knob. Simulator cost on `deepseek-flash` is small: $0.30 / $1.20 per M in/out at peak, half that off-peak, cache hit $0.006 peak (official pricing page, 2026-10-21). The target owner pays for their agent's own calls, so setup shows estimated target calls/day.

## Issues spotted in the design doc

- Closest competitor is named both "OnStatus (Carmel Labs)" and "AgentStatus" — pick one.
- Statistics as written are underpowered at N=3 (per-cycle Wilson, per-probe CI significance) — fixes in P2 and P4.
- Answer-verifiable tool probes need seeded fixtures. For third-party agents (a key wedge) the owner can't seed, so those targets only get the weaker inference-only tool check. Say so in positioning.

## Decisions (recommendation first)

1. Stack: Python + FastAPI + Postgres; Next.js dashboard.
2. Provider split: DeepSeek for generator / simulator / demo agent; Nebius for judges / safety / embeddings.
3. Demo agent: your own first (controllable faults); add an OSS agent in P4.
4. Golden labelling: you + one person on a 30-item overlap.

## Week 1 tasks

1. `uv init`, docker-compose Postgres, Alembic baseline schema with the locked fields above.
2. LLM client + slots + cost ledger + `agentmon doctor`.
3. Demo agent: KB, 3 tools, fixtures, first 3 toggles (`stale_kb`, `broken_tool`, `strict_prompt`).
4. `openai_chat` + `json_post` adapters.
5. 10 hand-written probes, runner, transport + contract checks.
