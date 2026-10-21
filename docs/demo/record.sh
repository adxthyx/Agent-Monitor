#!/bin/zsh
# Phase 1 demo: a classic uptime check stays green while agentmon catches a stale knowledge base.
#
# Needs: the demo agent on :8100, Postgres up. Email alerts go to a local SMTP sink, started here.
# Record:  uvx --from asciinema asciinema rec -c docs/demo/record.sh --cols 140 --rows 45 -i 2 --overwrite docs/demo/stale_kb.cast
set -e
cd "${0:A:h}/../.."
inbox=$(mktemp -t agentmon-inbox)
uv run python -u -m aiosmtpd -n -l 127.0.0.1:1025 -c aiosmtpd.handlers.Debugging stdout > $inbox 2>&1 &
sink=$!
trap 'kill $sink 2>/dev/null; curl -s -XPOST localhost:8100/admin/faults/reset >/dev/null; rm -f $inbox' EXIT
export SMTP_HOST=127.0.0.1 SMTP_PORT=1025 SMTP_SECURITY=none ALERT_EMAIL_TO=oncall@acme.example COLUMNS=140

say() { print -P "\n%B%F{cyan}# $1%f%b"; sleep 2 }
run() { print -nP "%F{yellow}\$%f "; print -r -- "${(j: :)${(q-)@}}"; sleep 1; "$@" }
probes=(-p canary-about -p fact-refund-window -p tool-forbidden-refund-question -p format-json-refund-window)

curl -s -XPOST localhost:8100/admin/faults/reset >/dev/null
say "1. Healthy agent: the classic uptime check and agentmon agree"
run uv run agentmon run suites/acme/suite.yaml $probes --note "demo: healthy"

say "2. Ship a stale knowledge base (refund window 30 days -> 14). The server still answers 200."
print -nP "%F{yellow}\$%f "; print -r -- "curl -XPOST localhost:8100/admin/faults -d '{\"stale_kb\": true}'"
curl -s -XPOST localhost:8100/admin/faults -H 'content-type: application/json' -d '{"stale_kb": true}' | jq -c '{stale_kb}'
run curl -s -o /dev/null -w "GET /health -> HTTP %{http_code}\n" localhost:8100/health

say "3. Same cycle: classic check green, agentmon red within one cycle, with transcript and evidence"
run uv run agentmon run suites/acme/suite.yaml $probes --note "demo: stale_kb"

say "4. Replay the failing run"
rid=$(docker exec agentmonitor-db-1 psql -U agentmon -tAc "select r.id from runs r join probes p on p.id = r.probe_id where p.probe_key = 'tool-forbidden-refund-question' order by r.started_at desc limit 1")
run uv run agentmon show ${rid:0:8}

say "5. The alert email that went out (local SMTP sink): summary part, transcripts follow below it"
sleep 1
uv run python - $inbox <<'PY'
import email, sys
raw = open(sys.argv[1]).read().split("---------- MESSAGE FOLLOWS ----------")[1:]
for m in [email.message_from_string(r.split("------------ END MESSAGE")[0].lstrip()) for r in raw]:
    if m["Subject"].startswith("[RESOLVED]"):
        continue
    print(f"Subject: {m['Subject']}\nTo: {m['To']}\n")
    print(m.get_payload(decode=True).decode().split("\nFailing run")[0].strip(), "\n" + "-" * 100)
PY
sleep 5
