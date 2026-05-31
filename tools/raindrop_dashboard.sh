#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workshop_session="robogenesis-raindrop-workshop"
replay_session="robogenesis-replay-server"
workshop_url="http://localhost:5899"
replay_url="http://localhost:61020"

cd "$repo_root"

if ! curl -fsS "$workshop_url/health" >/dev/null 2>&1; then
  tmux kill-session -t "$workshop_session" >/dev/null 2>&1 || true
  tmux new-session -d -s "$workshop_session" -c "$repo_root" \
    "bun external/raindrop-workshop/src/index.ts workshop start"
fi

if ! curl -fsS "$replay_url/health" >/dev/null 2>&1; then
  tmux kill-session -t "$replay_session" >/dev/null 2>&1 || true
  tmux new-session -d -s "$replay_session" -c "$repo_root" \
    "python tools/replay_server.py"
fi

for _ in $(seq 1 20); do
  if curl -fsS "$workshop_url/health" >/dev/null 2>&1 && curl -fsS "$replay_url/health" >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
done

curl -fsS "$workshop_url/health" >/dev/null
curl -fsS "$replay_url/health" >/dev/null

bun external/raindrop-workshop/src/index.ts replay register >/dev/null || true
RAINDROP_LOCAL_DEBUGGER="$workshop_url/v1/" \
  python tools/publish_raindrop_renders.py --db artifacts/research.db

cat <<EOF

Raindrop Workshop: $workshop_url
Replay/video server: $replay_url
EOF
