#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workshop_dir="$repo_root/external/raindrop-workshop"

if [ ! -d "$workshop_dir/.git" ]; then
  mkdir -p "$repo_root/external"
  git clone https://github.com/raindrop-ai/workshop.git "$workshop_dir"
fi

cd "$workshop_dir"
bun install

if [ ! -f app/dist/index.html ]; then
  bun run build:ui
fi

exec bun run dev
