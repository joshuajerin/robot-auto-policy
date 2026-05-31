# Agent Eval Workshop

This repo can run Raindrop Workshop as a local trace debugger for evaluating
agent behavior while RoboGenesis experiments run.

The workshop source is kept in an ignored local checkout:

```text
external/raindrop-workshop/
```

## Setup

```bash
git clone https://github.com/raindrop-ai/workshop.git external/raindrop-workshop
cd external/raindrop-workshop
bun install
```

## Run

The easiest repo-level command is:

```bash
tools/raindrop_dashboard.sh
```

That starts the Workshop app on port `5899`, starts the RoboGenesis
replay/video server on port `61020`, registers the replay agent, and publishes
local MP4-backed render runs into Workshop.

To run Workshop directly:

```bash
cd external/raindrop-workshop
bun src/index.ts workshop start
```

Workshop starts:

```text
ui/api: http://localhost:5899
```

Use it to inspect agent traces, tool calls, and replay/eval flows while keeping
RoboGenesis source changes separate from the workshop checkout.

## RoboGenesis Sim Traces

Use the dashboard setup script to start Workshop, start the local replay/video
server, register the replay agent, and publish local render artifacts into
Raindrop:

```bash
tools/raindrop_dashboard.sh
```

Or start just the local replay/video server before opening a run with MP4
artifacts:

```bash
python tools/replay_server.py
```

The replay/video server exposes:

- `GET /health` for Workshop replay registration
- `GET /artifact-video?path=<repo-relative-mp4>` for an HTML video page
- `GET /artifact-file?path=<repo-relative-file>` for raw artifacts and MP4 byte ranges

Dry-run AutoResearch traces publish task-level spans when Workshop is running:

```bash
RAINDROP_LOCAL_DEBUGGER=http://localhost:5899/v1/ \
python -m core.autoresearch_loop --dry-run --experiments 3
```

Modal phase-1 jobs write `raindrop_trace.json` inside each experiment artifact
directory. After a run finishes, sync artifacts locally; this downloads the
Modal Volume run, ingests it into SQLite, replays the Modal train/eval/render
task timeline into Raindrop, and adds links to the full rollout MP4:

```bash
python tools/autoresearch_orchestrator.py \
  --sync-artifacts \
  --experiment-id <experiment_id> \
  --parent-policy-id <parent_policy_id>
```

If `tools/replay_server.py` is running, the Raindrop run output includes local
video pages and raw MP4 links backed by the artifact server. Render tasks also
carry `render_videos` entries, and Workshop renders run-level
`video_artifacts[]` inline in the run Overview tab.
