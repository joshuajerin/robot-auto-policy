# RoboGenesis H1 AutoResearch

Autonomous policy iteration for Unitree H1 locomotion in Isaac Lab.

The repo is intentionally narrow: one robot, one locomotion family, one bounded
research loop, and one SQLite lineage record.

```text
propose patch -> validate -> train/evaluate on Modal -> score -> accept/reject -> record lineage
```

## Simulation Videos

Actual H1 Isaac camera rollout:

<video src="docs/media/h1-isaac-camera-run.mp4" poster="docs/media/h1-isaac-camera-frame.png" controls muted loop width="760"></video>

[Open camera rollout MP4](docs/media/h1-isaac-camera-run.mp4)

Telemetry rollout render from the same H1 policy pipeline:

<video src="docs/media/h1-telemetry-rollout.mp4" controls muted loop width="760"></video>

[Open telemetry rollout MP4](docs/media/h1-telemetry-rollout.mp4)

The tracked clips are small examples. Full training videos, checkpoints, Modal
downloads, SQLite DBs, and logs stay under ignored `artifacts/` paths.

## Core Scope

| Layer | Kept Surface |
| --- | --- |
| Robot | Unitree H1 with bundled USD assets in `assets/unitree_h1/` |
| Task | `Isaac-Velocity-Flat-H1-v0` locomotion |
| Planner | Deterministic fallback plus optional OpenAI structured patch planner |
| Patch Surface | `configs/locomotion/*.yaml` only |
| Runner | Modal Isaac Lab H1 train, evaluate, render, and artifact manifest jobs |
| Memory | SQLite experiment, policy, scenario, failure, and score lineage |
| Review | Simple Streamlit dashboard plus Raindrop replay/publishing support |
| Imitation | Video and motion-context helpers for H1 style-conditioning experiments |

## Repository Map

```text
core/                      AutoResearch loop, orchestration, SQLite, scoring, validation
adapters/locomotion/       H1 locomotion adapter, metrics, failures, scenarios
agents/                    Patch planner, OpenAI helper, reviewer, scenario wrapper
configs/locomotion/        Editable reward, curriculum, randomization, PPO, terrain, command YAML
modal_runner/              Modal app plus local spec/eval/render helpers
modal_runner/isaac_scripts/ Scripts copied into the Isaac Lab container
eval/                      Locked local scoring and safety helpers
specs/                     JSON schemas for task, robot, patch, scenario, and eval contracts
dashboard/                 Small local run/policy/scenario/video dashboard
tools/                     Orchestration, Raindrop replay, and imitation helper CLIs
skills/                    H1 motion-imitation helper metadata
tests/                     Focused tests for the retained H1 AutoResearch path
docs/media/                Small README simulation clips
artifacts/                 Ignored generated DBs, specs, checkpoints, videos, and logs
```

## Setup

```bash
python -m pip install -e '.[dev]'
```

Install the dashboard extra when you want the local UI:

```bash
python -m pip install -e '.[dev,dashboard]'
```

For OpenAI-backed patch proposals, set your key locally:

```bash
cp .env.example .env
# edit .env with OPENAI_API_KEY and optionally OPENAI_MODEL
```

The code never stores API keys in tracked files. Without `OPENAI_API_KEY`, the
planner uses the deterministic locomotion fallback.

## Local Dry Run

Run the complete local decision loop without Isaac Lab:

```bash
python -m core.autoresearch_loop --dry-run --experiments 3 --db artifacts/research.db
```

This validates a `PatchSpec`, generates locomotion scenarios, simulates candidate
metrics, applies the accept/reject rule, and records the result in SQLite.

Open the simple dashboard:

```bash
streamlit run dashboard/app.py -- --db artifacts/research.db
```

## Modal H1 Workflow

Build a phase-1 H1 run spec:

```bash
python modal_runner/phase1.py \
  --experiment h1_cleanup_smoke \
  --config configs/locomotion/phase1_h1.yaml \
  --num-envs 64 \
  --max-iterations 2 \
  --write-spec artifacts/specs/h1_cleanup_smoke.json
```

Prepare a bounded AutoResearch Modal iteration:

```bash
python tools/autoresearch_orchestrator.py \
  --db artifacts/research.db \
  --experiments 1 \
  --num-envs 512 \
  --max-iterations 10
```

Submit it to the deployed Modal app:

```bash
python tools/autoresearch_orchestrator.py \
  --db artifacts/research.db \
  --experiments 1 \
  --submit
```

Download, score, review, and record a completed Modal experiment:

```bash
python tools/autoresearch_orchestrator.py \
  --db artifacts/research.db \
  --sync-artifacts \
  --experiment-id <modal_experiment_id> \
  --parent-policy-id baseline_0000
```

The sync step downloads from the `robogenesis-runs` Modal volume, ingests
metrics and videos, scores the candidate, applies the locked reviewer, and
updates SQLite lineage.

## Raindrop Replay

Dry runs and Modal artifact syncs can publish Raindrop-compatible run traces.
Start the local Workshop/replay bridge from the repo root:

```bash
tools/raindrop_dashboard.sh
```

The script expects the ignored local Workshop checkout at
`external/raindrop-workshop/` and uses `.raindrop/agents.yaml` for replay agent
registration. See `docs/agent_eval_workshop.md` for setup details.

## Imitation Helpers

Generate a lightweight video-style context:

```bash
python tools/prepare_video_prompt.py \
  --video artifacts/video_prompts/normal_walk/source_video.webm \
  --output artifacts/video_prompts/normal_walk/context.json
```

Prepare a user-video motion-imitation helper package:

```bash
python tools/prepare_user_motion_skill.py \
  --video artifacts/video_prompts/normal_walk/source_video.webm \
  --output-dir artifacts/user_motion_skill
```

Prepare a CMU-style reference context:

```bash
python tools/prepare_motion_reference.py \
  --config configs/motion_references/cmu_human_walk.yaml \
  --output-dir artifacts/motion_references/cmu_human_walk
```

## Cleanup Policy

The repository keeps source and tiny representative media only. Generated state
is ignored and can be safely regenerated:

```text
artifacts/*        local DBs, specs, checkpoints, Modal downloads, videos, logs
external/          local third-party checkouts
*.db, *.sqlite*    local research databases
*.log              run logs
```

Keep new work inside the H1 locomotion, Raindrop review, dashboard, or imitation
surfaces unless it directly supports the bounded loop above.

## Tests

```bash
pytest -q
```

Focused smoke checks:

```bash
python -m core.autoresearch_loop --dry-run --experiments 1 --db artifacts/smoke.db
python modal_runner/phase1.py --experiment cleanup_smoke --config configs/locomotion/phase1_h1.yaml
python modal_runner/evaluate.py --help
```
