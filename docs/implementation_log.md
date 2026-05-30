# Implementation Log

This repository was built as a feature-by-feature RoboGenesis scaffold with a
commit after each coherent step.

## 1. Project Scope And Program Rules

Added:

- `README.md`
- `program.md`
- `pyproject.toml`
- `.gitignore`
- `artifacts/.gitkeep`

Purpose:

- define the locomotion-first AutoResearch thesis
- lock the evaluator and runner surfaces
- list the editable config files
- establish local test and artifact conventions

## 2. Contracts And Adapter Boundary

Added:

- JSON Schemas in `specs/`
- typed dataclasses in `core/schemas.py`
- generic `TaskAdapter` protocol
- `LocomotionAdapter`
- manipulation adapter stub

Purpose:

- make locomotion the first plugin rather than hardcoded behavior
- keep `PatchSpec`, `ScenarioSpec`, `TaskSpec`, and score contracts stable
- preserve a path to manipulation/navigation adapters later

## 3. Locomotion Config Defaults

Added:

- reward, curriculum, randomization, actuator, PPO, terrain, and eval YAML
- fixed eval seed file

Purpose:

- provide a constrained patch surface for the planner
- keep eval configuration and seeds outside the editable agent surface

## 4. Deterministic Safety Core

Added:

- locked score composition
- strict accept/reject rule
- PatchSpec validator
- YAML dry-run/application patcher
- scenario generator and frontier classifier
- metric-first failure diagnosis
- SQLite experiment memory

Purpose:

- reject unsafe patches before training
- prevent evaluator edits and torque-limit escalation
- track experiment, policy, scenario, and failure lineage

## 5. Agent Layer

Added:

- deterministic planner fallback
- reviewer checks
- scenario agent wrapper
- video context placeholder
- prompts for future OpenAI structured-output calls

Purpose:

- keep the system runnable offline
- make the OpenAI integration a swap-in replacement for deterministic proposal code
- represent video as style context, not direct control

## 6. Locked Evaluation Helpers

Added:

- locomotion safety checks
- score CLI wrapper

Purpose:

- keep safety/eval logic outside the editable patch surface
- provide a scriptable path from raw Isaac metrics to RoboGenesis score JSON

## 7. AutoResearch Loop And Modal Runner

Added:

- dry-run AutoResearch controller
- Modal Isaac Lab app
- train/evaluate/render helper scripts

Purpose:

- prove the full outer loop locally
- shape the Modal execution layer for Isaac Lab training artifacts
- keep Modal execution separate from patch planning and acceptance

## 8. Dashboard

Added:

- Streamlit app
- experiment, scenario, policy, and rollout loaders

Purpose:

- show the judge-facing policy lineage and scenario history
- read directly from SQLite research memory

## 9. Tests

Added:

- patch validator tests
- scoring tests
- scenario and DB tests
- dry-run loop test

Purpose:

- protect the locked evaluator and patch validator from regression
- verify the local loop can run without Isaac Lab
