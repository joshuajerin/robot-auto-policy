# OpenAI Setup

RoboGenesis uses OpenAI for research reasoning only:

- proposing a bounded `PatchSpec`
- diagnosing failures from metrics/video summaries
- proposing generated scenarios
- writing experiment notes

The LLM never controls robot torques and never edits arbitrary files. All model
outputs go through schemas and deterministic validators.

## Environment

Create a local `.env` or export these variables in your shell:

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-5.4-mini"
```

`OPENAI_MODEL` is optional. The code reads it so you can switch models without
editing source.

## Planner CLI

```bash
python -m agents.openai_planner \
  --context-json artifacts/planner_context.json
```

If `OPENAI_API_KEY` is not set, the planner uses the deterministic local
fallback. Use `--no-fallback` to require a real API call.

## Implementation Notes

The OpenAI client uses the Responses API with a JSON Schema text format. The
returned JSON is parsed into `PatchSpec`, then passed to `core.patch_validator`
before any config file can be changed.
