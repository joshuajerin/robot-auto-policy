# H1 Robot Asset

Phase 1 uses the real Unitree H1 model in two places.

Isaac Lab trains and evaluates the policy through the H1 task:

```text
Isaac-Velocity-Flat-H1-v0
```

The repo vendors a compact public Unitree H1 USD bundle here:

```text
assets/unitree_h1/usd/h1.usd
```

The Modal runner bakes `assets/unitree_h1` into the image when it exists,
records it in `h1_asset_report.json`, and exports it into each experiment
artifact directory. The Isaac Lab task remains the source of truth for
simulation dynamics.

Expected asset identifiers:

- robot id: `unitree_h1`
- embodiment: humanoid
- local USD: `assets/unitree_h1/usd/h1.usd`
- runtime Isaac task: `Isaac-Velocity-Flat-H1-v0`
- stretch task: `Isaac-Velocity-Rough-H1-v0`

`h1_isaac_reference.usda` is a lightweight composition reference for local USD
tooling. It is not used for training.
