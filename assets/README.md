# H1 Robot Asset

Phase 1 uses the real Unitree H1 model that ships with Isaac Lab / Isaac Sim
through the task:

```text
Isaac-Velocity-Flat-H1-v0
```

The repository does not vendor NVIDIA/Unitree USD meshes. Instead, the Modal
runner inspects the Isaac Lab container and writes the resolved H1 asset path to
each experiment's artifact manifest.

Expected asset identifiers:

- robot id: `unitree_h1`
- embodiment: humanoid
- Isaac task: `Isaac-Velocity-Flat-H1-v0`
- stretch task: `Isaac-Velocity-Rough-H1-v0`

`h1_isaac_reference.usda` is a lightweight composition reference for local USD
tooling. The actual robot mesh is resolved by Isaac Lab at runtime.
