# Unitree H1 Asset Bundle

This directory is populated by:

```bash
python tools/fetch_h1_model.py --output-dir assets/unitree_h1 --refresh
```

The files come from the official Unitree model repository:

```text
https://github.com/unitreerobotics/unitree_model
```

Phase 1 still trains the Isaac Lab native task:

```text
Isaac-Velocity-Flat-H1-v0
```

This asset bundle is mounted into Modal so every run can record exact H1 model
provenance and so non-Vulkan visualization/export code has a real H1 USD bundle
available.
