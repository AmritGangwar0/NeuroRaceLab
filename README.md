# NeuroRaceLab

Modular research framework for comparing compact neural computational architectures through a controlled racing environment.

## Initial architectures
1. RNN + Residual
2. Continuous-Time Neural Dynamical System
3. Implicit / Equilibrium Network

All use a 128-neuron primary computational state and a 36-dimensional observation:
32 ray measurements + scalar speed + slip angle + angular velocity + steering state.

## Benchmark regimes
- Capacity-matched: global budget = environment evaluations.
- Compute-matched: global budget = FLOPs.
- Wall-clock time is secondary.

## Training
ES until saturation -> EA until saturation -> repeat until the global budget is exhausted.

## Current phase
Structural foundation. Interfaces, registries, RNG management, accounting, evaluation skeleton,
configuration schema, ADRs and research-sanity tests are established. Physical simulation,
three research architectures and full optimizers remain the next design/implementation stage.

## v0.1 environment foundation

The first executable environment layer is now implemented:

- dynamic bicycle vehicle model
- fixed-step RK4 integration
- closed-loop track geometry
- deterministic procedural tracks
- 32-ray boundary sensor
- 36-dimensional observation
- 2-dimensional continuous action
- deterministic reset/rollout behavior

Run the environment smoke test with:

```bash
PYTHONPATH=src python scripts/smoke_test_environment.py
```

Run the complete test suite with:

```bash
pytest
```
