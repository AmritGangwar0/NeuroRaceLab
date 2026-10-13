# NeuroRaceLab — v0.1 Design Freeze

## Scientific question

How do compact recurrent, continuous-time, and equilibrium neural computational architectures compare on dynamical racing control when capacity and computational budget are controlled?

## Frozen architecture set

1. RNN + Residual — 128 neurons
2. Continuous-Time Neural Dynamical System — 128 neurons
3. Implicit / Equilibrium Network — 128 neurons

## Frozen observation

36 dimensions:

- 32 ray distances
- speed
- sideslip angle
- yaw rate
- current steering

## Frozen action

2 dimensions:

- steering
- throttle/brake

## Frozen simulator

- planar dynamic bicycle
- fixed-step RK4
- 100 Hz physics
- 20 Hz controller

## Frozen track philosophy

- closed centerline polyline
- 1 m resampling
- geometric boundaries
- deterministic procedural generation
- separate train/validation/test seeds

## Frozen benchmark regimes

1. Capacity matched
2. Compute matched

## Frozen training family

Alternating:

```text
ES until saturation
        ↓
EA until saturation
        ↓
ES until saturation
        ↓
EA until saturation
        ↓
...
```

The exact ES/EA algorithms and saturation thresholds remain separate training-policy decisions and must be specified before trainer implementation.

## Frozen scientific rules

- Same observation/action interface.
- Same environment families.
- Same evaluation protocol.
- Same random-seed protocol.
- Same training budget within each comparison regime.
- Test tracks remain isolated.
- All compute is explicitly accounted for.
- Architecture-specific advantages must not be hidden inside the generic trainer.
- No architecture may receive a hidden feature unavailable to the others.

## Intentionally not frozen yet

- exact ES algorithm
- exact EA algorithm
- population size
- mutation/crossover details
- ES noise schedule
- saturation threshold/window
- final fitness weights
- exact parameter tolerance
- compute-matching tolerance
- number of training/evaluation seeds

These are the next experimental-policy decisions.
