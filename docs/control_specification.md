# NeuroRaceLab — Control Specification

## Status
**Design frozen for v0.1 benchmark implementation.**

## Action

Two continuous actions:

```text
action[0] = steering ∈ [-1, 1]
action[1] = longitudinal ∈ [-1, 1]
```

Steering maps linearly to:

`[-0.45, +0.45] rad`

Longitudinal action maps to:

- `[0, 1]`: throttle
- `[-1, 0)`: braking

## Timing

Physics:

`100 Hz`

Controller:

`20 Hz`

Therefore:

`5 physics steps / control decision`

The network is evaluated exactly once per control decision.

## Action holding

The selected action remains constant during the five physics substeps.

## Initial state

Default start:

- centerline position: start/finish point
- lateral error: 0
- heading: tangent to centerline
- speed: 5 m/s
- lateral velocity: 0
- yaw rate: 0
- steering: 0

Initial-state perturbation is disabled for baseline training and enabled during robustness evaluation.

## Episode limits

Default maximum duration:

`120 s`

The exact lap-count requirement is benchmark-configurable.

## Fitness

The primary scalar fitness is:

\[
J =
w_p P
+ w_l L
- w_e E
- w_c C
- w_t T
\]

where:

- `P`: normalized forward progress
- `L`: completed-lap reward
- `E`: off-center/path error penalty
- `C`: crash/off-track penalty
- `T`: time penalty

The implementation must keep the components separately logged so the final fitness is auditable.

Fitness weights belong to configuration, not hard-coded trainer logic.
