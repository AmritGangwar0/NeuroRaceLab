# NeuroRaceLab — Compute and Capacity Accounting

## Status
**Design frozen for v0.1 benchmark implementation.**

## 1. Primary benchmark regimes

### Capacity-matched

All architectures use:

`128 primary computational neurons`

Parameter count should remain within a predefined tolerance around the baseline architecture.

Primary budget:

**environment evaluations**

### Compute-matched

Architectural width and/or iteration count may be changed to equalize inference compute.

Primary budget:

**estimated inference FLOPs**

All runs record both quantities regardless of regime.

## 2. FLOP definition

Count multiply-add operations as:

`1 multiply + 1 add = 2 FLOPs`

Count:

- matrix/vector multiplications
- elementwise nonlinearities using a documented fixed cost
- repeated equilibrium iterations
- repeated continuous-time evaluations

Do not count:

- Python control flow
- logging
- environment rendering
- file I/O

unless a benchmark explicitly targets end-to-end system latency.

## 3. Architecture-specific accounting

### RNN + Residual

One control step:

`F(h, x)` once.

### Continuous-time

If RK4 is used internally:

`F` is evaluated four times per controller step.

This multiplier is explicitly included in FLOP accounting.

### Equilibrium

If the equilibrium solver performs `K` iterations:

`F` is evaluated `K` times.

The actual iteration count is recorded for every run.

## 4. Environment evaluations

One environment evaluation means one complete episode rollout under one parameter set.

For population methods, every independently evaluated candidate counts separately.

## 5. Wall-clock time

Wall-clock time is secondary.

Record:

- elapsed training time
- evaluation time
- inference latency
- hardware/software metadata

but do not use wall-clock time as the primary cross-architecture scientific budget.

## 6. Required accounting fields

Every experiment records:

```text
parameter_count
flops_per_control_step
environment_evaluations
total_estimated_flops
wall_clock_seconds
peak_memory_bytes
```

The accounting implementation must be architecture-independent at the experiment layer and architecture-specific only inside model accounting adapters.
