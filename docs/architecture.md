# NeuroRaceLab Architecture

## Principle
Separate the experimental variable from the experimental infrastructure.

```text
core interfaces
      |
environment <---- tracks / sensors / vehicle
      |
agents <---- architecture implementations
      |
training <---- ES / EA / alternating
      |
evaluation / benchmarks
      |
experiments / accounting / serialization
```

## Agent interface
reset(), act(observation), get_state(), set_state(), parameters(), load_parameters(), metadata().

## Environment interface
reset(seed), step(action), observation(), terminated(), truncated(), info().

## Primary state
The initial comparison uses 128 persistent computational neurons. Projection parameters are separately counted.

## Architectures
RNN+Residual: h_(t+1) = h_t + F_theta(h_t,x_t)

Continuous-Time: dh/dt = F_theta(h,x), initially with bounded fixed-step integration.

Equilibrium: h* = F_theta(h*,x), initially with bounded iterative approximation.

## Registries
Architectures, environments and trainers are registered by stable identifiers. No simulator code may depend on a specific architecture.

## RNG
A master seed creates deterministic named child streams for tracks, optimization and evaluation.

## Accounting
Always record environment evaluations, FLOPs, wall-clock time and optimizer phases.
Budget exhaustion uses evaluations in capacity mode and FLOPs in compute mode.

## Evaluation
A top-level evaluator independently runs validation, test, robustness and efficiency suites.
Test-track resources cannot enter training APIs.

## Configuration
Validated experiment configuration is persisted with each run.

## Serialization
Checkpoints and exported models are self-describing and versioned.
