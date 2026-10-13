# NeuroRaceLab PRD

## Purpose
The racecar is the experimental platform; neural computational architecture is the research subject.

## Initial comparison
- RNN + Residual
- Continuous-Time Neural Dynamical System
- Implicit / Equilibrium Network

## Common interface
- 128 primary computational neurons
- 32 ray sensors
- scalar speed
- slip angle
- angular velocity
- current steering state
- steering action
- throttle/brake action

Observation dimension: 36.

## Training
All three architectures use the identical schedule:
ES until saturation -> EA until saturation -> repeat until the global budget is exhausted.

## Benchmark regimes
### Capacity-matched
128-neuron primary state, comparable capacity, primary budget in environment evaluations.

### Compute-matched
Approximately equal inference compute, with any architecture-specific adjustment recorded,
primary budget in FLOPs.

Wall-clock time is secondary.

## Environment and tracks
Initial simulator: lightweight 2D and cheap enough for repeated evolutionary evaluation.
Support predefined and seedable procedural tracks. Training, validation and test tracks are isolated.

## Evaluation
Record completion, progress, lap time, path deviation, crash rate, unseen-track performance,
parameters, FLOPs/control step, latency, memory, training evaluations, training compute/time,
and robustness under perturbations.

## Reproducibility
Record resolved configuration, seeds, architecture/version, track seeds, training/evaluation settings,
software version, checkpoints and results.

## Hugging Face
Export self-describing model artifacts containing weights, configuration, metadata, benchmark results and a model card.
