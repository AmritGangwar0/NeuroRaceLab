# NeuroRaceLab Memory

## Fixed decisions
- Research subject: compact neural computational architecture.
- Initial architectures: RNN+Residual, Continuous-Time, Equilibrium.
- Primary computational state: 128 neurons.
- Observation: 32 rays + speed + slip angle + angular velocity + steering = 36 dims.
- Actions: steering and throttle/brake.
- Training: alternating ES saturation -> EA saturation -> repeat.
- Capacity-matched primary budget: environment evaluations.
- Compute-matched primary budget: FLOPs.
- Wall-clock: secondary metric.
- Unseen-track generalization: primary benchmark.
- Hugging Face export: required.

## Unresolved before model implementation
- vehicle dynamics
- exact track representation/generator
- ray geometry/FOV/normalization
- control timestep
- input/output projections
- parameter/FLOP targets
- continuous-time step
- equilibrium iterations
- ES/EA variants and hyperparameters
- saturation criterion
- fitness function
- total budget
- benchmark difficulty protocol

## Current phase
Structural foundation implemented. Resolve remaining environment and optimizer design decisions before implementing the research models.
