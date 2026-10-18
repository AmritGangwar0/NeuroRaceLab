# Architecture

## Design goals

NeuroRaceLab separates the environment, controller, training algorithm, evaluation, and visualization so that architecture experiments do not require rewriting the simulator.

Core goals:

- Keep controller interfaces consistent.
- Make tracks and random seeds configurable.
- Evaluate models in the same environment.
- Separate training results from held-out evaluation.
- Keep expensive model/runtime integrations optional.

## System layers

### Environment

The environment owns the simulation step, vehicle state, observation construction, reward calculation, and episode termination. Its interface follows the Gymnasium-like pattern:

```python
observation = env.reset(seed=seed)
observation, reward, terminated, truncated, info = env.step(action)
```

The current implementation's `reset` returns the observation directly, rather than the `(observation, info)` tuple used by Gymnasium. Do not assume it is a drop-in Gymnasium environment.

Relevant module:

- `src/neuroracelab/environment/simulator/environment.py`

The environment is stepped at a physics timestep of approximately 0.01 seconds and the controller is called at approximately 0.05 seconds in the current configuration. Confirm `EnvironmentConfig` in the checked-out version before reporting these values as permanent defaults.

### Vehicle dynamics

A lightweight planar dynamic bicycle model advances the vehicle state. It is intended for controlled experiments, not high-fidelity vehicle engineering.

### Tracks and sensors

Track selection includes predefined oval and circuit layouts plus seeded procedural generation. Ray distances provide local geometric information to the controller.

Relevant modules include:

- `src/neuroracelab/environment/tracks/track.py`
- `src/neuroracelab/environment/tracks/selection.py`
- `src/neuroracelab/environment/tracks/procedural/generator.py`

### Observation and action

Current observation size: 36 values.

- 32 ray distances
- speed
- slip angle
- yaw rate
- current steering

Current action size: 2 values.

- steering
- longitudinal throttle/brake

Both action components are bounded to `[-1, 1]`. Confirm the exact ordering and units in the implementation before changing this interface.

### Controllers

The controller interface exposes an action method, reset/state behavior, parameter access, and metadata. The implemented experimental architectures are:

- `residual_rnn`: recurrent residual state update
- `continuous_time`: continuous-time-inspired hidden-state dynamics with bounded numerical integration
- `equilibrium`: bounded fixed-point/equilibrium-style hidden-state update

These are experimental implementations. Their names describe design intent, not a claim that they outperform other architectures.

Relevant module:

- `src/neuroracelab/agents/neural.py`

A deterministic gap-following baseline is also available:

- `src/neuroracelab/agents/baseline.py`

### Training

The training prototype alternates phases in the order:

1. Evolution Strategies (ES)
2. Evolutionary Algorithm (EA)
3. ES
4. EA

The run is bounded by an evaluation budget. The exact phase allocation, population behavior, mutation settings, and fitness function must be read from the current training implementation and run metadata; avoid assuming that phase names alone fully describe the optimizer.

Relevant module:

- `src/neuroracelab/training/evolution.py`

### Evaluation and reports

Evaluation loads saved parameters, runs episodes over configured seeds, and aggregates metrics. Reporting scripts render JSON/CSV summaries and plots.

Relevant scripts:

- `scripts/benchmark_baseline.py`
- `scripts/evaluate_checkpoints.py`
- `scripts/render_report.py`
- `scripts/render_training_report.py`

### GUI and population workbench

The Tkinter GUI manages model files and reports and launches the Pygame population workbench as a subprocess. The Pygame simulation still uses its own window. GUI training launch, embedded rendering, live structured telemetry charts, and GGUF inference are not implemented in the current documented scope.

## Extension rules

When adding an architecture:

1. Preserve the common controller interface.
2. Define observation and action assumptions explicitly.
3. Make initialization deterministic from a seed.
4. Expose a flat, numeric parameter vector with stable ordering.
5. Implement parameter loading with shape and finite-value checks.
6. Add unit tests for shape, bounds, deterministic initialization, reset behavior, and parameter round-trip.
7. Add the architecture to the factory and metadata.
8. Evaluate it using the same seeds and budget as comparison architectures.

When adding a track:

1. Use deterministic generation from the supplied seed.
2. Keep training/validation/test track seeds disjoint.
3. Test geometry validity and sensor behavior.
4. Add a visualization or regression test where appropriate.

## Data flow

```text
Track geometry
    ↓
Ray sensor + vehicle state
    ↓
36-value observation
    ↓
Controller (baseline or neural architecture)
    ↓
2-value action
    ↓
Vehicle dynamics + reward + termination
    ↓
Episode metrics
    ↓
Training history / checkpoint / evaluation report
```
