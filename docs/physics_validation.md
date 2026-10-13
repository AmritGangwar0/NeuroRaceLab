# NeuroRaceLab — Physics Validation Gate

## Purpose

These are fast sanity checks for the v0.1 bicycle model. Passing them means the implementation behaves consistently with basic qualitative expectations. It does **not** establish high-fidelity agreement with a real vehicle.

## Automated checks

Run:

```bash
pytest tests/environment/test_physics_validation.py -q
```

The checks cover:

- throttle increases speed from a low-speed cruise state;
- braking reduces speed and speed is clamped against reverse motion;
- opposite steering commands produce mirrored yaw/heading responses;
- steering actuator rate and angle limits are respected;
- zero steering preserves straight-line symmetry;
- aggressive alternating inputs do not produce non-finite states or violate state bounds.

## Manual numerical check

Run the same initial state/action sequence at `dt=0.01` and `dt=0.005`. Compare final position, heading, speed, and yaw rate. The smaller step should generally produce a nearby result. A tolerance should be chosen based on the scenario; this is a convergence sanity check, not proof of physical accuracy.

## Known scope limitations

- Linear tire forces with clipping are a simplified model.
- Reverse driving is not supported; longitudinal speed is clamped to non-negative values.
- This validation suite does not compare against measured vehicle telemetry.
- Track boundary and raycast tests are separate from vehicle physics tests.

## Gate before neural training

Do not begin comparative architecture experiments until the physics checks, track/sensor geometry checks, deterministic replay checks, and reward/progress checks pass. Record any changed model equations or parameters in this document and `docs/vehicle_dynamics.md`.
