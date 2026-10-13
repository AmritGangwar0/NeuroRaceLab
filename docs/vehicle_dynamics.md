# NeuroRaceLab — Vehicle Dynamics Specification

## Status
**Design frozen for v0.1 benchmark implementation.**

## 1. Purpose

The vehicle model must be lightweight, deterministic, numerically stable, and expressive enough to produce meaningful steering, slip, and angular-dynamics behavior.

The racecar is a controlled benchmark platform. Physical realism beyond what is required to produce a useful dynamical-control problem is explicitly out of scope for v0.1.

## 2. Model

Use a planar dynamic bicycle model with a single front steering input and a single longitudinal force input.

State:

\[
s = [x,\ y,\ \psi,\ u,\ v,\ r]
\]

where:

- `x, y`: world position
- `psi`: vehicle heading
- `u`: longitudinal body-frame velocity
- `v`: lateral body-frame velocity
- `r`: yaw rate

Derived sideslip/slip-angle state:

\[
\beta = \operatorname{atan2}(v,\max(|u|,u_\epsilon))
\]

with a small positive `u_epsilon` for numerical stability.

## 3. Parameters

Initial benchmark parameters:

| Parameter | Value |
|---|---:|
| mass `m` | 1200 kg |
| yaw inertia `Iz` | 1500 kg m² |
| CG → front axle `a` | 1.2 m |
| CG → rear axle `b` | 1.3 m |
| wheelbase `L` | 2.5 m |
| front cornering stiffness `Cf` | 70,000 N/rad |
| rear cornering stiffness `Cr` | 80,000 N/rad |
| maximum steering | ±0.45 rad |
| maximum drive force | 5,000 N |
| maximum braking force | 9,000 N |
| rolling resistance coefficient | 0.015 |
| aerodynamic drag coefficient | 0.35 |
| reference frontal area | 2.0 m² |
| air density | 1.225 kg/m³ |

The values are benchmark parameters, not claims of representing a particular production vehicle.

## 4. Tire model

Use a linear lateral tire model with force saturation:

\[
F_y = \operatorname{clip}(-C_\alpha\alpha,\ -F_{y,\max},\ F_{y,\max})
\]

Front slip angle:

\[
\alpha_f = \delta -
\operatorname{atan2}(v+a r,\max(u,u_\epsilon))
\]

Rear slip angle:

\[
\alpha_r =
-\operatorname{atan2}(v-b r,\max(u,u_\epsilon))
\]

The maximum lateral force is derived from the axle's nominal load and a configurable friction coefficient.

No Pacejka/Magic Formula is used in v0.1. It would add parameters and nonlinearities before they are scientifically justified.

## 5. Longitudinal dynamics

The longitudinal force command is:

\[
F_x =
\begin{cases}
F_{\max,drive} q, & q \ge 0\\
F_{\max,brake} q, & q < 0
\end{cases}
\]

where `q ∈ [-1, 1]` is the longitudinal action.

Include rolling resistance and quadratic aerodynamic drag:

\[
F_{drag} = \frac12 \rho C_d A u|u|
\]

The longitudinal acceleration is:

\[
\dot u = \frac{F_x - F_{drag} - F_{roll}}{m} + vr
\]

## 6. Lateral/yaw dynamics

\[
\dot v =
\frac{F_{y,f}+F_{y,r}}{m} - ur
\]

\[
\dot r =
\frac{aF_{y,f}-bF_{y,r}}{I_z}
\]

World-frame motion:

\[
\dot x=u\cos\psi-v\sin\psi
\]

\[
\dot y=u\sin\psi+v\cos\psi
\]

\[
\dot\psi=r
\]

## 7. Numerical integration

Use fixed-step RK4 integration.

Default physics timestep:

`dt = 0.01 s` (100 Hz)

The controller observes and acts at:

`control_dt = 0.05 s` (20 Hz)

Thus each control step performs five physics substeps.

This gives stable vehicle dynamics without requiring an expensive general-purpose ODE solver.

## 8. Termination

An episode terminates when:

- the vehicle crashes,
- the vehicle leaves the valid track corridor for longer than the configured grace period,
- the vehicle completes the required lap count,
- the maximum episode duration is reached.

A termination reason must always be recorded.

## 9. Determinism

Given:

- identical environment configuration,
- identical track,
- identical initial state,
- identical action sequence,

the simulator must produce identical state trajectories within the defined floating-point tolerance.

Randomness must never be obtained from an implicit global RNG.

## 10. Future extensions

The interface should allow later replacement with:

- nonlinear tire models,
- tire temperature,
- load transfer,
- actuator delay,
- drivetrain dynamics,
- model uncertainty.

These are not part of v0.1.
