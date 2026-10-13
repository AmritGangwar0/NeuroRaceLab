# NeuroRaceLab — Sensor Specification

## Status
**Design frozen for v0.1 benchmark implementation.**

## Observation dimension

The controller receives exactly **36 scalar observations**:

```text
0..31   32 raycast distances
32      normalized speed
33      sideslip/slip angle beta
34      normalized yaw rate
35      current steering state
```

## 1. Ray sensor

32 equally spaced rays over:

`FOV = 240°`

relative to vehicle heading.

Ray angles are:

\[
\theta_i=-120^\circ+i\frac{240^\circ}{31}
\]

for `i = 0 ... 31`.

Each ray returns normalized distance to the first track boundary intersection:

\[
d_i = clip(distance / d_{max},0,1)
\]

Default maximum range:

`d_max = 30 m`

If no boundary is hit within range, return `1.0`.

## 2. Speed

Speed is normalized against:

`v_ref = 30 m/s`

\[
v_n = clip(v/v_{ref},0,1)
\]

## 3. Slip angle

\[
\beta = atan2(v,u)
\]

Normalize to:

\[
\beta_n = clip(\beta/\beta_{max},-1,1)
\]

with:

`beta_max = 0.5 rad`

## 4. Yaw rate

Normalize:

\[
r_n = clip(r/r_{max},-1,1)
\]

with:

`r_max = 2.5 rad/s`

## 5. Current steering

The current steering command is normalized to:

`[-1,1]`

and directly included as the final observation.

This gives the network information about actuator state and makes the observation compatible with future actuator dynamics.

## 6. Observation invariance

Observations contain no absolute world `x/y` coordinates.

This is deliberate. The controller should learn vehicle-track interaction rather than memorize the location of particular training tracks.

## 7. Noise hooks

The sensor interface supports configurable noise, but the default benchmark uses deterministic noiseless observations.

Robustness experiments can independently enable:

- distance noise
- speed noise
- slip-angle noise
- yaw-rate noise
- ray dropout
