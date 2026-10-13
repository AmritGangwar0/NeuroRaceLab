# NeuroRaceLab — Track Specification

## Status
**Design frozen for v0.1 benchmark implementation.**

## 1. Representation

A track is a closed centerline represented by an ordered cyclic polyline:

```text
Track
├── centerline points [N, 2]
├── width profile [N]
├── cumulative arc length [N]
├── total length
└── metadata
```

The centerline is resampled at a fixed spatial interval:

`ds = 1.0 m`

Track width is allowed to vary but is constant by default.

## 2. Coordinate system

World coordinates are Cartesian metres.

The centerline parameter is arc length `s ∈ [0, L)`.

All track calculations use cyclic indexing.

## 3. Vehicle-track relationship

At each simulation step calculate the nearest centerline segment and:

- longitudinal progress `s`
- signed lateral error `e_y`
- local centerline heading
- local curvature estimate
- distance to left boundary
- distance to right boundary

Progress must be continuous across the start/finish seam.

## 4. Valid corridor

The vehicle is inside the valid corridor when:

\[
|e_y| \leq w(s)/2
\]

with a configurable tolerance.

The track boundary is geometric, not a texture or pixel mask.

## 5. Progress

Progress is measured using centerline arc length.

For a closed track:

\[
progress = \frac{s_{current}-s_{start}\pmod L}{L}
\]

A lap is completed when progress crosses the start/finish region in the forward direction after exceeding a minimum progress threshold.

## 6. Track families

The framework supports:

- predefined tracks
- procedural tracks
- user-supplied tracks

All three use the same `Track` interface.

## 7. v0.1 benchmark tracks

Initial predefined validation tracks should include:

1. **Oval** — baseline stability
2. **Figure Eight** — directional transitions
3. **Technical Loop** — tighter curvature variation

Training tracks will primarily be procedurally generated.

Test tracks must not be reused as training seeds.

## 8. Difficulty dimensions

Track difficulty is controlled independently through:

- curvature magnitude
- curvature variation
- minimum radius
- width
- length
- number of direction changes

Difficulty must be recorded numerically rather than only labelled "easy/medium/hard".
