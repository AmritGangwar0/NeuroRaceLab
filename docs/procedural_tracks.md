# NeuroRaceLab — Procedural Track Generation

## Status
**Design frozen for v0.1 benchmark implementation.**

## 1. Goal

Generate smooth, closed, non-self-intersecting tracks from a seed while allowing controlled difficulty variation.

## 2. Generation method

Use a polar Fourier-style radial representation:

\[
r(\theta)=r_0\left(1+\sum_{k=1}^{K}a_k\sin(k\theta+\phi_k)\right)
\]

with bounded coefficients.

The resulting closed curve is sampled densely, checked for validity, then converted to an arc-length-resampled centerline.

This provides:

- compact seed representation,
- reproducibility,
- smooth curvature,
- continuous variation between tracks.

## 3. Validity checks

Reject a generated track if:

- self-intersections occur,
- curvature exceeds the configured maximum,
- minimum radius falls below the configured threshold,
- width causes boundary overlap,
- track length is outside the configured range.

Regenerate using a deterministic child seed.

## 4. Seed hierarchy

Every generated track has:

```text
experiment_seed
    └── split into
        ├── training track seeds
        ├── validation track seeds
        └── test track seeds
```

A test seed is never generated from or reused by the training set.

## 5. Dataset split

The benchmark uses:

- training: procedural tracks
- validation: held-out procedural tracks
- testing: unseen procedural tracks + predefined tracks

The test set is immutable for a given benchmark version.

## 6. Track identity

Every track receives a deterministic ID derived from:

- generator version
- seed
- generator configuration

The ID must be stored in experiment metadata.
