from __future__ import annotations

import numpy as np

from ..track import Track


def _catmull_rom_closed(control: np.ndarray, samples_per_segment: int) -> np.ndarray:
    """Sample a closed Catmull-Rom spline through 2D control points."""
    points = np.asarray(control, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 4:
        raise ValueError("control must have shape (N, 2), N >= 4")
    if samples_per_segment < 2:
        raise ValueError("samples_per_segment must be >= 2")

    chunks = []
    n = len(points)
    for i in range(n):
        p0 = points[(i - 1) % n]
        p1 = points[i]
        p2 = points[(i + 1) % n]
        p3 = points[(i + 2) % n]
        t = np.linspace(0.0, 1.0, samples_per_segment, endpoint=False)[:, None]
        t2 = t * t
        t3 = t2 * t
        segment = 0.5 * (
            2.0 * p1
            + (-p0 + p2) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
        )
        chunks.append(segment)
    return np.vstack(chunks)


def circuit(width: float = 9.0, samples_per_segment: int = 24) -> Track:
    """Return a deterministic, smooth, closed circuit in world metres."""
    # Deliberately varied straights and corners, with no self-intersections.
    control = np.array(
        [
            [-52, -23],
            [-35, -34],
            [-8, -36],
            [15, -32],
            [35, -24],
            [47, -10],
            [45, 5],
            [34, 14],
            [19, 13],
            [8, 5],
            [-3, 4],
            [-13, 13],
            [-17, 25],
            [-31, 29],
            [-43, 21],
            [-49, 7],
            [-43, -3],
            [-29, -7],
            [-19, -1],
            [-13, -11],
            [-25, -19],
            [-42, -17],
        ],
        dtype=np.float64,
    )
    centerline = _catmull_rom_closed(control, samples_per_segment)
    return Track(centerline, width, track_id="reference_style_circuit")
