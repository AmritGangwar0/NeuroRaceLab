from __future__ import annotations

import math

import numpy as np

from ..tracks.track import Track


class RaycastSensor:
    def __init__(self, ray_count: int = 32, fov_deg: float = 240.0, max_range_m: float = 30.0):
        if ray_count < 2: raise ValueError("ray_count must be >= 2")
        self.ray_count = int(ray_count); self.fov_rad = math.radians(float(fov_deg)); self.max_range_m = float(max_range_m)
        self.angles = np.linspace(-self.fov_rad/2, self.fov_rad/2, self.ray_count)

    @staticmethod
    def _ray_segment(origin, direction, a, b):
        v = b - a
        den = direction[0]*(-v[1]) - direction[1]*(-v[0])
        if abs(den) < 1e-12: return None
        ao = a - origin
        t = (ao[0]*(-v[1]) - ao[1]*(-v[0])) / den
        u = (direction[0]*ao[1] - direction[1]*ao[0]) / den
        if t >= 0.0 and 0.0 <= u <= 1.0: return float(t)
        return None

    def observe(self, x: float, y: float, heading: float, track: Track) -> np.ndarray:
        origin = np.array([x, y], dtype=np.float64)
        left, right = track.boundaries()
        segments = []
        for boundary in (left, right):
            for i in range(len(boundary)):
                segments.append((boundary[i], boundary[(i+1) % len(boundary)]))
        out = np.ones(self.ray_count, dtype=np.float64)
        for j, a in enumerate(self.angles):
            ang = heading + a
            d = np.array([math.cos(ang), math.sin(ang)])
            best = self.max_range_m
            for p0, p1 in segments:
                hit = self._ray_segment(origin, d, p0, p1)
                if hit is not None and hit < best: best = hit
            out[j] = np.clip(best / self.max_range_m, 0.0, 1.0)
        return out
