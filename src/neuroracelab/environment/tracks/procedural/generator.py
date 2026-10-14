from __future__ import annotations

import hashlib

import numpy as np

from ..track import Track


class ProceduralTrackGenerator:
    def __init__(self, seed: int, points: int = 256, width_m: float = 8.0, harmonics: int = 3):
        self.seed = int(seed)
        self.points = int(points)
        self.width_m = float(width_m)
        self.harmonics = int(harmonics)

    def _rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)

    def generate(self) -> Track:
        rng = self._rng()
        t = np.linspace(0.0, 2*np.pi, self.points, endpoint=False)
        base = rng.uniform(35.0, 55.0)
        r = np.full_like(t, base)
        for k in range(1, self.harmonics + 1):
            amp = rng.uniform(0.02, 0.08) / k
            phase = rng.uniform(0.0, 2*np.pi)
            r += base * amp * np.sin(k*t + phase)
        aspect = rng.uniform(0.75, 1.25)
        pts = np.column_stack((r*np.cos(t), r*aspect*np.sin(t)))
        # Smooth, bounded perturbation is intentionally small in v0.1.
        for _ in range(2):
            pts = (np.roll(pts, 1, axis=0) + 2*pts + np.roll(pts, -1, axis=0)) / 4.0
        digest = hashlib.sha1(f"fourier-v1:{self.seed}:{self.points}:{self.width_m}:{self.harmonics}".encode()).hexdigest()[:12]
        return Track(pts, self.width_m, track_id=f"proc_{digest}")
