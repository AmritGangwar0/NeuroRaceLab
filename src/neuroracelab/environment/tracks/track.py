from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrackProjection:
    s: float
    lateral_error: float
    heading: float
    curvature: float
    distance: float

@dataclass
class Track:
    centerline: np.ndarray
    width: np.ndarray | float = 8.0
    track_id: str = "track"

    def __post_init__(self):
        pts = np.asarray(self.centerline, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 8:
            raise ValueError("centerline must have shape (N, 2), N >= 8")
        if np.linalg.norm(pts[0] - pts[-1]) < 1e-9:
            pts = pts[:-1]
        self.centerline = pts
        n = len(pts)
        if np.isscalar(self.width):
            self.width = np.full(n, float(self.width), dtype=np.float64)
        else:
            self.width = np.asarray(self.width, dtype=np.float64)
            if self.width.shape != (n,):
                raise ValueError("width must be scalar or shape (N,)")
        if np.any(self.width <= 0):
            raise ValueError("track widths must be positive")
        nxt = np.roll(pts, -1, axis=0)
        seg = nxt - pts
        self._seg = seg
        self._seg_len = np.linalg.norm(seg, axis=1)
        if np.any(self._seg_len <= 1e-9):
            raise ValueError("centerline contains duplicate consecutive points")
        self._s = np.concatenate(([0.0], np.cumsum(self._seg_len[:-1])))
        self.length = float(np.sum(self._seg_len))

    def point_at_s(self, s: float) -> np.ndarray:
        s = float(s) % self.length
        idx = int(np.searchsorted(self._s, s, side="right") - 1) % len(self.centerline)
        local = (s - self._s[idx]) / self._seg_len[idx]
        return self.centerline[idx] + local * self._seg[idx]

    def tangent_at_index(self, i: int) -> np.ndarray:
        t = self._seg[i % len(self._seg)] / self._seg_len[i % len(self._seg)]
        return t

    def heading_at_s(self, s: float) -> float:
        s = float(s) % self.length
        idx = int(np.searchsorted(self._s, s, side="right") - 1) % len(self.centerline)
        t = self.tangent_at_index(idx)
        return float(math.atan2(t[1], t[0]))

    def width_at_s(self, s: float) -> float:
        s = float(s) % self.length
        idx = int(np.searchsorted(self._s, s, side="right") - 1) % len(self.centerline)
        return float(self.width[idx])

    def project(self, x: float, y: float) -> TrackProjection:
        p = np.array([x, y], dtype=np.float64)
        starts = self.centerline
        rel = p[None, :] - starts
        denom = self._seg_len ** 2
        tau = np.clip(np.einsum("ij,ij->i", rel, self._seg) / denom, 0.0, 1.0)
        closest = starts + tau[:, None] * self._seg
        d2 = np.sum((closest - p[None, :]) ** 2, axis=1)
        i = int(np.argmin(d2))
        tangent = self.tangent_at_index(i)
        normal_left = np.array([-tangent[1], tangent[0]])
        error = float(np.dot(p - closest[i], normal_left))
        s = float((self._s[i] + tau[i] * self._seg_len[i]) % self.length)
        curvature = self.curvature_at_index(i)
        return TrackProjection(s, error, math.atan2(tangent[1], tangent[0]), curvature, math.sqrt(float(d2[i])))

    def curvature_at_index(self, i: int) -> float:
        n = len(self.centerline)
        prev = self.centerline[(i - 1) % n]
        cur = self.centerline[i]
        nxt = self.centerline[(i + 1) % n]
        a = cur - prev; b = nxt - cur
        la = np.linalg.norm(a); lb = np.linalg.norm(b)
        if la < 1e-9 or lb < 1e-9:
            return 0.0
        cross = a[0]*b[1] - a[1]*b[0]
        dot = float(np.dot(a, b))
        angle = math.atan2(cross, dot)
        return float(angle / max(0.5*(la+lb), 1e-9))

    def boundaries(self) -> tuple[np.ndarray, np.ndarray]:
        n = len(self.centerline)
        tangents = self._seg / self._seg_len[:, None]
        normals = np.column_stack((-tangents[:,1], tangents[:,0]))
        half = self.width[:, None] / 2.0
        return self.centerline + normals * half, self.centerline - normals * half
