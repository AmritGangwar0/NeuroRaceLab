from __future__ import annotations

import numpy as np

from ..track import Track


def oval(radius_x: float = 45.0, radius_y: float = 25.0, width: float = 8.0, points: int = 256) -> Track:
    t = np.linspace(0, 2*np.pi, points, endpoint=False)
    pts = np.column_stack((radius_x*np.cos(t), radius_y*np.sin(t)))
    return Track(pts, width, track_id=f"oval_{radius_x:g}x{radius_y:g}")
