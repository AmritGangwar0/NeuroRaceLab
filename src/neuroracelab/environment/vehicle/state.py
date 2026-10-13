from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class VehicleState:
    x: float = 0.0
    y: float = 0.0
    psi: float = 0.0
    u: float = 5.0
    v: float = 0.0
    r: float = 0.0
    steering: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.psi, self.u, self.v, self.r, self.steering], dtype=np.float64)

    @classmethod
    def from_array(cls, values: np.ndarray) -> "VehicleState":
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (7,):
            raise ValueError("Vehicle state array must have shape (7,)")
        return cls(*map(float, values))
