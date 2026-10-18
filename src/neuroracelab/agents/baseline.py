"""Deterministic, observation-only baseline controller."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import numpy as np

from neuroracelab.core.interfaces import Agent


class BaselineController(Agent):
    """Gap-following steering with conservative speed and stability control.

    Observation:
        0:32  normalized ray distances, ordered right to left
        32    normalized forward speed
        33    normalized slip angle
        34    normalized yaw rate
        35    normalized current steering

    Action:
        [steering, longitudinal], both in [-1, 1].
        Positive longitudinal accelerates; negative longitudinal brakes.
    """

    def __init__(
        self,
        target_speed: float = 0.45,
        steering_gain: float = 1.15,
        slip_gain: float = 0.18,
        yaw_gain: float = 0.12,
    ) -> None:
        if not 0.0 < target_speed <= 1.0:
            raise ValueError("target_speed must be in (0, 1]")
        self.target_speed = float(target_speed)
        self.steering_gain = float(steering_gain)
        self.slip_gain = float(slip_gain)
        self.yaw_gain = float(yaw_gain)

    def reset(self) -> None:
        """Reset controller state; this baseline is memoryless."""
        return None

    def act(self, observation: Sequence[float]) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float64)
        if obs.shape != (36,):
            raise ValueError(f"Expected observation shape (36,), got {obs.shape}")
        if not np.all(np.isfinite(obs)):
            raise ValueError("Observation contains non-finite values")

        rays = np.clip(obs[:32], 0.0, 1.0)
        speed, slip, yaw, current_steering = obs[32:]

        # Consider the forward-facing sector. Positive ray angles point left.
        angles = np.linspace(-2.0 * math.pi / 3.0,
                             2.0 * math.pi / 3.0, 32)
        sector = np.abs(angles) <= math.radians(85.0)
        sector_indices = np.flatnonzero(sector)

        # Smooth nearby ray readings so a single ray cannot dominate steering.
        scores = np.full(32, -np.inf)
        for i in sector_indices:
            lo = max(0, i - 2)
            hi = min(32, i + 3)
            local_clearance = float(np.mean(rays[lo:hi]))
            angle_penalty = 0.18 * abs(angles[i]) / math.radians(85.0)
            scores[i] = local_clearance - angle_penalty

        target_index = int(np.argmax(scores))
        target_angle = float(angles[target_index])

        steering = (
            self.steering_gain * target_angle / math.radians(55.0)
            - self.slip_gain * float(slip)
            - self.yaw_gain * float(yaw)
            - 0.08 * float(current_steering)
        )
        steering = float(np.clip(steering, -1.0, 1.0))

        # Slow down when the desired steering angle or instability is high.
        turn_demand = min(1.0, abs(target_angle) / math.radians(65.0))
        instability = min(1.0, 0.7 * abs(float(slip)) + 0.3 * abs(float(yaw)))
        speed_target = self.target_speed * (1.0 - 0.55 * turn_demand)
        speed_target *= 1.0 - 0.35 * instability

        speed_error = speed_target - float(speed)
        longitudinal = float(np.clip(1.8 * speed_error, -1.0, 1.0))

        return np.array([steering, longitudinal], dtype=np.float64)

    def get_state(self) -> Sequence[float]:
        return ()

    def set_state(self, state: Sequence[float]) -> None:
        if len(state) != 0:
            raise ValueError("BaselineController has no persistent state")

    def parameters(self) -> Sequence[float]:
        return ()

    def load_parameters(self, parameters: Sequence[float]) -> None:
        if len(parameters) != 0:
            raise ValueError("BaselineController has no trainable parameters")

    def metadata(self) -> Mapping[str, Any]:
        return {
            "name": "deterministic_gap_baseline",
            "trainable": False,
            "observation_size": 36,
            "action_size": 2,
            "target_speed": self.target_speed,
        }
