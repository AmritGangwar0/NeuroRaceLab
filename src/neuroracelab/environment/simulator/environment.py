from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..sensors.raycast import RaycastSensor
from ..tracks.predefined.basic import oval
from ..tracks.track import Track
from ..vehicle.dynamics import BicycleDynamics
from ..vehicle.parameters import VehicleParameters
from ..vehicle.state import VehicleState


@dataclass
class EnvironmentConfig:
    physics_dt_s: float = 0.01
    control_dt_s: float = 0.05
    max_duration_s: float = 120.0
    initial_speed_mps: float = 5.0
    speed_reference_mps: float = 30.0
    slip_angle_reference_rad: float = 0.5
    yaw_rate_reference_rad_s: float = 2.5
    off_track_grace_s: float = 0.25
    crash_speed_mps: float = 45.0

class RaceEnvironment:
    def __init__(self, track: Track | None = None, config: EnvironmentConfig | None = None, seed: int = 0):
        self.config = config or EnvironmentConfig()
        self.track = track or oval()
        self.vehicle = BicycleDynamics(VehicleParameters())
        self.sensor = RaycastSensor(32, 240.0, 30.0)
        self.rng = np.random.default_rng(seed)
        self.seed = int(seed)
        self.state = VehicleState()
        self.elapsed_s = 0.0
        self.last_s = 0.0
        self.progress = 0.0
        self.off_track_s = 0.0

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.seed = int(seed); self.rng = np.random.default_rng(self.seed)
        start = self.track.point_at_s(0.0)
        heading = self.track.heading_at_s(0.0)
        self.state = VehicleState(float(start[0]), float(start[1]), heading, self.config.initial_speed_mps, 0.0, 0.0, 0.0)
        self.elapsed_s = self.last_s = self.progress = self.off_track_s = 0.0
        return self._observation()

    def _observation(self) -> np.ndarray:
        s = self.track.project(self.state.x, self.state.y)
        rays = self.sensor.observe(self.state.x, self.state.y, self.state.psi, self.track)
        speed = np.clip(self.state.u / self.config.speed_reference_mps, 0.0, 1.0)
        beta = math.atan2(self.state.v, max(abs(self.state.u), 1e-6))
        beta_n = np.clip(beta / self.config.slip_angle_reference_rad, -1.0, 1.0)
        yaw_n = np.clip(self.state.r / self.config.yaw_rate_reference_rad_s, -1.0, 1.0)
        steer_n = np.clip(self.state.steering / self.vehicle.p.max_steering_rad, -1.0, 1.0)
        obs = np.concatenate((rays, [speed, beta_n, yaw_n, steer_n])).astype(np.float32)
        assert obs.shape == (36,)
        return obs

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (2,): raise ValueError("action must have shape (2,)")
        action = np.clip(action, -1.0, 1.0)
        old_proj = self.track.project(self.state.x, self.state.y)
        for _ in range(round(self.config.control_dt_s / self.config.physics_dt_s)):
            self.state = self.vehicle.step(self.state, float(action[0]), float(action[1]), self.config.physics_dt_s)
        self.elapsed_s += self.config.control_dt_s
        proj = self.track.project(self.state.x, self.state.y)
        ds = (proj.s - old_proj.s) % self.track.length
        if ds > self.track.length * 0.5: ds = 0.0
        self.progress += ds / self.track.length
        half_width = self.track.width_at_s(proj.s) / 2.0
        off_track = abs(proj.lateral_error) > half_width
        self.off_track_s = self.off_track_s + self.config.control_dt_s if off_track else 0.0
        terminated = self.off_track_s >= self.config.off_track_grace_s or self.state.u >= self.config.crash_speed_mps
        truncated = self.elapsed_s >= self.config.max_duration_s
        completed = self.progress >= 1.0
        terminated = bool(terminated or completed)
        reward = float(ds / self.track.length * 100.0 - 0.02 * abs(proj.lateral_error))
        if completed: reward += 100.0
        if off_track: reward -= 5.0
        info = {"progress": float(self.progress), "track_s": float(proj.s), "lateral_error": float(proj.lateral_error), "curvature": float(proj.curvature), "elapsed_s": float(self.elapsed_s), "termination_reason": "lap_complete" if completed else ("off_track" if self.off_track_s >= self.config.off_track_grace_s else ("speed_limit" if self.state.u >= self.config.crash_speed_mps else ("time_limit" if truncated else None)))}
        return self._observation(), reward, terminated, truncated, info
