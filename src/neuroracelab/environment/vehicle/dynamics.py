from __future__ import annotations

import math

import numpy as np

from .parameters import VehicleParameters
from .state import VehicleState


class BicycleDynamics:
    def __init__(self, params: VehicleParameters | None = None):
        self.p = params or VehicleParameters()
        self._u_eps = 0.5

    def derivative(self, state: VehicleState, longitudinal: float) -> np.ndarray:
        p = self.p
        u = max(state.u, self._u_eps)
        delta = state.steering
        alpha_f = delta - math.atan2(state.v + p.front_axle_m * state.r, u)
        alpha_r = -math.atan2(state.v - p.rear_axle_m * state.r, u)

        fz_f = p.mass_kg * p.gravity_mps2 * p.rear_axle_m / (p.front_axle_m + p.rear_axle_m)
        fz_r = p.mass_kg * p.gravity_mps2 * p.front_axle_m / (p.front_axle_m + p.rear_axle_m)
        fy_f_max = p.friction_coefficient * fz_f
        fy_r_max = p.friction_coefficient * fz_r
        fy_f = float(np.clip(p.front_cornering_stiffness_n_per_rad * alpha_f, -fy_f_max, fy_f_max))
        fy_r = float(np.clip(p.rear_cornering_stiffness_n_per_rad * alpha_r, -fy_r_max, fy_r_max))

        q = float(np.clip(longitudinal, -1.0, 1.0))
        fx = p.max_drive_force_n * q if q >= 0.0 else p.max_brake_force_n * q
        drag = 0.5 * p.air_density_kg_m3 * p.drag_coefficient * p.frontal_area_m2 * state.u * abs(state.u)
        rolling = p.rolling_resistance * p.mass_kg * p.gravity_mps2 * (1.0 if state.u >= 0 else -1.0)

        du = (fx - drag - rolling) / p.mass_kg + state.v * state.r
        dv = (fy_f + fy_r) / p.mass_kg - state.u * state.r
        dr = (p.front_axle_m * fy_f - p.rear_axle_m * fy_r) / p.yaw_inertia_kg_m2
        dx = state.u * math.cos(state.psi) - state.v * math.sin(state.psi)
        dy = state.u * math.sin(state.psi) + state.v * math.cos(state.psi)
        dpsi = state.r
        return np.array([dx, dy, dpsi, du, dv, dr, 0.0], dtype=np.float64)

    def step(self, state: VehicleState, steering: float, longitudinal: float, dt: float) -> VehicleState:
        p = self.p
        target = float(np.clip(steering, -1.0, 1.0)) * p.max_steering_rad
        max_change = p.steering_rate_rad_s * dt
        delta = float(np.clip(target, state.steering - max_change, state.steering + max_change))
        s0 = state.as_array(); s0[6] = delta

        def f(arr: np.ndarray) -> np.ndarray:
            return self.derivative(VehicleState.from_array(arr), longitudinal)
        k1 = f(s0)
        k2 = f(s0 + 0.5 * dt * k1)
        k3 = f(s0 + 0.5 * dt * k2)
        k4 = f(s0 + dt * k3)
        out = s0 + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        out[6] = delta
        out[3] = max(0.0, out[3])
        out[2] = math.atan2(math.sin(out[2]), math.cos(out[2]))
        return VehicleState.from_array(out)
