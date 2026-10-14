from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VehicleParameters:
    mass_kg: float = 1200.0
    yaw_inertia_kg_m2: float = 1500.0
    front_axle_m: float = 1.2
    rear_axle_m: float = 1.3
    front_cornering_stiffness_n_per_rad: float = 70_000.0
    rear_cornering_stiffness_n_per_rad: float = 80_000.0
    max_steering_rad: float = 0.45
    max_drive_force_n: float = 5_000.0
    max_brake_force_n: float = 9_000.0
    rolling_resistance: float = 0.015
    drag_coefficient: float = 0.35
    frontal_area_m2: float = 2.0
    air_density_kg_m3: float = 1.225
    friction_coefficient: float = 1.1
    gravity_mps2: float = 9.81
    steering_rate_rad_s: float = 4.0
