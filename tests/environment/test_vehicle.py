import numpy as np
from neuroracelab.environment.vehicle import BicycleDynamics, VehicleState

def test_straight_acceleration_is_finite():
    model = BicycleDynamics()
    s = VehicleState(u=5.0)
    for _ in range(100):
        s = model.step(s, 0.0, 0.5, 0.01)
    assert np.isfinite(s.as_array()).all()
    assert s.u > 5.0

def test_zero_steering_has_small_yaw_rate():
    model = BicycleDynamics()
    s = VehicleState(u=10.0)
    for _ in range(100):
        s = model.step(s, 0.0, 0.0, 0.01)
    assert abs(s.r) < 1e-6
