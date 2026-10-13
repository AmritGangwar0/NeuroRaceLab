"""Physics sanity tests; these are plausibility checks, not real-vehicle validation."""
import numpy as np
import pytest

from neuroracelab.environment.vehicle import BicycleDynamics, VehicleState


def rollout(model, state, steering, longitudinal, steps=200, dt=0.01):
    for _ in range(steps):
        state = model.step(state, steering, longitudinal, dt)
        assert np.isfinite(state.as_array()).all(), "non-finite vehicle state"
    return state


def test_throttle_increases_speed_from_cruise():
    model = BicycleDynamics()
    initial = VehicleState(u=5.0)
    final = rollout(model, initial, 0.0, 0.5, steps=100)
    assert final.u > initial.u + 1.0
    assert abs(final.v) < 1e-8
    assert abs(final.r) < 1e-8


def test_braking_reduces_speed_and_does_not_reverse():
    model = BicycleDynamics()
    initial = VehicleState(u=10.0)
    final = rollout(model, initial, 0.0, -1.0, steps=200)
    assert final.u < initial.u
    assert final.u >= 0.0
    assert abs(final.v) < 1e-8


def test_positive_and_negative_steering_produce_mirrored_yaw_response():
    model = BicycleDynamics()
    left = rollout(model, VehicleState(u=10.0), 0.3, 0.0)
    right = rollout(model, VehicleState(u=10.0), -0.3, 0.0)
    assert left.r > 0.1
    assert right.r < -0.1
    assert left.psi > 0.0
    assert right.psi < 0.0
    assert left.r == pytest.approx(-right.r, rel=1e-5, abs=1e-6)
    assert left.y == pytest.approx(-right.y, rel=1e-5, abs=1e-6)


def test_steering_actuator_respects_rate_limit_and_target_bounds():
    model = BicycleDynamics()
    state = VehicleState(u=5.0)
    one_step = model.step(state, 5.0, 0.0, 0.01)  # out-of-range command is clipped
    assert one_step.steering == pytest.approx(model.p.steering_rate_rad_s * 0.01)
    state = rollout(model, state, 5.0, 0.0, steps=200)
    assert abs(state.steering) <= model.p.max_steering_rad + 1e-12


def test_zero_steering_stays_straight_under_symmetric_conditions():
    model = BicycleDynamics()
    final = rollout(model, VehicleState(u=15.0), 0.0, 0.0, steps=500)
    assert abs(final.v) < 1e-8
    assert abs(final.r) < 1e-8
    assert abs(final.psi) < 1e-8
    assert abs(final.y) < 1e-8


def test_large_commands_remain_finite_and_state_is_bounded():
    model = BicycleDynamics()
    state = VehicleState(u=25.0)
    for i in range(1000):
        state = model.step(state, 1.0 if (i // 100) % 2 == 0 else -1.0, 1.0, 0.01)
        assert np.isfinite(state.as_array()).all()
        assert state.u >= 0.0
        assert -np.pi <= state.psi <= np.pi
        assert abs(state.steering) <= model.p.max_steering_rad + 1e-12
