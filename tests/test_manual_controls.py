from neuroracelab.visualization.controls import keys_to_action


def test_manual_controls_neutral():
    assert keys_to_action(left=False, right=False, throttle=False, brake=False) == (0.0, 0.0)


def test_manual_controls_steering_and_longitudinal():
    assert keys_to_action(left=True, right=False, throttle=True, brake=False) == (-1.0, 1.0)
    assert keys_to_action(left=False, right=True, throttle=False, brake=True) == (1.0, -1.0)


def test_opposing_keys_cancel():
    assert keys_to_action(left=True, right=True, throttle=True, brake=True) == (0.0, 0.0)
