from __future__ import annotations


def keys_to_action(*, left: bool, right: bool, throttle: bool, brake: bool) -> tuple[float, float]:
    """Convert held-key states to normalized (steering, throttle/brake) actions."""
    steering = float(right) - float(left)
    longitudinal = float(throttle) - float(brake)
    return steering, longitudinal
