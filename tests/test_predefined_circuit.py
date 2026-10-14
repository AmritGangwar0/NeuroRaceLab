import numpy as np

from neuroracelab.environment.tracks.predefined.circuit import circuit


def test_circuit_is_closed_and_deterministic():
    a = circuit()
    b = circuit()
    assert a.track_id == "reference_style_circuit"
    assert a.centerline.shape == b.centerline.shape
    assert np.allclose(a.centerline, b.centerline)
    assert len(a.centerline) >= 100
    assert a.length > 100.0


def test_circuit_has_valid_width_and_boundaries():
    track = circuit(width=9.0)
    left, right = track.boundaries()
    assert left.shape == track.centerline.shape
    assert right.shape == track.centerline.shape
    assert np.all(track.width > 0)
    assert np.all(np.isfinite(left))
    assert np.all(np.isfinite(right))
