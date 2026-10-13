import numpy as np
from neuroracelab.environment.sensors import RaycastSensor
from neuroracelab.environment.tracks.predefined import oval

def test_raycast_output():
    track = oval()
    sensor = RaycastSensor()
    p = track.point_at_s(0.0)
    h = track.heading_at_s(0.0)
    obs = sensor.observe(p[0], p[1], h, track)
    assert obs.shape == (32,)
    assert np.isfinite(obs).all()
    assert np.all((obs >= 0) & (obs <= 1))
