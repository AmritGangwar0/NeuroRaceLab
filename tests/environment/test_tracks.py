import numpy as np

from neuroracelab.environment.tracks.predefined import oval
from neuroracelab.environment.tracks.procedural import ProceduralTrackGenerator


def test_oval_projection_and_wrap():
    track = oval()
    p = track.point_at_s(track.length + 2.0)
    proj = track.project(*p)
    assert abs(proj.s - 2.0) < 1e-6
    assert abs(proj.lateral_error) < 1e-6

def test_procedural_track_reproducibility():
    a = ProceduralTrackGenerator(99).generate()
    b = ProceduralTrackGenerator(99).generate()
    np.testing.assert_allclose(a.centerline, b.centerline)
    assert a.track_id == b.track_id

def test_procedural_track_varies_by_seed():
    a = ProceduralTrackGenerator(99).generate()
    b = ProceduralTrackGenerator(100).generate()
    assert not np.allclose(a.centerline, b.centerline)
