import numpy as np

from neuroracelab.environment.tracks.selection import build_track


def test_circuit_selection_is_stable():
    first = build_track("circuit", seed=10, width=9.0)
    second = build_track("circuit", seed=99, width=9.0)
    assert first.track_id == second.track_id
    assert np.array_equal(first.centerline, second.centerline)


def test_procedural_track_is_reproducible_for_same_seed():
    first = build_track("procedural", seed=42, width=9.0)
    second = build_track("procedural", seed=42, width=9.0)
    assert first.track_id == second.track_id
    assert np.array_equal(first.centerline, second.centerline)


def test_procedural_track_changes_for_different_seeds():
    first = build_track("procedural", seed=42, width=9.0)
    second = build_track("procedural", seed=43, width=9.0)
    assert first.track_id != second.track_id
    assert not np.array_equal(first.centerline, second.centerline)


def test_oval_selection_is_stable():
    first = build_track("oval", seed=1, width=9.0)
    second = build_track("oval", seed=2, width=9.0)
    assert first.track_id == second.track_id
    assert np.array_equal(first.centerline, second.centerline)
