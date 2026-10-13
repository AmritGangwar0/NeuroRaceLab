from __future__ import annotations
import numpy as np
from neuroracelab.environment import RaceEnvironment
from neuroracelab.environment.tracks.procedural import ProceduralTrackGenerator

track = ProceduralTrackGenerator(seed=42).generate()
env = RaceEnvironment(track=track, seed=42)
obs = env.reset(seed=42)
assert obs.shape == (36,)

for _ in range(100):
    obs, reward, terminated, truncated, info = env.step(np.array([0.0, 0.5]))
    assert obs.shape == (36,)
    assert np.isfinite(obs).all()
    if terminated or truncated:
        break

print("Environment smoke test passed")
print(f"track={track.track_id}")
print(f"track_length_m={track.length:.3f}")
print(f"progress={info['progress']:.6f}")
print(f"elapsed_s={info['elapsed_s']:.3f}")
