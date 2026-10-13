import numpy as np
from neuroracelab.environment import RaceEnvironment
from neuroracelab.environment.tracks.procedural import ProceduralTrackGenerator

def test_reset_observation_shape_and_range():
    env = RaceEnvironment(track=ProceduralTrackGenerator(42).generate(), seed=42)
    obs = env.reset()
    assert obs.shape == (36,)
    assert np.all(np.isfinite(obs))
    assert np.all(obs[:32] >= 0) and np.all(obs[:32] <= 1)

def test_deterministic_rollout():
    track = ProceduralTrackGenerator(123).generate()
    actions = [np.array([0.1, 0.5]), np.array([-0.1, 0.4])] * 20
    trajectories = []
    for _ in range(2):
        env = RaceEnvironment(track=track, seed=77)
        obs = env.reset(seed=77)
        states = [env.state.as_array().copy()]
        for a in actions:
            obs, reward, terminated, truncated, info = env.step(a)
            states.append(env.state.as_array().copy())
            if terminated or truncated: break
        trajectories.append(np.stack(states))
    np.testing.assert_allclose(trajectories[0], trajectories[1], rtol=0, atol=1e-12)

def test_step_interface():
    env = RaceEnvironment(seed=1)
    obs = env.reset()
    next_obs, reward, terminated, truncated, info = env.step([0.0, 0.5])
    assert next_obs.shape == (36,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "progress" in info
