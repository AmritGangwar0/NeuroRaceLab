import numpy as np
import pytest

from neuroracelab.agents.baseline import BaselineController


def test_baseline_returns_valid_action():
    agent = BaselineController()
    action = agent.act(np.ones(36, dtype=np.float32) * 0.5)

    assert action.shape == (2,)
    assert np.all(np.isfinite(action))
    assert np.all(action >= -1.0)
    assert np.all(action <= 1.0)


def test_baseline_is_deterministic():
    agent = BaselineController()
    observation = np.linspace(0.2, 0.9, 36)

    np.testing.assert_array_equal(
        agent.act(observation),
        agent.act(observation),
    )


def test_baseline_rejects_wrong_observation_shape():
    agent = BaselineController()

    with pytest.raises(ValueError):
        agent.act(np.zeros(35))


def test_baseline_has_no_trainable_parameters():
    agent = BaselineController()

    assert agent.parameters() == ()
    assert agent.get_state() == ()
    assert agent.metadata()["trainable"] is False
