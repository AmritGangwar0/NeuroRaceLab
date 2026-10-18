import numpy as np
import pytest

from neuroracelab.agents.neural import (
    ARCHITECTURES,
    OBSERVATION_SIZE,
    ContinuousTimeController,
    EquilibriumController,
    ResidualRNNController,
    build_neural_controller,
)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_action_shape_bounds_and_finiteness(architecture):
    model = build_neural_controller(architecture, hidden_size=16, seed=7)
    action = model.act(np.zeros(OBSERVATION_SIZE))
    assert action.shape == (2,)
    assert np.all(np.isfinite(action))
    assert np.all((-1 <= action) & (action <= 1))

@pytest.mark.parametrize("kind", [ResidualRNNController, ContinuousTimeController, EquilibriumController])
def test_reset_reproduces_first_action(kind):
    model = kind(hidden_size=12, seed=11)
    observation = np.linspace(-0.5, 0.5, OBSERVATION_SIZE)
    first = model.act(observation)
    model.reset()
    np.testing.assert_allclose(first, model.act(observation))

@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_parameter_round_trip(architecture):
    model = build_neural_controller(architecture, hidden_size=10, seed=2)
    params = np.asarray(model.parameters())
    assert params.size == model.parameter_count
    model.load_parameters(params + 0.01)
    np.testing.assert_allclose(model.parameters(), params + 0.01)

@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_invalid_observations_rejected(architecture):
    model = build_neural_controller(architecture, hidden_size=8, seed=1)
    with pytest.raises(ValueError): model.act(np.zeros(OBSERVATION_SIZE - 1))
    bad = np.zeros(OBSERVATION_SIZE); bad[0] = np.nan
    with pytest.raises(ValueError): model.act(bad)

def test_default_hidden_size_is_128():
    for architecture in ARCHITECTURES:
        model = build_neural_controller(architecture, seed=3)
        assert model.metadata()["hidden_size"] == 128
        assert model.metadata()["parameter_count"] > 0

def test_unknown_architecture_rejected():
    with pytest.raises(ValueError, match="Unknown architecture"):
        build_neural_controller("invalid")

def test_numerical_options_validate():
    with pytest.raises(ValueError): ContinuousTimeController(integration_steps=0)
    with pytest.raises(ValueError): EquilibriumController(fixed_point_iterations=0)
