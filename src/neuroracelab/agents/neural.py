"""NumPy-only neural controllers for NeuroRaceLab; training is implemented separately."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from neuroracelab.core.interfaces import Agent

OBSERVATION_SIZE = 36
ACTION_SIZE = 2
DEFAULT_HIDDEN_SIZE = 128
ARCHITECTURES = ("residual_rnn", "continuous_time", "equilibrium")

def _obs(value: Any) -> np.ndarray:
    value = np.asarray(getattr(value, "values", value), dtype=np.float64)
    if value.shape != (OBSERVATION_SIZE,):
        raise ValueError(f"Expected observation shape ({OBSERVATION_SIZE},), got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError("Observation contains non-finite values")
    return value

class NeuralController(Agent):
    architecture = "base"
    def __init__(self, hidden_size: int = DEFAULT_HIDDEN_SIZE, seed: int = 0) -> None:
        if hidden_size < 1: raise ValueError("hidden_size must be positive")
        self.hidden_size, self.seed = int(hidden_size), int(seed)
        self._rng = np.random.default_rng(self.seed)
        self._parameters: dict[str, np.ndarray] = {}
        self._initialize_parameters()
        self._state = np.zeros(self.hidden_size, dtype=np.float64)
    def _weight(self, shape: tuple[int, ...], fan_in: int) -> np.ndarray:
        return self._rng.normal(0.0, 1.0 / np.sqrt(max(1, fan_in)), size=shape)
    def _initialize_parameters(self) -> None: raise NotImplementedError
    def reset(self) -> None: self._state.fill(0.0)
    def get_state(self) -> Sequence[float]: return self._state.copy()
    def set_state(self, state: Sequence[float]) -> None:
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (self.hidden_size,) or not np.all(np.isfinite(state)):
            raise ValueError(f"State must be a finite vector of shape ({self.hidden_size},)")
        self._state = state.copy()
    def parameters(self) -> Sequence[float]:
        return np.concatenate([v.ravel() for v in self._parameters.values()]).copy()
    def load_parameters(self, parameters: Sequence[float]) -> None:
        flat = np.asarray(parameters, dtype=np.float64)
        expected = sum(v.size for v in self._parameters.values())
        if flat.shape != (expected,): raise ValueError(f"Expected {expected} parameters, got shape {flat.shape}")
        if not np.all(np.isfinite(flat)): raise ValueError("Parameters contain non-finite values")
        offset = 0
        for name, value in self._parameters.items():
            size = value.size
            self._parameters[name] = flat[offset:offset+size].reshape(value.shape).copy()
            offset += size
    @property
    def parameter_count(self) -> int: return sum(v.size for v in self._parameters.values())
    def _output(self) -> np.ndarray:
        p = self._parameters
        return np.tanh(p["w_out"] @ self._state + p["b_out"])
    def metadata(self) -> Mapping[str, Any]:
        return {"name": self.architecture, "architecture": self.architecture, "trainable": True,
                "observation_size": OBSERVATION_SIZE, "action_size": ACTION_SIZE,
                "hidden_size": self.hidden_size, "parameter_count": self.parameter_count,
                "initialization_seed": self.seed}

class ResidualRNNController(NeuralController):
    """Discrete recurrent network with a residual hidden-state update."""
    architecture = "residual_rnn"
    def _initialize_parameters(self) -> None:
        n = self.hidden_size
        self._parameters = {"w_in": self._weight((n, OBSERVATION_SIZE), OBSERVATION_SIZE), "b_in": np.zeros(n),
            "w_rec": self._weight((n, n), n), "b_rec": np.zeros(n),
            "w_out": self._weight((ACTION_SIZE, n), n), "b_out": np.zeros(ACTION_SIZE)}
    def act(self, observation: Sequence[float]) -> np.ndarray:
        x, p = _obs(observation), self._parameters
        drive = np.tanh(p["w_in"] @ x + p["b_in"])
        recurrent = np.tanh(p["w_rec"] @ self._state + p["b_rec"])
        self._state = np.tanh(self._state + 0.25 * (drive + recurrent))
        return self._output()

class ContinuousTimeController(NeuralController):
    """Fixed-step integration of a continuous-time hidden-state equation."""
    architecture = "continuous_time"
    def __init__(self, hidden_size: int = DEFAULT_HIDDEN_SIZE, seed: int = 0,
                 integration_steps: int = 4, step_size: float = 0.25) -> None:
        if integration_steps < 1: raise ValueError("integration_steps must be positive")
        if not 0.0 < step_size <= 1.0: raise ValueError("step_size must be in (0, 1]")
        self.integration_steps, self.step_size = int(integration_steps), float(step_size)
        super().__init__(hidden_size, seed)
    def _initialize_parameters(self) -> None:
        n = self.hidden_size
        self._parameters = {"w_in": self._weight((n, OBSERVATION_SIZE), OBSERVATION_SIZE), "b_in": np.zeros(n),
            "w_dyn": self._weight((n, n), n), "b_dyn": np.zeros(n),
            "w_out": self._weight((ACTION_SIZE, n), n), "b_out": np.zeros(ACTION_SIZE)}
    def act(self, observation: Sequence[float]) -> np.ndarray:
        x, p = _obs(observation), self._parameters
        drive = p["w_in"] @ x + p["b_in"]
        h = self._state
        for _ in range(self.integration_steps):
            h = h + self.step_size * (-h + np.tanh(p["w_dyn"] @ h + drive + p["b_dyn"]))
        self._state = h
        return self._output()
    def metadata(self) -> Mapping[str, Any]:
        return {**super().metadata(), "integration_steps": self.integration_steps, "step_size": self.step_size}

class EquilibriumController(NeuralController):
    """Bounded damped fixed-point approximation to an equilibrium network."""
    architecture = "equilibrium"
    def __init__(self, hidden_size: int = DEFAULT_HIDDEN_SIZE, seed: int = 0,
                 fixed_point_iterations: int = 8, damping: float = 0.5) -> None:
        if fixed_point_iterations < 1: raise ValueError("fixed_point_iterations must be positive")
        if not 0.0 < damping <= 1.0: raise ValueError("damping must be in (0, 1]")
        self.fixed_point_iterations, self.damping = int(fixed_point_iterations), float(damping)
        super().__init__(hidden_size, seed)
    def _initialize_parameters(self) -> None:
        n = self.hidden_size
        self._parameters = {"w_in": self._weight((n, OBSERVATION_SIZE), OBSERVATION_SIZE), "b_in": np.zeros(n),
            "w_eq": self._weight((n, n), n) * 0.5, "b_eq": np.zeros(n),
            "w_out": self._weight((ACTION_SIZE, n), n), "b_out": np.zeros(ACTION_SIZE)}
    def act(self, observation: Sequence[float]) -> np.ndarray:
        x, p = _obs(observation), self._parameters
        drive = p["w_in"] @ x + p["b_in"]
        h = self._state
        for _ in range(self.fixed_point_iterations):
            target = np.tanh(p["w_eq"] @ h + drive + p["b_eq"])
            h = (1.0 - self.damping) * h + self.damping * target
        self._state = h
        return self._output()
    def metadata(self) -> Mapping[str, Any]:
        return {**super().metadata(), "fixed_point_iterations": self.fixed_point_iterations, "damping": self.damping}

def build_neural_controller(architecture: str, hidden_size: int = DEFAULT_HIDDEN_SIZE,
                            seed: int = 0) -> NeuralController:
    """Build a controller by stable architecture identifier."""
    constructors = {"residual_rnn": ResidualRNNController, "continuous_time": ContinuousTimeController,
                    "equilibrium": EquilibriumController}
    if architecture not in constructors:
        raise ValueError(f"Unknown architecture {architecture!r}; choose from {ARCHITECTURES}")
    return constructors[architecture](hidden_size=hidden_size, seed=seed)
