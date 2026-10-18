
"""Reproducible rollout evaluation for NeuroRaceLab."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from neuroracelab.core import EnvironmentEvaluation, Timer
from neuroracelab.core.types import StepResult


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration shared by all episodes in an evaluation."""

    max_steps: int
    episodes: int
    seed: int
    flops_per_step: int = 0

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if self.episodes < 1:
            raise ValueError("episodes must be at least 1")
        if self.flops_per_step < 0:
            raise ValueError("flops_per_step cannot be negative")


def normalize_step_result(result: Any) -> StepResult:
    """Normalize tuple-based and dataclass-based environment results."""

    if isinstance(result, StepResult):
        return result

    if isinstance(result, tuple) and len(result) == 5:
        observation, reward, terminated, truncated, info = result
        return StepResult(
            observation=observation,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=info,
        )

    raise TypeError(
        "step() must return StepResult or a five-element "
        "(observation, reward, terminated, truncated, info) tuple."
    )


class RolloutEvaluator:
    """Evaluate an agent over reproducible environment episodes."""

    def __init__(
        self,
        environment_factory: Callable[[], Any],
        fitness_fn: Callable[[Sequence[StepResult]], float],
    ) -> None:
        self.environment_factory = environment_factory
        self.fitness_fn = fitness_fn

    def evaluate(
        self,
        agent: Any,
        config: EvaluationConfig,
    ) -> list[EnvironmentEvaluation]:
        evaluations: list[EnvironmentEvaluation] = []

        for episode_index in range(config.episodes):
            env = self.environment_factory()
            observation = env.reset(seed=config.seed + episode_index)
            agent.reset()
            trajectory: list[StepResult] = []

            with Timer() as timer:
                for _ in range(config.max_steps):
                    action = agent.act(observation)
                    result = normalize_step_result(env.step(action))
                    trajectory.append(result)
                    observation = result.observation

                    if result.terminated or result.truncated:
                        break

            evaluations.append(
                EnvironmentEvaluation(
                    f"episode-{episode_index}",
                    float(self.fitness_fn(trajectory)),
                    len(trajectory),
                    1,
                    len(trajectory) * config.flops_per_step,
                    timer.elapsed,
                )
            )

        return evaluations
