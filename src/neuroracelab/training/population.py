"""Population evaluation and EA/ES generation operators for NeuroRaceLab.

This module deliberately keeps learning logic independent of Pygame rendering.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass
class Candidate:
    parameters: np.ndarray
    fitness: float = float("-inf")
    progress: float = 0.0
    crashed: bool = False


@dataclass(frozen=True)
class PopulationConfig:
    size: int = 12
    elite_count: int = 2
    mutation_sigma: float = 0.08
    es_sigma: float = 0.05
    es_learning_rate: float = 0.03
    seed: int = 42

    def __post_init__(self):
        if self.size < 4:
            raise ValueError("population size must be at least 4")
        if self.elite_count != 2:
            raise ValueError("EA currently selects exactly the best two parents")
        if self.mutation_sigma <= 0 or self.es_sigma <= 0 or self.es_learning_rate <= 0:
            raise ValueError("mutation and ES scales must be positive")


class PopulationTrainer:
    """EA (best-two parent mutation) and antithetic, fitness-weighted ES."""

    def __init__(self, parameter_count: int, config: PopulationConfig = PopulationConfig()):
        if parameter_count <= 0:
            raise ValueError("parameter_count must be positive")
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.parameter_count = int(parameter_count)
        self.generation = 0
        self.mode = "EA"
        self.population = [
            Candidate(self.rng.normal(0.0, 0.08, self.parameter_count))
            for _ in range(config.size)
        ]
        self.history: list[dict] = []

    def set_population(self, parameters: Sequence[np.ndarray]) -> None:
        if len(parameters) != self.config.size:
            raise ValueError("wrong population size")
        if any(np.asarray(p).shape != (self.parameter_count,) for p in parameters):
            raise ValueError("all candidate parameter vectors must have identical shape")
        self.population = [Candidate(np.asarray(p, dtype=np.float64).copy()) for p in parameters]

    def evolve(self) -> None:
        ranked = sorted(self.population, key=lambda c: c.fitness, reverse=True)
        scores = np.asarray([c.fitness for c in ranked], dtype=np.float64)
        finite = np.isfinite(scores)
        if not finite.all():
            scores = np.where(finite, scores, np.min(scores[finite]) - 1.0 if finite.any() else -1e6)
        self.history.append({
            "generation": self.generation,
            "mode": self.mode,
            "best_fitness": float(scores[0]),
            "mean_fitness": float(np.mean(scores)),
            "best_progress": float(ranked[0].progress),
            "crash_rate": float(np.mean([c.crashed for c in ranked])),
        })
        if self.mode == "EA":
            self._evolve_ea(ranked)
        elif self.mode == "ES":
            self._evolve_es(ranked)
        else:
            raise ValueError(f"unknown training mode: {self.mode}")
        self.generation += 1

    def _evolve_ea(self, ranked: list[Candidate]) -> None:
        # Keep best individual unchanged; use best two as parents for offspring.
        next_pop = [Candidate(ranked[0].parameters.copy())]
        parents = ranked[:2]
        while len(next_pop) < self.config.size:
            parent = parents[int(self.rng.integers(0, 2))]
            child = parent.parameters.copy()
            mask = self.rng.random(self.parameter_count) < 0.15
            child += self.rng.normal(0.0, self.config.mutation_sigma, self.parameter_count) * mask
            next_pop.append(Candidate(child))
        self.population = next_pop

    def _evolve_es(self, ranked: list[Candidate]) -> None:
        # Rank-normalized utilities drive a centered, antithetic ES update.
        utilities = np.linspace(-0.5, 0.5, len(ranked), dtype=np.float64)
        gradient = np.zeros(self.parameter_count, dtype=np.float64)
        pairs = 0
        for i in range(0, len(ranked) - 1, 2):
            plus = ranked[i].parameters
            minus = ranked[i + 1].parameters
            gradient += (utilities[i] - utilities[i + 1]) * (plus - minus)
            pairs += 1
        if pairs:
            gradient /= pairs * max(self.config.es_sigma, 1e-12)
        center = ranked[0].parameters + self.config.es_learning_rate * gradient
        next_pop = [Candidate(center.copy())]
        while len(next_pop) < self.config.size:
            noise = self.rng.normal(0.0, self.config.es_sigma, self.parameter_count)
            next_pop.append(Candidate(center + noise))
        self.population = next_pop

    def switch_mode(self) -> str:
        self.mode = "ES" if self.mode == "EA" else "EA"
        return self.mode


def evaluate_population(
    candidates: Sequence[Candidate],
    make_episode: Callable[[np.ndarray], tuple[float, float, bool]],
) -> list[Candidate]:
    """Evaluate candidates; make_episode returns (fitness, progress, crashed).

    Environments can be stepped in lockstep by a GUI, or evaluated sequentially
    for reproducibility. Rendering must never be called from this function.
    """
    results = []
    for candidate in candidates:
        fitness, progress, crashed = make_episode(candidate.parameters.copy())
        candidate.fitness = float(fitness)
        candidate.progress = float(progress)
        candidate.crashed = bool(crashed)
        results.append(candidate)
    return results
