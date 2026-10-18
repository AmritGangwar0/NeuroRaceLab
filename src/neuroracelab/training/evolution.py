"""Lightweight ES/EA training primitives for NeuroRaceLab controllers."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable
import json
import time
import numpy as np

FitnessFunction = Callable[[np.ndarray, int], float]

@dataclass(frozen=True)
class TrainingConfig:
    generations_per_phase: int = 2
    population_size: int = 8
    sigma: float = 0.03
    learning_rate: float = 0.02
    elite_fraction: float = 0.25
    seed: int = 123
    max_evaluations: int = 1000

    def __post_init__(self):
        if self.generations_per_phase < 1: raise ValueError("generations_per_phase must be positive")
        if self.population_size < 4: raise ValueError("population_size must be at least 4")
        if self.sigma <= 0 or self.learning_rate <= 0: raise ValueError("sigma and learning_rate must be positive")
        if not 0 < self.elite_fraction <= 0.5: raise ValueError("elite_fraction must be in (0, 0.5]")
        if self.max_evaluations < 2: raise ValueError("max_evaluations must be at least 2")

@dataclass
class TrainingResult:
    architecture: str
    best_parameters: np.ndarray
    best_fitness: float
    history: list[dict]
    evaluations: int
    elapsed_seconds: float
    seed: int

    def save(self, output_dir: Path, metadata: dict) -> tuple[Path, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{self.architecture}_seed{self.seed}"
        checkpoint = output_dir / f"{stem}_checkpoint.npz"
        history_path = output_dir / f"{stem}_history.json"
        np.savez_compressed(checkpoint, parameters=self.best_parameters)
        payload = {
            "architecture": self.architecture, "seed": self.seed,
            "best_fitness": self.best_fitness, "evaluations": self.evaluations,
            "elapsed_seconds": self.elapsed_seconds, "metadata": metadata,
            "history": self.history,
        }
        history_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        return checkpoint, history_path

class EvolutionTrainer:
    """Alternates ES -> EA -> ES -> EA, with one common fitness function.

    Each candidate evaluation must reset its controller and use deterministic
    environment seeds. Fitness is maximized. Every evaluated candidate counts
    against the explicit evaluation budget.
    """
    def __init__(self, fitness_fn: FitnessFunction, config: TrainingConfig):
        self.fitness_fn, self.config = fitness_fn, config

    def train(self, initial_parameters: np.ndarray, architecture: str) -> TrainingResult:
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        current = np.asarray(initial_parameters, dtype=np.float64).copy()
        if current.ndim != 1 or not np.all(np.isfinite(current)):
            raise ValueError("initial_parameters must be a finite one-dimensional vector")
        started = time.perf_counter()
        evaluations = 0
        best_params = current.copy()
        best_fitness = -np.inf
        history: list[dict] = []
        phase_order = ("ES", "EA", "ES", "EA")
        generation = 0

        while evaluations < cfg.max_evaluations:
            phase = phase_order[(generation // cfg.generations_per_phase) % len(phase_order)]
            if phase == "ES":
                current, record, used = self._es_generation(current, rng, generation, evaluations)
            else:
                current, record, used = self._ea_generation(current, rng, generation, evaluations)
            evaluations += used
            generation += 1
            record["generation"] = generation
            record["phase"] = phase
            record["cumulative_evaluations"] = evaluations
            record["cumulative_compute_estimate"] = evaluations
            history.append(record)
            if record["generation_best_fitness"] > best_fitness:
                best_fitness = record["generation_best_fitness"]
                best_params = record.pop("generation_best_parameters")
            else:
                record.pop("generation_best_parameters", None)
            if used == 0:
                break
        return TrainingResult(architecture, best_params, float(best_fitness), history,
                              evaluations, time.perf_counter() - started, cfg.seed)

    def _evaluate(self, parameters: np.ndarray, seed: int) -> float:
        value = float(self.fitness_fn(parameters, seed))
        if not np.isfinite(value): return -1e12
        return value

    def _es_generation(self, current, rng, generation, evaluations):
        remaining = self.config.max_evaluations - evaluations
        pairs = min(self.config.population_size // 2, remaining // 2)
        if pairs < 1: return current, {"generation_best_fitness": -1e12, "mean_fitness": -1e12}, 0
        noises = [rng.normal(size=current.size) for _ in range(pairs)]
        scores = []
        candidate_records = []
        for i, noise in enumerate(noises):
            seed = self.config.seed + generation * 100_003 + i
            plus = self._evaluate(current + self.config.sigma * noise, seed)
            minus = self._evaluate(current - self.config.sigma * noise, seed)
            scores.append((plus, minus))
            candidate_records.extend([(plus, current + self.config.sigma * noise),
                                      (minus, current - self.config.sigma * noise)])
        utilities = np.asarray([p - m for p, m in scores], dtype=np.float64)
        utilities = (utilities - utilities.mean()) / (utilities.std() + 1e-8)
        gradient = sum(u * noise for u, noise in zip(utilities, noises)) / (pairs * self.config.sigma)
        updated = current + self.config.learning_rate * gradient
        best_score, best_vector = max(candidate_records, key=lambda pair: pair[0])
        return updated, {"generation_best_fitness": float(best_score),
                         "generation_best_parameters": best_vector.copy(),
                         "mean_fitness": float(np.mean([s for s, _ in candidate_records])),
                         "evaluations_this_generation": 2 * pairs}, 2 * pairs

    def _ea_generation(self, current, rng, generation, evaluations):
        remaining = self.config.max_evaluations - evaluations
        size = min(self.config.population_size, remaining)
        if size < 1: return current, {"generation_best_fitness": -1e12, "mean_fitness": -1e12}, 0
        population = [current.copy()]
        while len(population) < size:
            population.append(current + rng.normal(0, self.config.sigma, size=current.size))
        scored = []
        for i, candidate in enumerate(population):
            score = self._evaluate(candidate, self.config.seed + generation * 100_003 + i)
            scored.append((score, candidate))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        elite_count = max(1, int(np.ceil(size * self.config.elite_fraction)))
        elites = scored[:elite_count]
        parent_a = elites[0][1]
        parent_b = elites[min(1, len(elites) - 1)][1]
        child = 0.5 * parent_a + 0.5 * parent_b
        child += rng.normal(0, self.config.sigma * 0.5, size=current.size)
        return child, {"generation_best_fitness": float(scored[0][0]),
                       "generation_best_parameters": scored[0][1].copy(),
                       "mean_fitness": float(np.mean([score for score, _ in scored])),
                       "evaluations_this_generation": size}, size
