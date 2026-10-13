from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

class BenchmarkRegime(str, Enum):
    CAPACITY_MATCHED = "capacity_matched"
    COMPUTE_MATCHED = "compute_matched"

@dataclass(frozen=True)
class Observation:
    values: tuple[float, ...]
    def __len__(self): return len(self.values)

@dataclass(frozen=True)
class Action:
    steering: float
    throttle_brake: float

@dataclass(frozen=True)
class StepResult:
    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class EnvironmentEvaluation:
    evaluation_id: str
    fitness: float
    steps: int
    environment_evaluations: int = 1
    flops: int = 0
    wall_time_seconds: float = 0.0
    info: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class RunAccounting:
    environment_evaluations: int = 0
    flops: int = 0
    wall_time_seconds: float = 0.0
    optimizer_phases: int = 0
    def add(self, other):
        return RunAccounting(self.environment_evaluations+other.environment_evaluations,self.flops+other.flops,self.wall_time_seconds+other.wall_time_seconds,self.optimizer_phases+other.optimizer_phases)
