import json
from pathlib import Path

from neuroracelab.core import BenchmarkRegime

from .schema import AgentConfig, BudgetConfig, ExperimentConfig, SensorConfig


def load_json(path):
    d=json.loads(Path(path).read_text())
    c=ExperimentConfig(d["name"],int(d["seed"]),AgentConfig(**d["agent"]),SensorConfig(**d["sensors"]),BudgetConfig(BenchmarkRegime(d["budget"]["regime"]),int(d["budget"]["limit"])),d.get("environment",{}),d.get("training",{}),d.get("evaluation",{}))
    c.validate();return c
