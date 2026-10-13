import json
from pathlib import Path
from .schema import ExperimentConfig,AgentConfig,SensorConfig,BudgetConfig
from neuroracelab.core import BenchmarkRegime
def load_json(path):
    d=json.loads(Path(path).read_text())
    c=ExperimentConfig(d["name"],int(d["seed"]),AgentConfig(**d["agent"]),SensorConfig(**d["sensors"]),BudgetConfig(BenchmarkRegime(d["budget"]["regime"]),int(d["budget"]["limit"])),d.get("environment",{}),d.get("training",{}),d.get("evaluation",{}))
    c.validate();return c
