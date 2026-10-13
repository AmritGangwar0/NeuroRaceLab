from dataclasses import dataclass, field
from neuroracelab.core import BenchmarkRegime
@dataclass(frozen=True)
class SensorConfig:
    ray_count:int=32
    include_speed:bool=True
    include_slip_angle:bool=True
    include_angular_velocity:bool=True
    include_steering_state:bool=True
    @property
    def observation_dim(self): return self.ray_count+int(self.include_speed)+int(self.include_slip_angle)+int(self.include_angular_velocity)+int(self.include_steering_state)
@dataclass(frozen=True)
class AgentConfig:
    architecture:str
    primary_neurons:int=128
@dataclass(frozen=True)
class BudgetConfig:
    regime:BenchmarkRegime
    limit:int
@dataclass(frozen=True)
class ExperimentConfig:
    name:str
    seed:int
    agent:AgentConfig
    sensors:SensorConfig
    budget:BudgetConfig
    environment:dict=field(default_factory=dict)
    training:dict=field(default_factory=dict)
    evaluation:dict=field(default_factory=dict)
    def validate(self):
        if self.agent.primary_neurons!=128: raise ValueError("Initial benchmark requires 128 primary computational neurons.")
        if self.sensors.observation_dim!=36: raise ValueError("Initial observation interface must be 36-dimensional.")
        if self.budget.limit<=0: raise ValueError("Budget must be positive.")
