from dataclasses import dataclass
from neuroracelab.core import EnvironmentEvaluation,Timer
@dataclass(frozen=True)
class EvaluationConfig:
    max_steps:int
    episodes:int
    seed:int
    flops_per_step:int=0
class RolloutEvaluator:
    def __init__(self,environment_factory,fitness_fn): self.environment_factory=environment_factory;self.fitness_fn=fitness_fn
    def evaluate(self,agent,config):
        out=[]
        for ep in range(config.episodes):
            env=self.environment_factory();obs=env.reset(seed=config.seed+ep);agent.reset();trajectory=[]
            with Timer() as timer:
                for _ in range(config.max_steps):
                    r=env.step(agent.act(obs));trajectory.append(r);obs=r.observation
                    if r.terminated or r.truncated: break
            out.append(EnvironmentEvaluation(f"episode-{ep}",float(self.fitness_fn(trajectory)),len(trajectory),1,len(trajectory)*config.flops_per_step,timer.elapsed))
        return out
