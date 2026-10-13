from dataclasses import dataclass
from time import perf_counter
from .types import BenchmarkRegime
@dataclass
class Budget:
    regime: BenchmarkRegime
    limit: int
    def consumed(self, accounting):
        return accounting.environment_evaluations if self.regime is BenchmarkRegime.CAPACITY_MATCHED else accounting.flops
    def exhausted(self, accounting): return self.consumed(accounting)>=self.limit
    def remaining(self, accounting): return max(0,self.limit-self.consumed(accounting))
class Timer:
    def __enter__(self): self.start=perf_counter();self.elapsed=0.0;return self
    def __exit__(self,*_): self.elapsed=perf_counter()-self.start
