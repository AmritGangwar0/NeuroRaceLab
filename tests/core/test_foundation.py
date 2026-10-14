from neuroracelab.core import BenchmarkRegime, Budget, Registry, RNGManager, RunAccounting
from neuroracelab.experiments.schema import SensorConfig


def test_observation_is_36d(): assert SensorConfig().observation_dim==36
def test_registry_rejects_duplicates():
 r=Registry();r.register("x",lambda:1)
 try:r.register("x",lambda:2)
 except ValueError:return
 assert False
def test_rng_reproducible():
 a=RNGManager(42);b=RNGManager(42)
 assert a.seed_for("tracks")==b.seed_for("tracks") and a.seed_for("tracks")!=a.seed_for("es")
def test_capacity_budget():
 b=Budget(BenchmarkRegime.CAPACITY_MATCHED,10)
 assert not b.exhausted(RunAccounting(environment_evaluations=9)) and b.exhausted(RunAccounting(environment_evaluations=10))
def test_compute_budget():
 b=Budget(BenchmarkRegime.COMPUTE_MATCHED,100)
 assert not b.exhausted(RunAccounting(flops=99)) and b.exhausted(RunAccounting(flops=100))
