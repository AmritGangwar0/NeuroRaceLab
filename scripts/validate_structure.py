from neuroracelab.experiments.loader import load_json
for p in ["configs/experiments/example_capacity_matched.json","configs/experiments/example_compute_matched.json"]:
 c=load_json(p);print(c.name,c.sensors.observation_dim,c.budget.regime.value,c.budget.limit)
print("NeuroRaceLab structural validation passed.")
