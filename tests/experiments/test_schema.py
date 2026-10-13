from neuroracelab.experiments.loader import load_json
def test_capacity_config():
 c=load_json("configs/experiments/example_capacity_matched.json");assert c.sensors.observation_dim==36 and c.agent.primary_neurons==128
def test_compute_config(): assert load_json("configs/experiments/example_compute_matched.json").budget.limit>0
