import numpy as np

from neuroracelab.training import EvolutionTrainer, TrainingConfig


def test_training_tracks_fitness_and_respects_evaluation_budget(tmp_path):
    def objective(parameters, seed):
        return -float(np.sum((parameters - 0.25) ** 2))
    config = TrainingConfig(generations_per_phase=1, population_size=4,
                            seed=7, max_evaluations=12)
    result = EvolutionTrainer(objective, config).train(np.zeros(10), "test_model")
    assert result.evaluations <= config.max_evaluations
    assert result.evaluations > 0
    assert result.history
    assert {row["phase"] for row in result.history} <= {"ES", "EA"}
    assert np.isfinite(result.best_fitness)
    checkpoint, history = result.save(tmp_path, {"hidden_size": 5})
    assert checkpoint.exists() and history.exists()
    saved = np.load(checkpoint)
    assert saved["parameters"].shape == (10,)

def test_es_ea_phase_order_alternates():
    objective = lambda p, seed: -float(np.sum(p*p))
    cfg = TrainingConfig(generations_per_phase=1, population_size=4,
                         seed=2, max_evaluations=16)
    result = EvolutionTrainer(objective, cfg).train(np.zeros(5), "test")
    phases = [row["phase"] for row in result.history]
    assert phases[:4] == ["ES", "EA", "ES", "EA"]
