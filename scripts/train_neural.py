"""Train NeuroRaceLab neural controllers with alternating ES/EA generations."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from neuroracelab.agents.neural import ARCHITECTURES, build_neural_controller
from neuroracelab.environment import RaceEnvironment
from neuroracelab.environment.tracks.selection import build_track
from neuroracelab.training import EvolutionTrainer, TrainingConfig

ROOT = Path(__file__).resolve().parents[1]

def make_fitness(architecture, hidden_size, model_seed, track_family, max_steps, width_m):
    def fitness(parameters, episode_seed):
        controller = build_neural_controller(architecture, hidden_size=hidden_size, seed=model_seed)
        controller.load_parameters(parameters)
        controller.reset()
        track = build_track(track_family, episode_seed, width_m)
        env = RaceEnvironment(track=track, seed=episode_seed)
        obs = env.reset(seed=episode_seed)
        total_reward = 0.0
        info = {}
        for _ in range(max_steps):
            obs, reward, terminated, truncated, info = env.step(controller.act(obs))
            total_reward += float(reward)
            if terminated or truncated: break
        # Reward already values progress and completion; add a small explicit progress term.
        return total_reward + 10.0 * float(info.get("progress", 0.0))
    return fitness

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--architectures", nargs="+", choices=ARCHITECTURES, default=list(ARCHITECTURES))
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--generations-per-phase", type=int, default=2)
    p.add_argument("--population", type=int, default=8)
    p.add_argument("--max-evaluations", type=int, default=64)
    p.add_argument("--max-steps", type=int, default=2400)
    p.add_argument("--track", choices=("oval", "circuit", "procedural"), default="oval")
    p.add_argument("--width", type=float, default=8.0)
    p.add_argument("--output", type=Path, default=ROOT / "results" / "training")
    args = p.parse_args()
    if args.max_steps < 1: p.error("--max-steps must be positive")
    if args.width <= 0: p.error("--width must be positive")
    if args.hidden_size < 1: p.error("--hidden-size must be positive")

    for index, architecture in enumerate(args.architectures):
        model_seed = args.seed + index
        model = build_neural_controller(architecture, args.hidden_size, model_seed)
        cfg = TrainingConfig(generations_per_phase=args.generations_per_phase,
                             population_size=args.population, seed=args.seed,
                             max_evaluations=args.max_evaluations)
        fitness = make_fitness(architecture, args.hidden_size, model_seed, args.track,
                               args.max_steps, args.width)
        result = EvolutionTrainer(fitness, cfg).train(model.parameters(), architecture)
        metadata = {**model.metadata(), "track_family": args.track,
                    "max_steps": args.max_steps, "training_config": vars(cfg)}
        checkpoint, history = result.save(args.output, metadata)
        print(json.dumps({"architecture": architecture, "best_fitness": result.best_fitness,
                          "evaluations": result.evaluations, "checkpoint": str(checkpoint),
                          "history": str(history)}, indent=2))

if __name__ == "__main__": main()
