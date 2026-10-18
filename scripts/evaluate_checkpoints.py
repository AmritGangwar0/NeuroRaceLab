"""Evaluate saved NeuroRaceLab checkpoints on full episodes.

Run from the repository root with the project's virtual environment active.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from neuroracelab.agents.neural import build_neural_controller
from neuroracelab.environment.simulator.environment import RaceEnvironment
from neuroracelab.environment.tracks.selection import build_track


def evaluate_one(architecture: str, checkpoint_path: Path, history_path: Path,
                 track_name: str, seeds: list[int], max_steps: int | None) -> dict:
    history_data = json.loads(history_path.read_text())
    metadata = history_data["metadata"]
    model = build_neural_controller(architecture, seed=int(history_data.get("seed", 123)),
                                    hidden_size=int(metadata["hidden_size"]))
    with np.load(checkpoint_path) as checkpoint:
        model.load_parameters(checkpoint["parameters"])

    episodes = []
    for seed in seeds:
        track = build_track(track_name, seed=seed, width=float(metadata.get('track_width', 10.0)))
        env = RaceEnvironment(track=track, seed=seed)
        obs = env.reset(seed=seed)
        terminated = truncated = False
        total_reward = 0.0
        steps = 0
        inference_ms = []
        abs_lateral_errors = []
        started = time.perf_counter()
        while not (terminated or truncated):
            t0 = time.perf_counter()
            action = model.act(obs)
            inference_ms.append((time.perf_counter() - t0) * 1000.0)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            if "lateral_error" in info:
                abs_lateral_errors.append(abs(float(info["lateral_error"])))
            steps += 1
            if max_steps is not None and steps >= max_steps and not (terminated or truncated):
                truncated = True
        wall_seconds = time.perf_counter() - started
        final_info = info if steps else {}
        episodes.append({
            "seed": seed,
            "track": track_name,
            "steps": steps,
            "total_reward": total_reward,
            "progress": float(final_info.get("progress", 0.0)),
            "mean_abs_lateral_error_m": float(np.mean(abs_lateral_errors))
                if abs_lateral_errors else float("nan"),
            "termination_reason": final_info.get("termination_reason", "unknown"),
            "lap_complete": final_info.get("termination_reason") == "lap_complete",
            "mean_inference_ms": float(np.mean(inference_ms)) if inference_ms else 0.0,
            "p95_inference_ms": float(np.percentile(inference_ms, 95)) if inference_ms else 0.0,
            "wall_seconds": wall_seconds,
        })
    return {
        "architecture": architecture,
        "checkpoint": str(checkpoint_path),
        "training_history": str(history_path),
        "track": track_name,
        "evaluation_seeds": seeds,
        "hidden_size": metadata["hidden_size"],
        "parameter_count": metadata["parameter_count"],
        "episodes": episodes,
        "summary": {
            "completion_rate": float(np.mean([e["lap_complete"] for e in episodes])),
            "mean_progress": float(np.mean([e["progress"] for e in episodes])),
            "mean_reward": float(np.mean([e["total_reward"] for e in episodes])),
            "mean_abs_lateral_error_m": float(np.mean([e["mean_abs_lateral_error_m"] for e in episodes])),
            "mean_inference_ms": float(np.mean([e["mean_inference_ms"] for e in episodes])),
            "mean_wall_seconds": float(np.mean([e["wall_seconds"] for e in episodes])),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architectures", nargs="+",
                        choices=["residual_rnn", "continuous_time", "equilibrium"],
                        default=["residual_rnn", "continuous_time", "equilibrium"])
    parser.add_argument("--track", default="oval")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Optional episode cap. Omit for the environment's full episode limit.")
    parser.add_argument("--results-dir", type=Path, default=Path("results/training"))
    args = parser.parse_args()
    outputs = []
    for arch in args.architectures:
        checkpoint = args.results_dir / f"{arch}_seed123_checkpoint.npz"
        history = args.results_dir / f"{arch}_seed123_history.json"
        if not checkpoint.exists() or not history.exists():
            raise FileNotFoundError(f"Missing checkpoint/history for {arch}: {checkpoint}, {history}")
        result = evaluate_one(arch, checkpoint, history, args.track, args.seeds, args.max_steps)
        out = args.results_dir / f"{arch}_evaluation.json"
        out.write_text(json.dumps(result, indent=2) + "\n")
        outputs.append(result)
        print(json.dumps({"architecture": arch, **result["summary"]}, indent=2))
    comparison = {
        "track": args.track,
        "evaluation_seeds": args.seeds,
        "note": "Evaluation results are comparable only when track, seeds, episode limits, and environment settings match.",
        "architectures": outputs,
    }
    (args.results_dir / "checkpoint_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")


if __name__ == "__main__":
    main()
