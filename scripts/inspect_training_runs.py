"""Inspect saved NeuroRaceLab training histories and checkpoint metadata."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def main() -> None:
    base = Path("results/training")
    for path in sorted(base.glob("*_history.json")):
        data = json.loads(path.read_text())
        meta = data.get("metadata", {})
        rows = data.get("history", [])
        print(f"\n=== {path.name} ===")
        print(f"architecture={data.get('architecture')} seed={data.get('seed')} "
              f"evaluations={data.get('evaluations')} best_fitness={data.get('best_fitness')}")
        print(f"hidden_size={meta.get('hidden_size')} params={meta.get('parameter_count')} "
              f"track={meta.get('track_family')} max_steps={meta.get('max_steps')}")
        phases = {}
        for row in rows:
            phase = row.get("phase", "unknown")
            phases[phase] = phases.get(phase, 0) + int(row.get("evaluations_this_generation", 0))
            print("  generation={generation} phase={phase} best={generation_best_fitness:.6f} "
                  "mean={mean_fitness:.6f} evals={cumulative_evaluations}".format(
                      generation=row.get("generation", -1),
                      phase=phase,
                      generation_best_fitness=row.get("generation_best_fitness", float("nan")),
                      mean_fitness=row.get("mean_fitness", float("nan")),
                      cumulative_evaluations=row.get("cumulative_evaluations", -1)))
        print(f"evaluations by phase: {phases}")
        checkpoint = base / f"{data.get('architecture')}_seed{data.get('seed')}_checkpoint.npz"
        if checkpoint.exists():
            with np.load(checkpoint) as ckpt:
                print(f"checkpoint={checkpoint}")
                for key in ckpt.files:
                    print(f"  {key}: shape={ckpt[key].shape} dtype={ckpt[key].dtype}")
        else:
            print(f"WARNING: expected checkpoint missing: {checkpoint}")

if __name__ == "__main__":
    main()
