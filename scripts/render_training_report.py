"""Render training curves and checkpoint-evaluation comparison plots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_json(path: Path):
    return json.loads(path.read_text())


def render_history(history_path: Path, output_dir: Path) -> None:
    data = load_json(history_path)
    rows = data.get("history", [])
    if not rows:
        print(f"Skipping empty history: {history_path}")
        return
    generations = [r["generation"] for r in rows]
    best = [r["generation_best_fitness"] for r in rows]
    mean = [r["mean_fitness"] for r in rows]
    evaluations = [r["cumulative_evaluations"] for r in rows]
    arch = data["architecture"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(generations, best, marker="o", label="Generation best")
    axes[0].plot(generations, mean, marker="s", label="Population mean")
    axes[0].set(xlabel="Generation", ylabel="Fitness", title=f"{arch}: fitness")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(evaluations, best, marker="o", label="Generation best")
    axes[1].set(xlabel="Cumulative environment evaluations", ylabel="Fitness",
                title="Fitness vs evaluation budget")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(f"{arch} | seed={data.get('seed')} | params={data.get('metadata', {}).get('parameter_count')}")
    fig.tight_layout()
    fig.savefig(output_dir / f"{arch}_training.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def render_comparison(evaluation_path: Path, output_dir: Path) -> None:
    data = load_json(evaluation_path)
    rows = data.get("architectures", [])
    if not rows:
        return
    names = [r["architecture"] for r in rows]
    summaries = [r["summary"] for r in rows]
    metrics = [
        ("completion_rate", "Lap completion rate", (0, 1)),
        ("mean_progress", "Mean progress", None),
        ("mean_abs_lateral_error_m", "Mean absolute lateral error (m)", None),
        ("mean_inference_ms", "Mean inference latency (ms)", None),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (key, title, limits) in zip(axes.flat, metrics):
        vals = [s.get(key, 0.0) for s in summaries]
        ax.bar(names, vals)
        ax.set_title(title)
        if limits:
            ax.set_ylim(*limits)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(f"Checkpoint evaluation | track={data.get('track')} | seeds={data.get('evaluation_seeds')}")
    fig.tight_layout()
    fig.savefig(output_dir / "checkpoint_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    lines = [
        "# NeuroRaceLab checkpoint evaluation",
        "",
        f"- Track: `{data.get('track')}`",
        f"- Seeds: `{data.get('evaluation_seeds')}`",
        "",
        "| Architecture | Parameters | Completion | Mean progress | Mean reward | Mean abs. lateral error (m) | Mean inference (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        s = r["summary"]
        lines.append(
            f"| {r['architecture']} | {r.get('parameter_count', 'n/a')} | "
            f"{s['completion_rate']:.1%} | {s['mean_progress']:.3f} | "
            f"{s['mean_reward']:.3f} | {s['mean_abs_lateral_error_m']:.3f} | "
            f"{s['mean_inference_ms']:.3f} |"
        )
    lines += [
        "",
        "## Interpretation notes",
        "",
        "- Compare architectures only under identical tracks, seeds, episode limits, and simulator settings.",
        "- Small evaluation sets are noisy; report variability and repeat with multiple training seeds before making claims.",
        "- `mean_inference_ms` measures controller call latency on this machine, not FLOPs or total model compute.",
        "- The training curves show the history recorded by the trainer; they do not establish generalization.",
    ]
    (output_dir / "checkpoint_comparison.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/training"))
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(args.results_dir.glob("*_history.json")):
        render_history(path, args.results_dir)
    evaluation = args.results_dir / "checkpoint_comparison.json"
    if evaluation.exists():
        render_comparison(evaluation, args.results_dir)
    else:
        print("No checkpoint_comparison.json yet; run scripts/evaluate_checkpoints.py first.")


if __name__ == "__main__":
    main()
