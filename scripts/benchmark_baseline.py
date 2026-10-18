"""Run reproducible baseline evaluations across NeuroRaceLab track families."""
from __future__ import annotations
import argparse, csv, json, platform, statistics, sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter, perf_counter_ns
import numpy as np
from neuroracelab.agents.baseline import BaselineController
from neuroracelab.environment import EnvironmentConfig, RaceEnvironment
from neuroracelab.environment.tracks.selection import build_track

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "baseline"
TRACK_FAMILIES = ("oval", "circuit", "procedural")

def run_episode(track_family, seed, controller, max_steps, max_duration_s, width_m):
    track = build_track(track_family, seed, width_m)
    env = RaceEnvironment(track=track, config=EnvironmentConfig(max_duration_s=max_duration_s), seed=seed)
    observation = env.reset(seed=seed)
    controller.reset()
    episode_return, inference_times_ns, absolute_lateral_errors = 0.0, [], []
    steps, terminated, truncated, info = 0, False, False, {}
    wall_start = perf_counter()
    for steps in range(1, max_steps + 1):
        inference_start = perf_counter_ns()
        action = controller.act(observation)
        inference_times_ns.append(perf_counter_ns() - inference_start)
        observation, reward, terminated, truncated, info = env.step(action)
        episode_return += float(reward)
        if "lateral_error" in info:
            absolute_lateral_errors.append(abs(float(info["lateral_error"])))
        if terminated or truncated:
            break
    wall_seconds = perf_counter() - wall_start
    completed = info.get("termination_reason") == "lap_complete"
    latency_ms = [v / 1_000_000 for v in inference_times_ns]
    return {
        "track_family": track_family, "track_id": getattr(track, "track_id", track_family), "seed": seed,
        "steps": steps, "completed": completed,
        "termination_reason": info.get("termination_reason") or "step_limit",
        "progress": float(info.get("progress", 0.0)), "episode_return": episode_return,
        "simulated_time_s": float(info.get("elapsed_s", 0.0)), "wall_time_s": wall_seconds,
        "inference_latency_mean_ms": statistics.fmean(latency_ms) if latency_ms else None,
        "inference_latency_median_ms": statistics.median(latency_ms) if latency_ms else None,
        "inference_latency_p95_ms": float(np.percentile(latency_ms, 95)) if latency_ms else None,
        "inference_latency_max_ms": max(latency_ms) if latency_ms else None,
        "mean_abs_lateral_error_m": float(np.mean(absolute_lateral_errors)) if absolute_lateral_errors else None,
        "final_lateral_error_m": info.get("lateral_error"), "controller": controller.metadata(),
        "terminated": bool(terminated), "truncated": bool(truncated),
    }

def summarize(records):
    if not records: return {"episodes": 0}
    completed = [r for r in records if r["completed"]]
    reasons = sorted({r["termination_reason"] for r in records})
    def mean_metric(key):
        values = [r[key] for r in records if r.get(key) is not None]
        return statistics.fmean(values) if values else None
    return {
        "episodes": len(records), "completed_episodes": len(completed), "completion_rate": len(completed) / len(records),
        "mean_progress": mean_metric("progress"), "mean_return": mean_metric("episode_return"),
        "mean_steps": mean_metric("steps"), "mean_simulated_time_s": mean_metric("simulated_time_s"),
        "mean_wall_time_s": mean_metric("wall_time_s"), "mean_inference_latency_ms": mean_metric("inference_latency_mean_ms"),
        "mean_abs_lateral_error_m": mean_metric("mean_abs_lateral_error_m"),
        "termination_reasons": {reason: sum(r["termination_reason"] == reason for r in records) for reason in reasons},
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", nargs="+", choices=TRACK_FAMILIES, default=list(TRACK_FAMILIES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--max-steps", type=int, default=2400)
    parser.add_argument("--max-duration", type=float, default=120.0)
    parser.add_argument("--width", type=float, default=8.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.max_steps < 1: parser.error("--max-steps must be positive")
    if args.max_duration <= 0: parser.error("--max-duration must be positive")
    if args.width <= 0: parser.error("--width must be positive")
    if not args.seeds: parser.error("At least one seed is required")
    outdir = args.output.expanduser().resolve(); outdir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("baseline_%Y%m%dT%H%M%SZ")
    controller, records = BaselineController(), []
    config = {"run_id": run_id, "created_at_utc": datetime.now(timezone.utc).isoformat(), "model": controller.metadata(),
              "track_families": args.tracks, "seeds": args.seeds, "max_steps": args.max_steps,
              "max_duration_s": args.max_duration, "track_width_m": args.width, "python_version": sys.version,
              "platform": platform.platform(), "numpy_version": np.__version__,
              "purpose": "baseline smoke benchmark; not a held-out generalization claim"}
    print(f"Run: {run_id}\nEpisodes planned: {len(args.tracks) * len(args.seeds)}")
    for track in args.tracks:
        for seed in args.seeds:
            print(f"Running track={track}, seed={seed} ...")
            row = run_episode(track, seed, controller, args.max_steps, args.max_duration, args.width)
            row["run_id"] = run_id; records.append(row)
            print(f"  completed={row['completed']}, progress={row['progress']:.3f}, reason={row['termination_reason']}, steps={row['steps']}, latency={row['inference_latency_mean_ms']:.4f} ms")
    grouped = {t: summarize([r for r in records if r["track_family"] == t]) for t in args.tracks}
    report = {"config": config, "overall_summary": summarize(records), "track_summaries": grouped, "episodes": records}
    jp, cp = outdir / f"{run_id}.json", outdir / f"{run_id}.csv"
    jp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with cp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys())); writer.writeheader()
        for row in records: writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    print("\nOverall summary:"); print(json.dumps(report["overall_summary"], indent=2)); print(f"\nJSON: {jp}\nCSV:  {cp}")
    return 0
if __name__ == "__main__": raise SystemExit(main())
