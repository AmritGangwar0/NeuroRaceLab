from __future__ import annotations

import argparse

from neuroracelab.environment.simulator.environment import RaceEnvironment
from neuroracelab.environment.tracks.selection import build_track
from neuroracelab.visualization.renderer import TrackRenderer

#def build_track(track_name: str, seed: int, width: float):
 #   """Build a reproducible track from a name and seed."""
  #  if track_name == "circuit":
   #     return circuit(width=width)
    #if track_name == "oval":
     #   return oval(width=width)
   # if track_name == "procedural":
    #    return ProceduralTrackGenerator(seed=seed, width_m=width).generate()
   # raise ValueError(f"Unknown track type: {track_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="View a NeuroRaceLab track.")
    parser.add_argument(
        "--track",
        choices=("circuit", "oval", "procedural"),
        default="circuit",
        help="Track type (default: circuit).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Procedural track seed; the same seed reproduces the same track (default: 0).",
    )
    parser.add_argument("--width", type=float, default=9.0, help="Road width in metres.")
    parser.add_argument("--zoom", type=float, default=5.5, help="Initial pixels per metre.")
    parser.add_argument("--size", default="1200x800", help="Window size, e.g. 1280x800.")
    args = parser.parse_args()

    try:
        width, height = (int(value) for value in args.size.lower().split("x", 1))
        if width <= 0 or height <= 0:
            raise ValueError
    except (ValueError, TypeError):
        parser.error("--size must look like 1200x800 with positive dimensions")

    track = build_track(args.track, args.seed, args.width)
    print(f"Track: {track.track_id} | seed: {args.seed} | length: {track.length:.1f} m")
    env = RaceEnvironment(track=track, seed=args.seed)
    env.reset(seed=args.seed)
    renderer = TrackRenderer(track, size=(width, height), pixels_per_metre=args.zoom)
    renderer.run(env.state)


if __name__ == "__main__":
    main()
