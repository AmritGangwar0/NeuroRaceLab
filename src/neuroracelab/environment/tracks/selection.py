from neuroracelab.environment.tracks.predefined.basic import oval
from neuroracelab.environment.tracks.predefined.circuit import circuit
from neuroracelab.environment.tracks.procedural.generator import (
    ProceduralTrackGenerator,
)


def build_track(track_name: str, seed: int, width: float):
    """Build a reproducible track from a name and seed."""
    if track_name == "circuit":
        return circuit(width=width)

    if track_name == "oval":
        return oval(width=width)

    if track_name == "procedural":
        return ProceduralTrackGenerator(
            seed=seed,
            width_m=width,
        ).generate()

    raise ValueError(f"Unknown track type: {track_name}")
