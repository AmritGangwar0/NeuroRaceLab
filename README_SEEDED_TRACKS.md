
# Seeded track selection patch

This patch adds selectable track types to the standalone viewer:

- `circuit`: fixed reference-style course (default)
- `oval`: fixed oval
- `procedural`: generated from the supplied integer seed

## Apply

From the NeuroRaceLab repository root:

```bash
unzip -n ~/Downloads/NeuroRaceLab_seeded_tracks_patch_v0_1.zip -d .
```

## Examples

```bash
python scripts/view_track.py --track circuit
python scripts/view_track.py --track oval
python scripts/view_track.py --track procedural --seed 42
python scripts/view_track.py --track procedural --seed 43
```

The same procedural seed recreates the same track; a different seed creates a different
track. The viewer prints the track ID and length before opening. The seed is also passed
to the environment reset. The fixed circuit and oval remain unchanged regardless of seed.
