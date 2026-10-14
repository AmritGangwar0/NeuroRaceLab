# NeuroRaceLab live driving viewer

Adds keyboard-driven simulation and orange sensor-ray overlays.

Run from the repository root:
```bash
python scripts/drive_track.py
python scripts/drive_track.py --track procedural --seed 42
```

Controls: Up accelerate, Down brake, Left/Right steer, Space reset episode, R reset camera,
mouse wheel zoom, Esc quit. The viewer uses the existing RaceEnvironment step interface and
does not modify the physics or training code.
