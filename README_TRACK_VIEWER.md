
# NeuroRaceLab track viewer patch

This patch adds a deterministic, smooth predefined circuit and an optional Pygame CE viewer.

## Apply

From the repository root, extract this archive so its `src/`, `scripts/`, and `tests/`
directories merge into the existing directories. It does not overwrite existing modules.

## Run

```bash
python -m pip install -e ".[dev,render]"
python scripts/view_track.py
pytest -q
```

Controls: arrow keys pan, mouse wheel zooms, `R` resets the view, `Esc` exits.

The renderer only reads track geometry and a vehicle state. It does not change simulation
physics, sensors, rewards, or training behavior.
