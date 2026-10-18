"""Compatibility fix for build_track's required width argument."""
from pathlib import Path
path = Path("scripts/evaluate_checkpoints.py")
if not path.exists():
    raise SystemExit(f"Run from the NeuroRaceLab repository root; missing {path}")
source = path.read_text()
old = "track = build_track(track_name, seed=seed)"
new = "track = build_track(track_name, seed=seed, width=float(metadata.get('track_width', 10.0)))"
if old not in source:
    if new in source:
        print("Width fix already applied; no changes made.")
    else:
        raise SystemExit("Could not find expected build_track call. Inspect scripts/evaluate_checkpoints.py and patch the call manually.")
else:
    source = source.replace(old, new)
    path.write_text(source)
    print("Patched scripts/evaluate_checkpoints.py to pass track width.")
