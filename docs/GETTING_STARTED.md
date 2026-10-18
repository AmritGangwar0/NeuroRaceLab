# Getting started

This guide takes you from a fresh checkout to tests, a manual simulation, and a small training smoke test.

## 1. Requirements

- Git
- Python version supported by the checked-out branch
- A C-capable system environment only if one of the installed dependencies needs compilation
- A graphical desktop session for the Pygame workbench and Tkinter GUI

The development machine used during recent work was Fedora Linux. Other operating systems may work, but they have not been established as tested platforms.

## 2. Clone and create an environment

```bash
git clone <YOUR_REPOSITORY_URL>
cd NeuroRaceLab

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install dependencies using the repository's canonical dependency file, if present. The GUI's optional document-export functionality may require `matplotlib`, `numpy`, `reportlab`, `python-docx`, and `python-pptx`. Do not install optional dependencies unless you need those features.

## 3. Run tests

```bash
PYTHONPATH=src pytest -q
```

A test result only applies to the commit and environment in which it ran. Record the Python version and commit SHA when preparing a release.

## 4. View or drive a track

Use the script's own help output first:

```bash
PYTHONPATH=src python scripts/drive_track.py --help
```

Example:

```bash
PYTHONPATH=src python scripts/drive_track.py --track circuit
```

The manual driving workbench is a separate Pygame application.

## 5. Launch the desktop GUI

```bash
python scripts/gui_app.py
```

The GUI can register supported model files, export native racing-controller bundles, export reports from JSON/CSV data, and launch the population workbench in a separate Pygame window.

## 6. Run a small training smoke test

```bash
PYTHONPATH=src python scripts/train_neural.py \
  --architectures residual_rnn continuous_time equilibrium \
  --hidden-size 16 \
  --population 8 \
  --max-evaluations 64 \
  --max-steps 300 \
  --track oval
```

Use a small budget to validate the pipeline. Do not interpret this run as a meaningful architecture comparison.

## 7. Where outputs go

Typical generated output locations in the current implementation:

- `results/baseline/`: baseline benchmark JSON/CSV
- `results/training/`: controller checkpoint `.npz` files and training-history JSON
- `results/`: evaluation summaries, plots, and generated reports
- `models/library/`: models imported through the GUI

Generated outputs may be ignored by Git. Check `.gitignore` before expecting these files to be tracked.

## Troubleshooting

### `ModuleNotFoundError: neuroracelab`

Run scripts from the repository root with `PYTHONPATH=src`, or install the package in editable mode once the project has a valid packaging configuration.

### GUI opens but a report export fails

Check that the relevant optional dependency is installed in the same virtual environment used to launch the GUI. Read the error message before changing the project's core dependencies.

### Pygame window does not appear

Confirm you are running inside a graphical desktop session and that Pygame can create a display. Headless CI environments generally cannot show the interactive window.

### A training checkpoint has a high fitness but the car does not finish a lap

Fitness is an optimization signal, not a proof of task success. Evaluate saved checkpoints over multiple seeds and inspect completion rate, progress, lateral error, reward, and termination reason.
