# NeuroRaceLab

**A modular evolutionary neural-control laboratory for autonomous racing.**

NeuroRaceLab is an experimental platform for studying compact neural architectures in a controlled, continuous-control racing task. It combines a lightweight vehicle simulator, configurable tracks, neural controllers, evolutionary optimization, population visualization, and experiment reporting.

The project is designed around a research question: **how do different compact recurrent or dynamical neural architectures behave when trained under the same environment and evaluation budget?**

> **Research status:** NeuroRaceLab is an active experimental project, not a validated autonomous-driving system. The current pilot controllers did not complete evaluation laps in the recorded five-seed oval evaluation. Treat results as engineering diagnostics, not evidence of competitive driving or generalization.

## Features

- **Modular racing environment** with a planar vehicle model, track geometry, and ray-based observations.
- **Configurable tracks:** oval, circuit, and seeded procedural tracks.
- **Three experimental controller architectures:**
  - `residual_rnn`
  - `continuous_time`
  - `equilibrium`
- **Evolutionary training prototype** with a staged ES → EA → ES → EA schedule.
- **Population workbench** for observing multiple independently parameterized cars.
- **Baseline benchmarking** and checkpoint evaluation across multiple seeds.
- **JSON/CSV metrics and plot/report generation.**
- **Desktop GUI shell** for model-library management, report export, and launching the population workbench.
- **Portable controller bundles:** `.nrlmodel` is the intended NeuroRaceLab format for controller parameters and metadata.

GGUF files are a separate format intended for compatible language-model runtimes. Importing a GGUF file into the GUI does not make it a racing controller and does not enable inference.

## Project status

| Area | Status |
|---|---|
| Environment, vehicle dynamics, tracks and sensors | Implemented prototype |
| Baseline controller and benchmark runner | Implemented |
| Three neural controller interfaces | Implemented |
| ES/EA training pipeline | Experimental prototype |
| Multi-car population workbench | Implemented prototype |
| Checkpoint evaluation and report rendering | Implemented prototype |
| GUI model library and document export | Implemented prototype; verify optional dependencies locally |
| GUI launch/stop for Pygame workbench | Implemented; Pygame remains a separate window |
| GGUF inference from the GUI | Not implemented |
| GUI-launched training and embedded simulation view | Not implemented |
| Research-quality trained racing policy | Not yet demonstrated |

## Quick start

The exact dependency set can vary by your current branch. The following assumes a Linux development environment and a repository containing `.venv`, `src/neuroracelab`, and the scripts listed below.

```bash
git clone <YOUR_REPOSITORY_URL>
cd NeuroRaceLab

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the project's declared dependencies if a `pyproject.toml` or requirements file is present. If the project currently has no canonical dependency file, use the dependency instructions maintained in `docs/GETTING_STARTED.md` and avoid assuming that optional GUI/report packages are required for core simulation.

Run the test suite:

```bash
PYTHONPATH=src pytest -q
```

Launch the GUI:

```bash
python scripts/gui_app.py
```

Launch the population workbench directly:

```bash
PYTHONPATH=src python scripts/population_workbench.py \
  --track oval \
  --population 12 \
  --hidden-size 16
```

If the repository's current scripts differ from these names or arguments, inspect `python scripts/<script>.py --help` and update this README before release.

## Run a baseline benchmark

Example smoke benchmark:

```bash
PYTHONPATH=src python scripts/benchmark_baseline.py \
  --seeds 42 43 44 \
  --tracks oval circuit procedural
```

Use `--help` to see the exact options supported by the checked-out version. Outputs are written under `results/baseline/` in the current implementation.

## Train the experimental controllers

Example small smoke run:

```bash
PYTHONPATH=src python scripts/train_neural.py \
  --architectures residual_rnn continuous_time equilibrium \
  --hidden-size 16 \
  --population 8 \
  --max-evaluations 64 \
  --max-steps 300 \
  --track oval
```

This is a **pipeline smoke test**, not a recommended scientific training budget. It verifies that training runs and writes checkpoints/history files. A successful process exit does not mean the resulting policy drives well.

Inspect a training history:

```bash
PYTHONPATH=src python scripts/inspect_training_runs.py
```

Evaluate checkpoints:

```bash
PYTHONPATH=src python scripts/evaluate_checkpoints.py --help
```

Render training reports:

```bash
PYTHONPATH=src python scripts/render_training_report.py
```

Check the script's `--help` output and generated metadata for the exact supported options on your branch.

## Architecture overview

```text
Tracks ──> Environment / Vehicle Dynamics ──> Observation (36 values)
                         ▲                            │
                         │                            ▼
                    Action (2) <── Neural Controller / Baseline
                         ▲
                         │
                ES / EA Training Pipeline
                         │
              Checkpoints + History Metadata
                         │
             Evaluation + Reports + GUI Library
```

The current observation schema contains 32 ray distances plus speed, slip angle, yaw rate, and current steering. The controller emits two bounded values: steering and longitudinal throttle/brake.

Read [the architecture guide](docs/ARCHITECTURE.md) before changing environment/controller interfaces.

## Documentation

- [Getting started](docs/GETTING_STARTED.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Training and experiments](docs/TRAINING_AND_EXPERIMENTS.md)
- [Evaluation and reporting](docs/EVALUATION_AND_REPORTING.md)
- [Model formats](docs/MODEL_FORMATS.md)
- [GUI guide](docs/GUI_GUIDE.md)
- [Contributing](CONTRIBUTING.md)
- [Security policy](SECURITY.md)
- [Release checklist](docs/RELEASE_CHECKLIST.md)

## Reproducibility notes

- Record the exact Git commit, Python version, dependency versions, seed, track configuration, controller architecture, hidden size, training algorithm, and evaluation budget.
- Keep training, validation, and test seeds separate. Do not tune a controller against the final test seeds.
- Compare architectures under both parameter-matched and compute/evaluation-budget-matched settings where feasible.
- Report completion rate, progress, lateral error, reward, inference latency, and wall-clock cost together. Reward alone can hide failure modes.
- A few seeds and short runs are smoke tests, not statistically reliable evidence.
- Do not commit generated checkpoints, large result files, virtual environments, or personal machine paths unless the repository explicitly intends to version them.

## Limitations and safety

NeuroRaceLab is a research simulator. It does not model the full complexity of real vehicles, tire dynamics, perception, traffic, weather, or road safety. Its outputs must not be used to control real vehicles or make safety-critical decisions.

The current neural policies and training procedure are experimental. Baseline completion does not imply that a learned controller is successful, and a positive training fitness does not guarantee lap completion.

## License

No license is declared by this documentation patch. Choose a license intentionally before making the repository public; see [the release checklist](docs/RELEASE_CHECKLIST.md). Do not imply that the project is open source until a license is present.

## Acknowledgements

NeuroRaceLab is developed as an independent engineering and research project. Add specific third-party libraries, inspirations, datasets, and references here before release.
