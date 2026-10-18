# Contributing

Thanks for considering a contribution to NeuroRaceLab.

## Before opening a change

- Open an issue for substantial features or behavior changes.
- Keep changes focused and explain the research or engineering motivation.
- Avoid unrelated formatting churn.
- Do not commit virtual environments, caches, local machine paths, checkpoints, or large generated results by accident.
- Do not change the observation/action interface without updating controllers, tests, metadata, and documentation.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
PYTHONPATH=src pytest -q
```

Install project and optional dependencies according to the repository's current dependency configuration.

## Testing expectations

Run the full test suite after changes:

```bash
PYTHONPATH=src pytest -q
```

For controller changes, add tests for:

- Deterministic initialization with a fixed seed
- Observation/action shapes
- Action bounds
- Parameter vector round-trip
- Invalid parameter rejection
- Reset behavior

For environment or track changes, test termination, sensor behavior, geometry validity, and seeded reproducibility. For report/GUI changes, test import/export paths and handle missing optional dependencies clearly.

## Research integrity

- Use identical budgets and comparable settings when comparing architectures.
- Separate training, validation, and test seeds.
- Report unsuccessful runs.
- Do not claim generalization from a few smoke-test episodes.
- Include enough configuration metadata for another person to reproduce the run.

## Pull requests

A good PR includes:

1. Motivation and scope
2. Design summary
3. Tests run and their results
4. Expected behavioral changes
5. Screenshots or sample reports for visible GUI/report changes
6. Limitations and follow-up work

## Style

Follow the style already used in the touched module. Avoid mass reformatting unrelated files.
