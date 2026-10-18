# Evaluation and reporting

## Baseline benchmark

The deterministic baseline provides a reference point for checking the environment and evaluation pipeline before comparing learned policies.

Example:

```bash
PYTHONPATH=src python scripts/benchmark_baseline.py --help
```

The smoke benchmark previously evaluated seeds 42, 43, and 44 on oval, circuit, and procedural tracks. It completed all three oval episodes and all three procedural episodes, but completed none of the three circuit episodes. This is a small diagnostic run, not a generalization result.

## Evaluate trained checkpoints

Use the evaluator's current help output to confirm arguments:

```bash
PYTHONPATH=src python scripts/evaluate_checkpoints.py --help
```

Evaluation should use fixed, documented seeds and the same track configuration for every architecture. Avoid evaluating only the training seed.

The evaluator reports metrics including completion rate, progress, reward, absolute lateral error, inference latency, and wall time. Inspect termination reasons and per-episode outputs when aggregate metrics look suspicious.

## Reports

Generate the available reports with:

```bash
PYTHONPATH=src python scripts/render_report.py
PYTHONPATH=src python scripts/render_training_report.py
```

Use `--help` if supported by the checked-out version. Report scripts may expect prior benchmark/evaluation JSON or CSV files to exist.

## GUI document export

The GUI can export selected JSON/CSV experiment data into PDF, DOCX, or PPTX. It can generate a simple numeric trend plot and embed it in the output. The report is a convenient data summary, not a peer-reviewed paper or a substitute for documenting experimental methodology.

## Reproducibility checklist

- [ ] Commit SHA recorded
- [ ] Python and dependency versions recorded
- [ ] Architecture and hidden size recorded
- [ ] Optimizer settings and evaluation budget recorded
- [ ] Training seeds separated from test seeds
- [ ] Per-episode results retained
- [ ] Completion and failure counts reported
- [ ] Plot axes, units, and sample sizes clearly labelled
- [ ] Failed or incomplete runs included in the analysis
