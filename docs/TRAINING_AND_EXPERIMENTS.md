# Training and experiments

## Purpose

The training pipeline is currently an experimental research harness. Its first job is to make controlled experiments repeatable, not to guarantee a successful racing policy.

## Example smoke run

```bash
PYTHONPATH=src python scripts/train_neural.py \
  --architectures residual_rnn continuous_time equilibrium \
  --hidden-size 16 \
  --population 8 \
  --max-evaluations 64 \
  --max-steps 300 \
  --track oval
```

This is a small pipeline test. The hidden size of 16 is intentionally smaller than the planned research configuration and is not a capacity-matched benchmark against larger controllers.

## Optimizer schedule

The prototype alternates ES → EA → ES → EA until the configured evaluation budget is exhausted. Each architecture should receive the same evaluation budget and environment conditions for a fair comparison.

Evolution Strategies generally use perturbations of parameter vectors to estimate useful update directions. Evolutionary Algorithms generally maintain a population and select/mutate candidates. The exact implementation details are defined in `src/neuroracelab/training/evolution.py`; this guide does not substitute for that source.

## Experimental protocol

For a meaningful comparison:

1. Freeze the simulator, observation schema, reward, and termination rules for the experiment.
2. Specify architecture, hidden size, initialization seed, optimizer schedule, and total evaluation budget before running.
3. Use the same training seeds and evaluation budget across architectures.
4. Reserve disjoint validation and test seeds.
5. Repeat across multiple independent training seeds.
6. Save source commit SHA and dependency versions with every run.
7. Evaluate checkpoints on held-out episodes after training.
8. Report failed episodes and termination reasons, not just best fitness.
9. Repeat runs when stochastic variation could change conclusions.

## Metrics to report

At minimum:

- Completion rate
- Mean progress
- Mean episode reward
- Mean absolute lateral error
- Mean inference latency
- Wall-clock training/evaluation time
- Termination-reason counts
- Parameter count
- Total environment evaluations

Fitness and reward should never be the only reported metrics. A controller can exploit the reward function while failing to complete the intended task.

## Existing pilot result: interpret carefully

A recorded pilot used hidden size 16 and 64 evaluations per architecture. The resulting checkpoint comparison over oval seeds 42–46 reported zero lap completions for all three architectures. This is evidence that the pilot policies were not successful under that evaluation setup; it is not enough data to conclude that any architecture is inherently inferior.

Do not present the pilot as a successful racing result. Use it as a baseline for debugging the objective, optimizer handoff, episode budget, and control behavior.

## Output files

The current training script writes checkpoint parameter vectors and history JSON under `results/training/`. File naming may include architecture and seed. Keep training artifacts reproducible and avoid committing large generated outputs unless they are intentionally part of a release artifact.
