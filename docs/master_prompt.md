# NeuroRaceLab Master Prompt

Read README.md and docs/prd.md, docs/architecture.md, docs/memory.md, docs/glossary.md and all ADRs before changing code.

Treat the repository as a neural-computation research framework, not a generic autonomous-driving project.
The racecar is the benchmark; architecture is the experimental variable.

Non-negotiable initial conditions:
- RNN+Residual, Continuous-Time, Equilibrium
- 128 primary computational neurons
- 36-D observation: 32 rays + speed + slip angle + angular velocity + steering
- steering + throttle/brake actions
- same training method: ES saturation -> EA saturation -> repeat
- capacity-matched regime budgeted by environment evaluations
- compute-matched regime budgeted by FLOPs
- wall-clock secondary
- unseen-track generalization is primary
- explicit RNG, accounting, registries, configuration and reproducibility
- Hugging Face export required

Do not implement unresolved research decisions silently. If a choice affects fairness,
record an ADR before implementation. Do not couple simulator code to a particular architecture.
Do not allow test tracks into training. Do not use different optimizers for different models.

Current phase: structural foundation. Next, resolve vehicle dynamics, track model/generation,
sensor geometry/normalization, control timestep, exact architecture computation, ES/EA details,
saturation, fitness and global budgets before full training implementation.
