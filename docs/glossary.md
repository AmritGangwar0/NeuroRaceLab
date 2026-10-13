# NeuroRaceLab Glossary

**Agent:** controller mapping observations to actions with optional persistent state.

**Primary computational state:** the 128-neuron persistent state used by the initial models.

**Observation:** standardized input at a control step.

**Action:** standardized continuous vehicle control.

**Environment evaluation:** one complete candidate rollout under a defined condition.

**Fitness:** scalar optimization objective.

**ES:** Evolution Strategies; parameter-space perturbation/search.

**EA:** Evolutionary Algorithm; population-level selection and variation.

**Saturation:** configured multi-check plateau condition that triggers a phase switch.

**Capacity-matched:** comparison centered on equal primary state and comparable parameter capacity.

**Compute-matched:** comparison centered on comparable inference FLOPs.

**Run:** reproducible execution of an experiment configuration.

**Research-sanity test:** automated test protecting scientific validity assumptions.
