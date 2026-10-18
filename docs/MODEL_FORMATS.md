# Model formats

NeuroRaceLab handles two different model families. They are not interchangeable.

## Native racing-controller checkpoint: `.npz`

The current training pipeline stores a flat numeric parameter vector in an NPZ archive under the `parameters` key. The associated training-history JSON contains metadata such as architecture, hidden size, seed, track, and training budget.

An NPZ checkpoint is a parameter container, not a self-describing executable model. Preserve the matching metadata file.

## Portable controller bundle: `.nrlmodel`

`.nrlmodel` is the NeuroRaceLab-specific portable bundle. In the current v1 format, it is a ZIP archive containing:

- `manifest.json`: format/version, architecture, hidden size, observation/action schema, parameter count, and source checkpoint name
- `parameters.npz`: flat numeric parameter vector
- `metadata.json`: training metadata, when available

The current GUI can export a checkpoint to `.nrlmodel`, validate a bundle, and import it into the model library. A validated bundle is not automatically loaded into a running car; an architecture-specific runtime-loading workflow is a separate integration.

## GGUF: `.gguf`

GGUF is commonly used by compatible language-model runtimes. The GUI can import a GGUF file and check its header, but the current application does not execute GGUF inference.

A racing controller maps a 36-value observation to two control outputs. A language model generates tokens from a prompt/context. Renaming an NPZ file to `.gguf` cannot bridge that architectural difference.

## Validation expectations

Before using a model bundle, validate:

- Bundle format and version
- Required files
- Architecture identifier
- Observation/action schema
- Parameter count and shape
- Numeric type and finite parameter values
- Compatibility between the architecture and its parameter vector

Do not load untrusted model archives into a privileged process. Treat imported files as untrusted input.
