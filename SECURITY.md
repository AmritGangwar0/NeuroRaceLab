# Security policy

## Scope

NeuroRaceLab is a research and simulation project. Security issues may include unsafe handling of imported model archives, path traversal during file import/export, unintended command execution, or sensitive data accidentally included in repository artifacts.

## Reporting a vulnerability

Do not publish exploit details in a public issue if they could affect users. Contact the repository maintainer privately through the security contact configured in the GitHub repository. Add that contact before making the repository public.

## Safe handling

- Treat imported `.npz`, `.nrlmodel`, `.gguf`, JSON, and CSV files as untrusted.
- Avoid loading NumPy object arrays from untrusted archives; use `allow_pickle=False`.
- Validate archive member names and expected schema before extraction.
- Do not run arbitrary commands or scripts contained in model metadata.
- Do not commit access tokens, credentials, private data, or local environment files.
