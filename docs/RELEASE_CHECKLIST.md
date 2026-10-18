# Release checklist

Complete this checklist before making the repository public or tagging the first release.

## Repository hygiene

- [ ] Confirm the correct repository and default branch.
- [ ] Confirm all intended work is committed and the working tree is clean.
- [ ] Inspect `git status --short` and `git diff --check`.
- [ ] Review `.gitignore`; remove virtual environments, caches, generated results, and local machine files from tracking.
- [ ] Search history and current files for API keys, tokens, credentials, and personal information.
- [ ] Confirm no copyrighted or third-party assets are included without permission.

## Documentation

- [ ] Replace `<YOUR_REPOSITORY_URL>` in the README and getting-started guide.
- [ ] Verify every command and option against the actual checked-out scripts.
- [ ] Add project screenshots only after capturing them from the real application.
- [ ] Add third-party acknowledgements and references.
- [ ] Confirm project status and limitations are accurate.
- [ ] Add a `CITATION.cff` if you want the research project to be citable.
- [ ] Choose and add an explicit license before claiming the project is open source.
- [ ] Add a private security contact in GitHub settings if needed.

## Validation

- [ ] Run `PYTHONPATH=src pytest -q` on the release commit.
- [ ] Record the exact test result and Python version.
- [ ] Launch the GUI from a fresh environment.
- [ ] Open the population workbench.
- [ ] Import and export one `.npz` / `.nrlmodel` sample.
- [ ] Export sample PDF, DOCX, and PPTX reports.
- [ ] Verify no sample contains private paths or personal information.
- [ ] Confirm README quick-start commands work on a fresh checkout.

## GitHub release

- [ ] Set repository description and topics.
- [ ] Add README, contributing guide, security policy, and issue templates.
- [ ] Configure a CI workflow after confirming the supported Python/dependency matrix.
- [ ] Add a license file and confirm the selected license fits all dependencies/assets.
- [ ] Create a version tag only after validation.
- [ ] Write release notes with implemented features, known limitations, and reproducibility caveats.
- [ ] Do not attach large checkpoints unless their license, provenance, and usefulness are clear.

## Suggested first release wording

Describe this as an **experimental research prototype**. State that the current pilot controllers did not complete evaluation laps under the recorded five-seed oval evaluation. Do not market the system as a solved autonomous-driving agent.
