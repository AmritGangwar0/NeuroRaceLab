# Publishing notes

This documentation pack intentionally does not publish or push anything to GitHub. Review the files against the actual repository first.

## Local integration

1. Extract this patch at the repository root.
2. Review the README commands against `--help` output from the current scripts.
3. Merge the `.gitignore.additions` entries into the existing `.gitignore`; do not replace the existing file blindly.
4. Decide on a license before making the repository public.
5. Run the release checklist.

## Suggested Git workflow

```bash
git status --short
git diff --check
PYTHONPATH=src pytest -q

# Review each file before staging.
git add README.md CONTRIBUTING.md SECURITY.md docs .github
git diff --cached --check
git diff --cached --stat
git commit -m "docs: prepare NeuroRaceLab for public release"
git status --short
```

Do not add the documentation patch ZIP, generated reports, local model weights, or temporary patch directories unless intentionally required.

## Publish after review

If the GitHub repository already exists, push the reviewed branch using the repository's normal remote and branch workflow. If it does not exist, create it on GitHub, select the license and visibility, add the remote, and push.

Do not make the repository public until the license, Git history, tracked artifacts, and README claims have been checked. Public visibility without a license is not a substitute for deciding what other people are legally allowed to do with the code.
