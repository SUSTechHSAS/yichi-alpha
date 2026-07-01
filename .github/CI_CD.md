# GitHub Actions CI/CD

This directory contains GitHub Actions workflow definitions for the YichiAlpha project.

## Workflows

### `build.yml` — Build & Test

Runs automatically on:
- Push to `main` / `master`
- Pull request to `main` / `master`
- Manual dispatch (via GitHub Actions UI)

It contains 4 jobs:

| Job | Purpose | Runner | Approx. time |
|-----|---------|--------|--------------|
| **python-tests** | Run game engine unit tests, verify model/MCTS/self-play, smoke-train 2 iterations, evaluate | `ubuntu-latest` × Python 3.11/3.12/3.13 | ~5-8 min |
| **cpp-build** | Install CMake + LibTorch (CPU), compile `yichi_selfplay`, smoke-test 1 game | `ubuntu-latest`, `ubuntu-22.04` | ~3-5 min (cached: ~1 min) |
| **lint** | flake8 (Python) + cpplint (C++) — best-effort, non-blocking | `ubuntu-latest` | ~30 sec |
| **rule-alignment** | Fetch original game HTML, extract JS, run Node.js to generate 50 random games, compare Python engine state-by-state | `ubuntu-latest` | ~1-2 min |

## Artifacts

Each successful run uploads:
- `python-checkpoints-py3{11,12,13}` — Trained model checkpoints + training logs from each Python version
- `yichi_selfplay-{ubuntu-latest,ubuntu-22.04}` — Compiled C++ binaries

Artifacts are retained for 7 days.

## Caching

- **LibTorch**: Cached by `actions/cache@v4` keyed on version + OS. First run downloads ~123MB; subsequent runs use cache (~1 min vs ~5 min).
- **pip**: Cached by `actions/setup-python@v5` keyed on `requirements.txt`.

## Local testing of the workflow

To test the workflow locally before pushing, use [`act`](https://github.com/nektos/act):

```bash
# Install act
brew install act   # macOS
# or: curl -s https://raw.githubusercontent.com/nektos/act/master/install.sh | bash

# Run a specific job locally
act -W .github/workflows/build.yml -j python-tests

# Run all jobs
act -W .github/workflows/build.yml
```

## Adding new test scripts

If you add a new test script in `python/` (e.g. `test_mcts.py`), add a step to the `python-tests` job:

```yaml
- name: Run new test
  run: python test_mcts.py
```

## Status badge

Add this to the top of `README.md` to show CI status:

```markdown
![Build & Test](https://github.com/SUSTechHSAS/yichi-alpha/actions/workflows/build.yml/badge.svg)
```
