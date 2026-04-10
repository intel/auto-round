# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

AutoRound — post-training quantization for LLMs/VLMs using sign-gradient descent. Publishes as `auto-round` (GPU/CPU) and `auto-round-hpu` (Intel Gaudi).

## Build & Install

```bash
# From source (GPU/CPU) — --no-build-isolation is required when PyTorch is already installed
pip install --no-build-isolation -e .

# HPU variant
BUILD_HPU_ONLY=1 pip install --no-build-isolation .
# or: python setup.py hpu install

# XPU variant — install Intel PyTorch first
pip install torch --index-url https://download.pytorch.org/whl/xpu
pip install --no-build-isolation .
```

## Testing

```bash
# CPU tests (most common during development)
pytest test/test_cpu/ -x -q

# Single test
pytest test/test_cpu/ -k "test_name" -x -q

# Hardware-specific
pytest test/test_cuda/
pytest test/test_hpu/ --mode=lazy   # or --mode=compile
pytest test/test_xpu/
```

Test fixtures create tiny models (OPT-125M, Qwen-0.6B) at session scope — first run downloads them.

## Code Style

- **Line length: 120** (non-default) — enforced by black, isort, ruff, pylint
- **Formatter: black** (profile used by isort)
- **Import sorting: isort** with `profile=black`, first-party: `auto_round`, `auto_round_extension`
- **Linter: ruff** — rules `E4, E7, E9, F, NPY, FURB`; E501/E402/F401/F403 are intentionally ignored
- **License header**: Apache 2.0 auto-inserted into `.py/.yaml/.yml/.sh` under `auto_round/` and `auto_round_extension/`
- Pre-commit config: `.pre-commit-config.yaml`

## Commit & PR Conventions

- Conventional commits: `feat:`, `fix:`, `chore:`, `docs:`, `refactor:`, `test:`
- PRs target `main`, squash-merged
- **CN docs rule**: any change to a `.md` file must include a matching update to its `_CN` counterpart (e.g., `README.md` → `README_CN.md`)

## Key Environment Variables

- `BUILD_HPU_ONLY=1` — build HPU package variant
- `AR_USE_MODELSCOPE=1` — use ModelScope instead of HuggingFace for model downloads
- `FORCE_BF16=1` — force BF16 in tests (used in CI)

## Source Layout

- `auto_round/` — core library (AutoRound class, sign-SGD, exporters, eval, data types)
- `auto_round_extension/` — hardware backends (CUDA, HPU, IPEX/XPU, Triton, ARK, vLLM)
- `test/` — tests organized by hardware: `test_cpu/`, `test_cuda/`, `test_hpu/`, `test_xpu/`
- `examples/` — usage examples for different model types

## Gotchas

- `setup.py` forces `CC=CXX=g++` at import time
- Version is computed dynamically from git tags — untagged commits produce dev versions
- Some test dependencies (AutoAWQ, GPTQModel, llama-cpp) require manual git installs — see comments in `test/test_cuda/requirements.txt`
