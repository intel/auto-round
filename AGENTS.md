# AGENTS.md

Agent-facing notes for working on AutoRound.

## Project overview
- AutoRound is a Python toolkit for low-bit quantization of LLMs and VLMs.
- Primary entry points are CLI commands like `auto-round`, `auto-round-best`, and `auto-round-light`.

## Repository layout
- `auto_round/`: core library and quantization logic
- `docs/`: user guides and technical notes
- `test/`: unit tests
- `.azure-pipelines/`: CI scripts and test orchestration

## Architecture overview
- CLI entry points: `auto_round/__main__.py`
- Core quantization API: `auto_round/autoround.py`
- Compressor implementations: `auto_round/compressors/` (LLM/MLLM/Diffusion)
- Schemes and presets: `auto_round/schemes.py`
- Export formats/pipeline: `auto_round/formats.py`, `auto_round/export/`
- AutoScheme generator: `auto_round/auto_scheme/`
- Data type definitions: `auto_round/data_type/`
- Model-specific patches: `auto_round/modeling/`
- Shared utilities: `auto_round/utils/`
- Evaluation entry points: `auto_round/eval/`

## Setup
- Python 3.10+ is expected.
- Install for development: `pip install -e .`
- Install runtime deps only: `pip install -r requirements.txt`

## Build and test commands
- Build from source: `pip install .`
- Run unit tests: `pytest test`

## Common commands
- CLI help: `auto-round -h`
- List supported formats: `auto_round list format`

## Minimal examples
- CLI quantization:

```bash
auto-round --model Qwen/Qwen3-0.6B --scheme "W4A16" --format "auto_round" --output_dir ./tmp_autoround
```

- API quantization:

```python
from auto_round import AutoRound

ar = AutoRound("Qwen/Qwen3-0.6B", scheme="W4A16")
ar.quantize_and_save(output_dir="./qmodel", format="auto_round")
```

- GGUF export (single format):

```bash
auto-round --model Qwen/Qwen3-0.6B --scheme "W4A16" --format "gguf:q4_k_m" --output_dir ./tmp_autoround_gguf
```

- CompressedTensors export (LLM-Compressor format):

```bash
auto-round --model Qwen/Qwen3-0.6B --scheme "NVFP4" --format "llm_compressor" --output_dir ./tmp_autoround_ct
```

## Tests
- Quick local run: `pytest test`
- CI runs split CPU tests and optional LLM-compressor tests via
  `.azure-pipelines/scripts/ut/run_ut.sh` (uses `uv`, `numactl`, and
  installs extra deps). Expect longer runtimes and more system
  requirements.
- Prefer targeted tests for the area you changed.

## Code style and linting
- Python formatting is aligned with Black/Ruff, line length 120.
- Prefer double quotes for new strings (Ruff format default).
- Keep imports sorted (isort profile: black).

## Docs and translations
- If you change a markdown doc that has a `_CN` counterpart, update the
  Chinese file to keep content and structure aligned (for example,
  `README.md` and `README_CN.md`).

## Contributions
- DCO sign-off is required for commits (see `CONTRIBUTING.md`).

## Data and large files
- Avoid committing model weights, large binaries, or datasets.
- If you need sample data, prefer small fixtures or use public URLs.

## Common agent pitfalls
- Do not mix GGUF export with other formats; choose a single GGUF format.
- When editing a markdown doc with a `_CN` counterpart, update both files.
- Avoid running full CI locally unless needed; use targeted tests first.
- Do not add large artifacts (models, binaries, datasets) to the repo.
