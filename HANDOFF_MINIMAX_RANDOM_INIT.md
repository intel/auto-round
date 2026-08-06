# AutoRound Random-Init Diffusion Handoff

## Scope

This handoff covers the work done to enable an API path for quantizing a local diffusion-style repository with random-initialized transformer weights, intended for MiniMax-style layouts where the full checkpoints are not yet available.

Target behavior:

- local path input
- diffusion model path
- `init_mode="random"`
- `scheme="W4A16"`
- `iters=0`
- `disable_opt_rtn=True`
- `model_free=False`

Current status:

- implemented
- syntax-checked
- directly validated with a dummy local MiniMax-style repo
- not yet validated against real MiniMax checkpoints

## Repo And Environment

Machine:

- `a100` (`yi4l@mlp-dgx-01.sh.intel.com`)

Repo:

- `~/workspace/auto-round`

Branch / commit at handoff time:

- branch: `main`
- base commit: `a2f6b3161ab3fadb723e82513bf13bd4e1e9e2f0`

Important environment notes:

- `diffusers` is not installed in `~/workspace/auto-round/.venv`
- `pytest` is not installed in `~/workspace/auto-round/.venv`
- validation was done with direct Python scripts and monkeypatched fake `diffusers` classes

## Files Changed

Code:

- `auto_round/autoround.py`
- `auto_round/compressors/base.py`
- `auto_round/context/model.py`
- `auto_round/utils/model.py`

Tests:

- `test/test_cpu/core/test_entry_contract.py`
- `test/test_cpu/models/test_diffusion.py`
- `test/test_cpu/quantization/test_model_free.py`

Artifacts created for validation:

- `artifacts/minimax_dummy_random_init/source`
- `artifacts/minimax_dummy_random_init/quantized_w4a16`

## What Was Implemented

### 1. New Entry Option

`init_mode` is now threaded through the AutoRound entry path and model loading path.

Supported values:

- `pretrained` (default)
- `random`

### 2. Validation Rules For `init_mode="random"`

The new mode is intentionally narrow. It currently accepts only:

- local directory path
- diffusion repository layout
- fixed non-auto scheme
- pure zero-shot RTN flow

It rejects:

- `model_free=True`
- non-local model paths
- non-diffusion repos
- `AutoScheme`
- activation-calibration-required schemes
- anything other than pure RTN zero-shot behavior

Practical meaning:

- use `iters=0`
- use `disable_opt_rtn=True`
- use a weight-only zero-shot scheme such as `W4A16`

### 3. Random-Init Diffusion Loader

For `init_mode="random"`, the diffusion loader does not call `DiffusionPipeline.from_pretrained(...)`.

Instead it:

- reads `model_index.json`
- instantiates only transformer-like components from their local `config.json`
- keeps non-transformer components as metadata-copy proxies
- preserves root metadata when saving

Current assumption:

- a primary `transformer/` component must exist

### 4. Model-Free Routing

`is_model_free_route(...)` now returns `False` for `init_mode="random"`.

This forces the flow through the new non-model-free path.

## Dummy Quantized Artifact

A full end-to-end dummy quantization was run successfully on `a100`.

Source repo:

- `/home/yi4l/workspace/auto-round/artifacts/minimax_dummy_random_init/source`

Quantized output:

- `/home/yi4l/workspace/auto-round/artifacts/minimax_dummy_random_init/quantized_w4a16`

Saved files:

- `README.md`
- `model_index.json`
- `processor/processor_config.json`
- `tokenizer/tokenizer_config.json`
- `transformer/config.json`
- `transformer/model.safetensors`
- `transformer/quantization_config.json`

Output size:

- about `140K`

Why it is small:

- this is a tiny dummy transformer, not the real MiniMax model
- the dummy config used `hidden_size=128` and `num_hidden_layers=2`

Quantization config summary:

- `bits=4`
- `group_size=128`
- `data_type=int`
- `sym=true`
- `quant_method=auto-round`
- `packing_format=auto_round:auto_gptq`

## Verification Completed

Completed:

- `python -m py_compile` on edited modules and tests
- direct validation script for:
  - entry kwarg routing
  - model-free route rejection
  - random-init diffusion loading
  - metadata preservation on save
  - missing-transformer failure case
- full dummy end-to-end quantize-and-save run

Not completed:

- real `pytest` run, because `pytest` is not installed in `.venv`
- real diffusers-backed load, because `diffusers` is not installed in `.venv`
- quantization with actual MiniMax checkpoints

## Known Limitations

### 1. Real MiniMax Weights Are Still Missing

The real HF download was not completed during this work. The code path is ready, but the exported artifact produced so far is only a dummy proof artifact.

### 2. Current Random-Init Mode Is Intentionally Restricted

This is not a general random-init loader for all diffusion repos yet. It is a narrow functionality-first implementation.

### 3. Only Primary `transformer` Is Required

The current implementation expects a primary `transformer/` directory. Multi-transformer cases may need extra validation when real MiniMax checkpoints are available.

### 4. Test Environment Is Incomplete

To run the actual tests in the repo virtual environment, install at least:

- `pytest`
- `diffusers`

## Recommended Next Steps

### Option A: Continue Functionality Work First

1. Install missing runtime/test packages in `~/workspace/auto-round/.venv`.
2. Run the new targeted tests under real `pytest`.
3. Keep the dummy artifact as a regression check for the random-init path.

### Option B: Move To Real MiniMax Checkpoints

1. Finish downloading the actual MiniMax repository contents.
2. Confirm the repo layout matches the assumptions:
   - `model_index.json`
   - `transformer/`
   - component subfolders for tokenizer / processor / VAE / encoder pieces
3. Run the same AutoRound invocation against the real local repo.
4. Check whether real MiniMax uses:
   - a single `transformer`
   - multiple transformer-like components
   - any config shapes that require loader adjustments

## Suggested Commands

### Check current repo state

```bash
ssh a100
cd ~/workspace/auto-round
git status --short
```

### Re-run syntax check

```bash
cd ~/workspace/auto-round
. .venv/bin/activate
python -m py_compile \
  auto_round/autoround.py \
  auto_round/compressors/base.py \
  auto_round/context/model.py \
  auto_round/utils/model.py \
  test/test_cpu/core/test_entry_contract.py \
  test/test_cpu/models/test_diffusion.py \
  test/test_cpu/quantization/test_model_free.py
```

### Install missing test/runtime packages

```bash
cd ~/workspace/auto-round
. .venv/bin/activate
pip install pytest diffusers
```

### Run targeted tests

```bash
cd ~/workspace/auto-round
. .venv/bin/activate
python -m pytest -q \
  test/test_cpu/core/test_entry_contract.py::test_split_entry_kwargs_partitions_owned_fields \
  test/test_cpu/quantization/test_model_free.py::test_model_free_route_rejects_random_init \
  test/test_cpu/models/test_diffusion.py::test_diffusion_random_init_loads_transformer_and_preserves_metadata \
  test/test_cpu/models/test_diffusion.py::test_diffusion_random_init_requires_transformer_dir
```

### Shape Of The Intended Real Invocation

Example API call:

```python
from auto_round import AutoRound

autoround = AutoRound(
    "/path/to/local/MiniMax-style-repo",
    tokenizer=None,
    scheme="W4A16",
    iters=0,
    disable_opt_rtn=True,
    model_free=False,
    init_mode="random",
    device_map="cpu",
    enable_torch_compile=False,
)

autoround.quantize_and_save("/path/to/output")
```

## Notes About The Dummy Model

The dummy model used for validation was intentionally small and fake. It monkeypatched a local `diffusers` module object and used a minimal transformer implementation with:

- `transformer_blocks` as a `ModuleList`
- simple linear sublayers
- a local `save_pretrained(...)`

This was only to validate the AutoRound control flow, block discovery, save path, and export behavior.

## Handoff Summary

If you continue on another node, the most direct path is:

1. open `~/workspace/auto-round`
2. install `pytest` and `diffusers` in `.venv`
3. run the targeted tests
4. finish downloading the real MiniMax checkpoints
5. run the real local random-init quantization
6. inspect whether real MiniMax needs support beyond the current single-primary-transformer assumption
