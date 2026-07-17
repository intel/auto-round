# SVDQuant AWQ Smooth Search Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace SVDQuant's fixed-alpha smoothing with AWQ-style grid search using the actual terminal quantizer QDQ and output MSE.

**Architecture:** Extract AWQ's ratio generation, scale construction, and finite-error candidate selection into a shared internal service, while retaining `QDQTool` as the shared quantization dispatcher. AWQ keeps its existing cross-layer folding evaluator, while SVDQuant supplies a per-Linear evaluator, stores the selected runtime smooth, and performs SVD exactly once after search.

**Tech Stack:** Python 3.12, PyTorch, AutoRound algorithm pipeline, pytest, ruff, uv environment at `/home/user2/data/xixi/torch213-cu130-env/.venv`.

**Working-tree constraint:** The branch already contains unrelated uncommitted SVDQuant work. Never reset, checkout, or overwrite those edits. Before each commit, inspect the staged diff and stage only the task's intended hunks. If clean hunk isolation is not practical, defer the commit rather than including unrelated changes.

---

### Task 1: Replace the fixed-alpha configuration contract

**Files:**
- Modify: `test/test_cpu/algorithms/test_svdquant.py`
- Modify: `test/test_cpu/utils/test_cli_usage.py`
- Modify: `auto_round/algorithms/transforms/svdquant/config.py`
- Modify: `auto_round/cli/parser.py`
- Modify: `auto_round/cli/algorithms.py`
- Modify: `auto_round/autoround.py`
- Modify: `auto_round/compressors/entry.py`

**Step 1: Write failing configuration tests**

Add assertions equivalent to:

```python
config = SVDQuantConfig()
assert config.smooth_enabled is False
assert config.smooth_n_grid == 20
assert not hasattr(config, "smooth_alpha")

with pytest.raises(ValueError, match="smooth_n_grid"):
    SVDQuantConfig(smooth_n_grid=1)
with pytest.raises(ValueError, match="smooth_n_grid"):
    SVDQuantConfig(smooth_n_grid=True)
```

Change existing smoothing CLI tests to assert that
`--enable_svdquant_smooth --svdquant_smooth_n_grid 8` produces
`svdquant_smooth_enabled=True` and `svdquant_smooth_n_grid=8`. Add a parser test
that passing `--svdquant_smooth_alpha` raises `SystemExit` as an unknown option.

**Step 2: Run tests and confirm the old contract fails**

Run:

```bash
/home/user2/data/xixi/torch213-cu130-env/.venv/bin/python -m pytest \
  test/test_cpu/algorithms/test_svdquant.py \
  test/test_cpu/utils/test_cli_usage.py -k 'svdquant and (config or smooth)' -q
```

Expected: failures referencing missing `smooth_n_grid` and the still-present
`smooth_alpha` option.

**Step 3: Implement the new contract**

In `SVDQuantConfig`, replace `smooth_alpha` with `smooth_n_grid: int = 20`,
reject bool/non-int values and values below 2, expose it in `__repr__`, and keep
`requires_calibration = smooth_enabled`.

Replace the CLI option with:

```python
rt.add_argument(
    "--svdquant_smooth_n_grid",
    default=20,
    type=int,
    help="Number of AWQ ratio candidates used by SVDQuant smooth search.",
)
```

Thread `svdquant_smooth_n_grid` through `AlgorithmRegistry.build_configs`, the
compatibility entry point, and AutoRound's accepted keyword list. Remove every
runtime reference to `svdquant_smooth_alpha`; do not add an alias.

**Step 4: Run focused tests**

Run the command from Step 2. Expected: PASS.

**Step 5: Checkpoint**

Review `git diff` for the seven files. Commit only isolated task hunks with:

```bash
git commit -s -m "refactor: replace SVDQuant fixed smooth alpha"
```

---

### Task 2: Extract a shared AWQ scale-search service

**Files:**
- Create: `auto_round/algorithms/transforms/awq/scale_search.py`
- Create: `test/test_cpu/algorithms/test_activation_aware_scale_search.py`
- Modify: `auto_round/algorithms/transforms/awq/__init__.py`

**Step 1: Write failing service tests**

Cover:

- `n_grid=4, duo_scaling=True` generates deterministic ratios from 0 through 1;
- activation-only and duo-scaling formulas match the existing AWQ formulas;
- scale normalization produces finite, positive values;
- the first minimum wins ties;
- non-finite candidates are skipped;
- all-candidate failure raises a module-qualified `ValueError`; and
- the service invokes the supplied evaluator once per valid candidate.

Use a small public internal interface:

```python
search = ActivationAwareScaleSearch(n_grid=4, duo_scaling=True)
result = search.search(
    module_name="model.layers.0.proj",
    x_mean=x_mean,
    w_mean=w_mean,
    evaluate=lambda scale: mse_for(scale),
)
assert result.scale.shape == x_mean.shape
assert 0.0 <= result.ratio <= 1.0
```

**Step 2: Run the new test and confirm import failure**

```bash
/home/user2/data/xixi/torch213-cu130-env/.venv/bin/python -m pytest \
  test/test_cpu/algorithms/test_activation_aware_scale_search.py -q
```

Expected: FAIL because `scale_search.py` does not exist.

**Step 3: Implement the service**

Create immutable `ScaleSearchResult(scale, ratio, error, use_duo_scaling)` and
`ActivationAwareScaleSearch`. Move the following behavior out of
`AWQTransform` without changing its formulas:

```python
ratio_grid = idx / (n - 1)
scale = x_mean.pow(ratio) / (w_mean.pow(1 - ratio) + 1e-4)
scale = scale / (scale.max() * scale.min()).sqrt()
```

The service owns candidate iteration, finite checks, evaluator invocation,
strictly-lower-error selection, and final failure reporting. Keep tensor/device
ownership explicit and do not mutate caller weights.

**Step 4: Run service tests**

Expected: PASS.

**Step 5: Checkpoint**

```bash
git commit -s -m "refactor: extract activation-aware scale search"
```

---

### Task 3: Refactor AWQ to consume the shared service

**Files:**
- Modify: `auto_round/algorithms/transforms/awq/base.py`
- Modify: `test/test_cpu/algorithms/test_awq.py`

**Step 1: Add regression tests around delegation**

Add a spy test proving AWQ calls `ActivationAwareScaleSearch.search` for a
resolved mapping and applies the returned scale. Preserve tests for MXFP4 QDQ,
duo-scaling modes, parent-forward loss, and fallback local loss.

**Step 2: Run the AWQ tests before refactoring**

```bash
/home/user2/data/xixi/torch213-cu130-env/.venv/bin/python -m pytest \
  test/test_cpu/algorithms/test_awq.py -q
```

Expected: the new delegation test fails; existing tests pass.

**Step 3: Delegate candidate iteration to the service**

Instantiate the shared service from `AWQConfig.n_grid` and `duo_scaling`.
Retain AWQ-specific responsibilities in `AWQTransform`:

- resolved smooth/balance mappings;
- parent input/kwargs capture;
- `QDQTool` configuration and per-layer params;
- temporary candidate-weight mutation and restoration;
- parent-forward or local-output MSE evaluator; and
- applying the selected scale to upstream and balance layers.

Delete `_get_grid_search_params` and duplicate candidate-selection code only
after all call sites use the service. Keep `QDQTool` in `awq/qdq.py` as the
single quantization dispatcher.

**Step 4: Run AWQ and shared-service tests**

Expected: PASS with no changed AWQ numerical assertions.

**Step 5: Checkpoint**

```bash
git commit -s -m "refactor: share AWQ scale search engine"
```

---

### Task 4: Add bounded per-Linear calibration capture for SVDQuant

**Files:**
- Modify: `auto_round/algorithms/transforms/svdquant/apply.py`
- Modify: `test/test_cpu/algorithms/test_svdquant.py`

**Step 1: Replace amax tests with input-capture tests**

Test that smoothing-enabled hooks:

- flatten inputs to `[tokens, in_features]`;
- retain deterministic bounded samples on CPU;
- append inputs from multiple calibration forwards up to the bound;
- keep separate buffers per Linear;
- reject incompatible input shapes; and
- release buffers after successful or failed block preprocessing.

Keep the existing test proving smoothing-disabled execution installs no hooks.

**Step 2: Run focused tests and confirm failure**

```bash
/home/user2/data/xixi/torch213-cu130-env/.venv/bin/python -m pytest \
  test/test_cpu/algorithms/test_svdquant.py -k 'smooth or activation or capture' -q
```

**Step 3: Implement bounded capture**

Replace `_act_max` with a per-module CPU input buffer. Use the same deterministic
token cap as the shared AWQ search service. Register a forward hook only for
targeted Linear modules. Always pop a module's buffer before decomposition so
exceptions cannot retain large tensors.

Add `bind`/runtime wiring needed to configure the shared `QDQTool` from the
terminal quantizer and attach the resolved `layer_config` before search.

**Step 4: Run focused tests**

Expected: PASS and no retained input buffers.

**Step 5: Checkpoint**

```bash
git commit -s -m "feat: capture SVDQuant smooth calibration inputs"
```

---

### Task 5: Search SVDQuant runtime smooth with MXFP4 QDQ

**Files:**
- Modify: `auto_round/algorithms/transforms/svdquant/apply.py`
- Modify: `test/test_cpu/algorithms/test_svdquant.py`

**Step 1: Write failing numerical tests**

Construct a small Linear, deterministic calibration inputs, and a fake QDQ
function where one candidate ratio has a known lower output MSE. Assert:

- the selected runtime `smooth` is the reciprocal of the winning scale;
- candidate evaluation uses `x / scale` and `QDQ(W * scale)`;
- bias is unchanged;
- the transformed floating-point wrapper preserves the original forward before
  residual quantization;
- per-layer quantization overrides reach `QDQTool.resolve_params`;
- no usable calibration input fails with the global module name;
- all non-finite candidates fail clearly; and
- `_truncated_svd` is called exactly once per Linear even when `n_grid > 2`.

**Step 2: Run numerical tests and confirm failure**

```bash
/home/user2/data/xixi/torch213-cu130-env/.venv/bin/python -m pytest \
  test/test_cpu/algorithms/test_svdquant.py -k 'smooth or qdq or svd_once' -q
```

**Step 3: Implement `_search_smooth`**

For each target Linear:

```python
reference = F.linear(inputs, weight, bias)
x_mean = inputs.abs().mean(dim=0)
w_mean = compute_normalized_weight_mean([module], group_size)

def evaluate(scale):
    scaled_weight = weight * scale.view(1, -1)
    qdq_weight = qdq_tool.qdq(scaled_weight, resolved_params, ...)
    candidate = F.linear(inputs / scale, qdq_weight, bias)
    return F.mse_loss(candidate.float(), reference.float())
```

Use `ActivationAwareScaleSearch` to select `scale`, return
`smooth = reciprocal(scale)`, and then execute the existing SVD/residual path
once against `weight_hat = weight / smooth`.

Do not add AWQ mappings or fold scales into neighboring layers. Do not run AWQ
clipping. Do not import DeepCompressor or Nunchaku.

**Step 4: Run all SVDQuant tests**

```bash
/home/user2/data/xixi/torch213-cu130-env/.venv/bin/python -m pytest \
  test/test_cpu/algorithms/test_svdquant.py -q
```

Expected: PASS.

**Step 5: Checkpoint**

```bash
git commit -s -m "feat: search SVDQuant smooth scales with AWQ QDQ"
```

---

### Task 6: Update compatibility paths, scripts, and documentation

**Files:**
- Modify: `scripts/quantize_flux_svdquant_nunchaku.py`
- Modify: `docs/svdquant_nunchaku_mxfp4_review.md`
- Modify: relevant existing runbook under `/home/user2/data/xixi`
- Modify: `test/test_cpu/core/test_pipeline_fail_fast.py`
- Modify: `test/test_cpu/utils/test_cli_usage.py`

**Step 1: Add compatibility-path tests**

Verify the legacy constructor/entry function forwards `smooth_n_grid`, does not
forward `smooth_alpha`, and smoothing-disabled RTN remains calibration-free.

**Step 2: Run tests and confirm stale references fail**

```bash
/home/user2/data/xixi/torch213-cu130-env/.venv/bin/python -m pytest \
  test/test_cpu/core/test_pipeline_fail_fast.py \
  test/test_cpu/utils/test_cli_usage.py -k svdquant -q
```

**Step 3: Remove stale fixed-alpha usage**

Run:

```bash
rg -n 'svdquant_smooth_alpha|smooth_alpha' auto_round scripts test docs
```

Remove only SVDQuant fixed-alpha references. Do not remove unrelated SmoothQuant
or SpinQuant alpha concepts. Update examples to show
`--enable_svdquant_smooth --svdquant_smooth_n_grid 20`.

**Step 4: Run compatibility tests**

Expected: PASS.

**Step 5: Checkpoint**

```bash
git commit -s -m "docs: document SVDQuant AWQ smooth search"
```

---

### Task 7: Full verification and FLUX smoke validation

**Files:**
- No code changes expected
- Output log: `/home/user2/data/xixi/autoround-flux-mxfp4-r32-awq-smooth-smoke.log`

**Step 1: Run focused algorithm tests**

```bash
/home/user2/data/xixi/torch213-cu130-env/.venv/bin/python -m pytest \
  test/test_cpu/algorithms/test_activation_aware_scale_search.py \
  test/test_cpu/algorithms/test_awq.py \
  test/test_cpu/algorithms/test_svdquant.py \
  test/test_cpu/core/test_pipeline_fail_fast.py \
  test/test_cpu/utils/test_cli_usage.py -q
```

Expected: PASS.

**Step 2: Run export regression tests**

```bash
/home/user2/data/xixi/torch213-cu130-env/.venv/bin/python -m pytest \
  test/test_cpu/export/test_svdquant_nunchaku_export.py \
  test/test_cpu/export/test_svdquant_nunchaku_format.py -q
```

Expected: PASS with unchanged runtime smooth tensor names and shapes.

**Step 3: Run lint on touched Python files**

```bash
/home/user2/data/xixi/torch213-cu130-env/.venv/bin/python -m ruff check \
  auto_round/algorithms/transforms/awq \
  auto_round/algorithms/transforms/svdquant \
  auto_round/cli/algorithms.py auto_round/cli/parser.py \
  auto_round/compressors/entry.py auto_round/autoround.py \
  test/test_cpu/algorithms/test_activation_aware_scale_search.py \
  test/test_cpu/algorithms/test_awq.py test/test_cpu/algorithms/test_svdquant.py
```

Expected: no lint errors.

**Step 4: Run a minimal FLUX calibration smoke on an actually free GPU**

Use one sample and one diffusion step first:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
/home/user2/data/xixi/torch213-cu130-env/.venv/bin/python -u -m auto_round \
  --model /home/user2/data/xixi/FLUX.1-dev \
  --model_dtype bf16 \
  --scheme MXFP4 \
  --algorithm rtn --iters 0 --disable_opt_rtn \
  --nsamples 1 --batch_size 1 --dataset coco2014 --num_inference_steps 1 \
  --enable_svdquant --svdquant_rank 32 \
  --enable_svdquant_smooth --svdquant_smooth_n_grid 4 \
  --svdquant_residual_iters 1 \
  --svdquant_model_adapter flux \
  --format svdquant_nunchaku \
  --device 0 --low_gpu_mem_usage --disable_low_cpu_mem_usage \
  --output_dir /home/user2/data/xixi/autoround-flux-mxfp4-r32-awq-smooth-smoke \
  2>&1 | tee /home/user2/data/xixi/autoround-flux-mxfp4-r32-awq-smooth-smoke.log
```

Before running, require sufficient GPU headroom and at least 18GB free disk.
Expected: calibration enters AWQ smooth candidate search, completes quantization,
exports the pipeline, and Nunchaku loads the exported transformer.

**Step 5: Inspect final diff and report residual risk**

Confirm no fixed-alpha SVDQuant references remain, no generated model files are
staged, and all pre-existing unrelated working-tree edits remain intact. Report
the smoke artifact path, selected-search log evidence, test counts, runtime, and
any skipped full-quality FLUX run.
