# SVDQuant Low-Rank-Aware Smooth Search Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace fixed-alpha SVDQuant smoothing with a layer-output grid search that evaluates every scale using a low-rank branch plus QDQ residual.

**Architecture:** SVDQuant discovers projection groups, captures bounded shared-projection and parent-module calibration inputs, generates the alpha/beta grid from AbsMax spans, and scores each scale after one temporary shared low-rank decomposition plus per-projection residual QDQ. The winning scale is stored in the existing runtime-smooth direction, then grouped final residual iteration produces shared-down/split-up branches for export.

**Tech Stack:** Python 3.12, PyTorch, AutoRound quantization pipeline, pytest, ruff, uv environment at `/home/user2/data/xixi/torch213-cu130-env/.venv`.

**Working-tree constraint:** The branch contains existing uncommitted SVDQuant work. Never reset or overwrite it. Inspect every diff and stage only intended hunks; defer a checkpoint commit when hunk isolation is unsafe.

---

### Task 1: Replace Fixed-Alpha Configuration

**Files:**
- Modify: `test/test_cpu/algorithms/test_svdquant.py`
- Modify: `test/test_cpu/utils/test_cli_usage.py`
- Modify: `auto_round/algorithms/transforms/svdquant/config.py`
- Modify: `auto_round/cli/parser.py`
- Modify: `auto_round/cli/algorithms.py`
- Modify: `auto_round/autoround.py`
- Modify: `auto_round/compressors/entry.py`

**Step 1: Add failing contract tests**

Assert:

```python
config = SVDQuantConfig()
assert config.smooth_enabled is False
assert config.smooth_num_grids == 20
assert not hasattr(config, "smooth_alpha")

for value in (True, 1, 1.5, None):
    with pytest.raises(ValueError, match="smooth_num_grids"):
        SVDQuantConfig(smooth_num_grids=value)
```

Add CLI coverage for:

```text
--enable_svdquant_smooth --svdquant_smooth_num_grids 8
```

and verify `--svdquant_smooth_alpha` is rejected as unknown.

**Step 2: Confirm failures**

```bash
/home/user2/data/xixi/torch213-cu130-env/.venv/bin/python -m pytest \
  test/test_cpu/algorithms/test_svdquant.py \
  test/test_cpu/utils/test_cli_usage.py -k 'svdquant and (config or smooth)' -q
```

Expected: failures for the old alpha contract.

**Step 3: Implement the new contract**

Replace `smooth_alpha` with `smooth_num_grids: int = 20`, validate integer type
and minimum 2, and thread it through every constructor and compatibility entry.
Add only this CLI option:

```python
rt.add_argument(
    "--svdquant_smooth_num_grids",
    default=20,
    type=int,
    help="Number of candidates per SVDQuant smooth search grid family.",
)
```

Do not add aliases or a mode switch.

**Step 4: Run focused tests**

Expected: PASS.

**Step 5: Checkpoint**

```bash
git commit -s -m "refactor: replace fixed SVDQuant smoothing config"
```

---

### Task 2: Implement Candidate And Span Primitives

**Files:**
- Create: `auto_round/algorithms/transforms/svdquant/smooth.py`
- Create: `test/test_cpu/algorithms/test_svdquant_smooth.py`

**Step 1: Add failing pure-function tests**

Test exact candidate order:

```python
assert build_alpha_beta_candidates(4) == [
    (0.0, 0.0),
    (0.25, 0.0),
    (0.5, 0.0),
    (0.75, 0.0),
    (0.25, 0.75),
    (0.5, 0.5),
    (0.75, 0.25),
]
assert len(build_alpha_beta_candidates(20)) == 39
```

Test independent references for:

```python
x_span = inputs.abs().reshape(-1, in_features).amax(dim=0)
w_span = weight.abs().amax(dim=0)
scale = x_span.pow(alpha) / w_span.pow(beta)
```

Cover zero-to-one normalization, identity candidate, whole-scale identity
fallback for NaN/Inf, and later-minimum tie selection.

**Step 2: Confirm import failure**

```bash
/home/user2/data/xixi/torch213-cu130-env/.venv/bin/python -m pytest \
  test/test_cpu/algorithms/test_svdquant_smooth.py -q
```

**Step 3: Implement pure primitives**

Create:

```python
@dataclass(frozen=True)
class SmoothCandidate:
    alpha: float
    beta: float
    scale: torch.Tensor


def build_alpha_beta_candidates(num_grids: int) -> list[tuple[float, float]]: ...
def absmax_channel_span(tensor: torch.Tensor, channels_dim: int) -> torch.Tensor: ...
def build_smooth_scale(x_span, w_span, alpha, beta) -> torch.Tensor: ...
```

Keep this module independent of AWQ and external compressor packages.

**Step 4: Run tests**

Expected: PASS.

**Step 5: Checkpoint**

```bash
git commit -s -m "feat: add SVDQuant smooth search primitives"
```

---

### Task 3: Define Search Groups And Flux Discovery

**Files:**
- Create: `auto_round/algorithms/transforms/svdquant/smooth_adapters/__init__.py`
- Create: `auto_round/algorithms/transforms/svdquant/smooth_adapters/base.py`
- Create: `auto_round/algorithms/transforms/svdquant/smooth_adapters/flux.py`
- Create: `test/test_cpu/algorithms/test_svdquant_smooth_adapters.py`
- Modify: `auto_round/algorithms/transforms/svdquant/apply.py`

**Step 1: Add failing group-discovery tests**

Define tiny Flux-shaped blocks and assert discovery produces:

- one shared group for Q/K/V projections;
- one shared group for added Q/K/V projections when present;
- attention output projection groups;
- feed-forward up/gate and down projection groups as applicable;
- stable global names and cache keys;
- parent evaluation module and output normalizer; and
- single-Linear fallback groups for an unknown block type.

Assert every grouped Linear has the same input width and appears in exactly one
group for the current smoothing phase.

**Step 2: Confirm failures**

```bash
/home/user2/data/xixi/torch213-cu130-env/.venv/bin/python -m pytest \
  test/test_cpu/algorithms/test_svdquant_smooth_adapters.py -q
```

**Step 3: Implement the neutral adapter protocol**

Create `SmoothSearchGroup` with projection modules/names, input source,
evaluation module, evaluation input source, kwargs filtering, and output
normalization. Register adapters by model/block type without importing the
export adapter package. Implement Flux discovery first and a generic
single-Linear fallback.

**Step 4: Run focused tests**

Expected: PASS.

**Step 5: Checkpoint**

```bash
git commit -s -m "feat: discover SVDQuant smooth search groups"
```

---

### Task 4: Capture Bounded Group And Parent Inputs

**Files:**
- Modify: `auto_round/algorithms/transforms/svdquant/apply.py`
- Modify: `test/test_cpu/algorithms/test_svdquant.py`
- Modify: `test/test_cpu/algorithms/test_svdquant_smooth_adapters.py`

**Step 1: Add failing capture tests**

Test that smoothing-enabled calibration captures:

- shared projection inputs for each search group;
- parent evaluation inputs and supported kwargs;
- normalized floating-point parent outputs;
- deterministic bounded CPU samples;
- repeated forwards without exceeding sample/token caps; and
- no duplicate hooks for grouped projections.

Assert every cache is released after successful or failing preprocessing.
Preserve the disabled-smoothing test asserting zero hooks and zero buffers.

**Step 2: Confirm failures**

```bash
/home/user2/data/xixi/torch213-cu130-env/.venv/bin/python -m pytest \
  test/test_cpu/algorithms/test_svdquant.py \
  test/test_cpu/algorithms/test_svdquant_smooth_adapters.py \
  -k 'smooth or capture or buffer or group' -q
```

**Step 3: Implement block-local capture**

Store a `SmoothGroupCalibration` per group. Use hooks and the existing
`BlockContext`/`BlockIO` forwarding APIs rather than retaining the whole model's
activations. Pop calibration ownership before search and restore/remove every
hook in `finally` blocks.

**Step 4: Run focused tests**

Expected: PASS.

**Step 5: Checkpoint**

```bash
git commit -s -m "feat: capture grouped SVDQuant smooth calibration"
```

---

### Task 5: Implement Grouped Low-Rank-Aware Candidate Scoring

**Files:**
- Modify: `auto_round/algorithms/transforms/svdquant/smooth.py`
- Modify: `auto_round/algorithms/transforms/svdquant/apply.py`
- Modify: `test/test_cpu/algorithms/test_svdquant_smooth.py`
- Modify: `test/test_cpu/algorithms/test_svdquant.py`

**Step 1: Add a failing independent numerical oracle**

For two tiny Linear modules sharing an input and `num_grids=4`, independently
calculate every candidate:

```python
scaled = [weight * scale.view(1, -1) for weight in weights]
stacked = torch.cat(scaled, dim=0)
low_rank, shared_down, stacked_up = truncated_svd(stacked, rank)
deployed_down = shared_down.to(low_rank_dtype)
deployed_up = stacked_up.to(low_rank_dtype)
deployed_low_rank = deployed_up.float() @ deployed_down.float()
split_low_rank = deployed_low_rank.split(output_sizes, dim=0)
residuals = [w - lr for w, lr in zip(scaled, split_low_rank)]
qdq_residuals = [residual_qdq(r, scheme) for r in residuals]
# Temporarily patch the grouped projections and run the parent evaluator.
error = (candidate_parent_output.float() - reference_parent_output.float()).square().sum()
```

Assert the production search selects the same candidate and runtime
`smooth == reciprocal(scale)`.

Also assert:

- one shared temporary SVD per candidate, not one per Linear;
- shared-down and split-up low-rank factor structure;
- low-rank dtype rounding affects scoring;
- each residual QDQ receives its resolved MXFP4 scheme;
- parent-module output determines the winner even when isolated Linear MSE would
  choose another candidate;
- temporary weights, hooks, and branches are restored after every candidate;
- later candidate wins equal finite layer error;
- invalid candidates are skipped; and
- all-candidate failure names the global Linear.

**Step 2: Confirm numerical test failure**

```bash
/home/user2/data/xixi/torch213-cu130-env/.venv/bin/python -m pytest \
  test/test_cpu/algorithms/test_svdquant_smooth.py \
  test/test_cpu/algorithms/test_svdquant.py -k 'smooth or candidate or qdq' -q
```

**Step 3: Implement grouped search scoring**

Add `SVDQuantSmoothSearcher` with explicit dependencies:

```python
searcher = SVDQuantSmoothSearcher(
    num_grids=config.smooth_num_grids,
    rank=config.rank,
    low_rank_dtype=resolved_dtype,
    residual_qdq=residual_module.rtn_qdq_residual,
)
```

The searcher receives a `SmoothSearchGroup`, its captured parent evaluation
data, and one resolved `ResidualQuantScheme` per projection. It concatenates
scaled weights, computes a shared decomposition, splits the output factor,
installs temporary QDQ residual/low-rank/input-smooth behavior, runs the parent
evaluation module, and restores state in `finally`. It returns only the winning
scale and diagnostics.

**Step 4: Integrate before final grouped decomposition**

For each discovered group:

```python
scale = searcher.search(group, calibration, ...) if smooth_enabled else torch.ones(...)
smooth = torch.reciprocal(scale)
weight_hats = [weight / smooth.view(1, -1) for weight in group.weights]
```

Pass `weight_hats` to the grouped final decomposition phase.

**Step 5: Run all SVDQuant tests**

```bash
/home/user2/data/xixi/torch213-cu130-env/.venv/bin/python -m pytest \
  test/test_cpu/algorithms/test_svdquant_smooth.py \
  test/test_cpu/algorithms/test_svdquant.py -q
```

Expected: PASS.

**Step 6: Checkpoint**

```bash
git commit -s -m "feat: score grouped SVDQuant smooth candidates"
```

---

### Task 6: Implement Grouped Final Residual Iteration

**Files:**
- Modify: `test/test_cpu/algorithms/test_svdquant.py`
- Modify: `auto_round/algorithms/transforms/svdquant/apply.py`
- Modify: `auto_round/algorithms/transforms/svdquant/wrapper.py` if a shared-down wrapper is required

**Step 1: Add grouped decomposition tests**

With `num_grids=N` and `residual_iters=R`, assert:

- smooth calibration performs `2N-1` temporary shared decompositions per group;
- selected-scale final calibration performs configured grouped residual
  iterations, subject to early stop;
- multi-Linear groups share the down factor and split the up factor by output
  channels;
- each final residual uses its own resolved quantization scheme;
- candidate search does not mutate final residual iteration state; and
- disabled smoothing still uses grouped final decomposition where an adapter
  defines a group.

Verify split wrappers reproduce the concatenated grouped reconstruction and
remain exportable through the existing per-Linear Nunchaku keys.

**Step 2: Run tests**

Expected: PASS with no default change from `residual_iters=1`.

**Step 3: Checkpoint**

```bash
git commit -s -m "feat: add grouped SVDQuant residual iteration"
```

---

### Task 7: Remove Stale API And Update Usage

**Files:**
- Modify: `scripts/quantize_flux_svdquant_nunchaku.py`
- Modify: `docs/svdquant_nunchaku_mxfp4_review.md`
- Modify: relevant SVDQuant runbook under `/home/user2/data/xixi`
- Modify: `test/test_cpu/core/test_pipeline_fail_fast.py`
- Modify: `test/test_cpu/utils/test_cli_usage.py`

**Step 1: Add compatibility-entry tests**

Verify compatibility constructors forward `smooth_num_grids`, never
`smooth_alpha`, and disabled smoothing remains calibration-free.

**Step 2: Remove stale references**

```bash
rg -n 'svdquant_smooth_alpha|smooth_alpha' auto_round scripts test docs
```

Remove only SVDQuant fixed-alpha references. Update commands to:

```text
--enable_svdquant_smooth --svdquant_smooth_num_grids 20
```

Use the neutral feature name "low-rank-aware smooth search" in help and logs.

**Step 3: Run compatibility tests**

```bash
/home/user2/data/xixi/torch213-cu130-env/.venv/bin/python -m pytest \
  test/test_cpu/core/test_pipeline_fail_fast.py \
  test/test_cpu/utils/test_cli_usage.py -k svdquant -q
```

Expected: PASS.

**Step 4: Checkpoint**

```bash
git commit -s -m "docs: document SVDQuant smooth grid search"
```

---

### Task 8: Full Verification And FLUX Smoke

**Files:**
- No code changes expected
- Output log: `/home/user2/data/xixi/autoround-flux-mxfp4-r32-smooth-smoke.log`

**Step 1: Run focused tests**

```bash
/home/user2/data/xixi/torch213-cu130-env/.venv/bin/python -m pytest \
  test/test_cpu/algorithms/test_svdquant_smooth.py \
  test/test_cpu/algorithms/test_svdquant_smooth_adapters.py \
  test/test_cpu/algorithms/test_svdquant.py \
  test/test_cpu/algorithms/test_awq.py \
  test/test_cpu/core/test_pipeline_fail_fast.py \
  test/test_cpu/utils/test_cli_usage.py -q
```

Expected: PASS, including unchanged AWQ behavior.

**Step 2: Run export regressions**

```bash
/home/user2/data/xixi/torch213-cu130-env/.venv/bin/python -m pytest \
  test/test_cpu/export/test_svdquant_nunchaku_export.py \
  test/test_cpu/export/test_svdquant_nunchaku_format.py -q
```

Expected: unchanged smooth tensor keys, shapes, and direction.

**Step 3: Run lint**

```bash
/home/user2/data/xixi/torch213-cu130-env/.venv/bin/python -m ruff check \
  auto_round/algorithms/transforms/svdquant \
  auto_round/cli/algorithms.py auto_round/cli/parser.py \
  auto_round/compressors/entry.py auto_round/autoround.py \
  test/test_cpu/algorithms/test_svdquant_smooth.py \
  test/test_cpu/algorithms/test_svdquant_smooth_adapters.py \
  test/test_cpu/algorithms/test_svdquant.py
```

Expected: no lint errors.

**Step 4: Run a reduced FLUX smoke on a free GPU**

Use one sample, one diffusion step, and `smooth_num_grids=4` first. Require at
least 18GB free disk and enough GPU headroom before launch. The command is the
existing FLUX SVDQuant command with:

```text
--enable_svdquant_smooth
--svdquant_smooth_num_grids 4
--svdquant_rank 32
--svdquant_residual_iters 1
```

Expected log evidence:

- seven candidates per search group;
- low-rank-aware candidate scoring;
- grouped parent-output evaluation;
- shared-down/split-up low-rank diagnostics;
- selected candidate diagnostics;
- final residual phase starts after scale selection;
- successful pipeline export and Nunchaku load.

**Step 5: Run the quality configuration**

After smoke success and resource confirmation, use:

```text
--svdquant_smooth_num_grids 20
--svdquant_rank 32
--svdquant_residual_iters <requested final iteration count>
--enable_svdquant_residual_early_stop
```

Report runtime separately for smooth candidate search and final residual
calibration. Do not claim quality success until Nunchaku generates a non-noise
image with the same prompt, seed, steps, and dimensions used for BF16/NVFP4
comparison.
