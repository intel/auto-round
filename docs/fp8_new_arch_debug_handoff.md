# FP8 Scheme Debug Handoff — `hengguo/new_ar_arch` (PR #1542)

> **Purpose**: Hand-off for the next AI/engineer continuing FP8 regression
> debugging on the new AutoRound architecture (`auto_round/compressors_new/`).
> Skip the discovery phase — this doc captures what's broken, what's already
> fixed, how to reach the HPU test environment, and the exact commands that
> reproduce results. Read this end-to-end **before touching code**.

---

## 1. Context

- Repo: `intel/auto-round`
- Branch: `hengguo/new_ar_arch`  (PR #1542 "Step1 new architecture for auto_round")
- Origin: `18ba254d  merge main` (last pushed commit; several fixes are still **uncommitted** locally — see §5)
- Two parallel code paths exist:
  - **Old arch**: `auto_round/compressors/` (baseline, `main` branch)
  - **New arch**: `auto_round/compressors_new/` + `auto_round/algorithms/` + `auto_round/context/` (this PR)
- Routing entry: `auto_round/compressors_new/entry.py::AutoRound.__new__` picks old vs new compressor class by config type.

### Key architectural differences vs old arch

- `BaseCompressor.__getattr__` delegates attribute access to **three contexts** in order: `quantize_config`, `model_context`, `compress_context`. Missing attribute on all three → `AttributeError`. Many latent bugs were caused by attributes existing on the wrong context.
- Per-sample batching logic moved from `BaseCompressor._get_batch_data` (old) to `BaseQuantizers._sampling_inputs` in `auto_round/algorithms/quantization/base.py` (new).
- `share_cache_keys = ('position_ids', 'cache_position', 'position_embeddings')` — values cached **once** (not per-sample), typically wrapped by hook as `[val]`. New arch needs to unwrap and pass through regardless of `len(indices)`.
- Immediate packing flag is `self.compress_context.is_immediate_packing` (not `is_immediate_saving`). Conflating the two caused the original FP8_STATIC RAM regression.

---

## 2. The FP8_STATIC Host-RAM Regression (primary motivating bug)

### Symptom
On HPU, new-arch `FP8_STATIC` (static W8A8-FP8) tuning leaked ~GBs of host RAM per block vs old arch — traced to HPU eager-pipeline host-side growth when the static-activation calibration path runs.

### Root causes & fixes (already in tree)

1. **Immediate packing trigger flag**  
   `auto_round/compressors_new/calib.py` ~L1031:
   ```python
   if self.compress_context.is_immediate_packing:  # was: is_immediate_saving
       ...
   ```
   Without this, packed weights were held in CPU RAM indefinitely.

2. **`tmp_dtype` missing**  
   `auto_round/compressors_new/calib.py` ~L1424: added
   ```python
   tmp_dtype = self.model_context.amp_dtype if self.model_context.amp else torch.float32
   ```
   Matches old arch.

### Status
**Fix #1 verified on HPU** (see §6). Do not revert these two pieces.

> **Note (2026-04-22):** An earlier version of this branch added `_needs_hpu_fp8_static_eager_guard` /
> `_maybe_disable_hpu_eager_pipeline` in `entry.py` that set `PT_HPU_EAGER_PIPELINE_ENABLE=0`
> for FP8_STATIC on HPU. This was **speculative** (never confirmed to reduce RAM) and has been
> **deleted**. The `is_immediate_packing` fix in calib.py is the real fix.

---

## 3. Latest performance report from user (the trigger for this handoff)

Reported CI (from `performance_ut.sh`, model `Qwen/Qwen3-0.6B`, scheme likely W4A16 per default):

```
Tuning Time (s)  : Current = 1192.5 | Baseline = 445.7  (+167.58%)  FAIL
Peak RAM  (GB)   : Current = 3.68   | Baseline = 4.05   (−9.14%)    FAIL (tolerance)
Peak VRAM (GB)   : Current = 1.29   | Baseline = 26.73  (−95.17%)   FAIL
Output Size (GB) : Current = 0.7114 | Baseline = 0.7114 (+0.00%)    PASS
```

### CRITICAL finding from direct HPU test (iters=20, same scheme/model)

Running the exact same binary on the HPU box (see §4) produced:

```
Quantizing model.layers.0   ... peak_ram=2.83GB  peak_vram=1.25GB   (block 0 only, expected)
Quantizing model.layers.1   ... peak_ram=2.86GB  peak_vram=26.43GB  ← HPU *is* being used
Quantizing model.layers.27  ... peak_ram=3.62GB  peak_vram=26.61GB
quantization tuning time 68.34s  for 20 iters → linear scale 200 iters ≈ 683s
real 1m24s  user 11m16s
```

**So on the actual HPU**, `Peak VRAM = 26.61 GB` (matches baseline 26.73 GB).  
The CI-reported `1.29 GB` equals the **first block only** (`model.layers.0`), before HPU allocation expands. This strongly suggests:

- `check_performance.py` in the CI pipeline is picking up the first `peak_vram` log line (block 0 = 1.25–1.29 GB) and missing later ones, **OR**
- The CI run crashed/exited after block 0 (VRAM never grew) but reported partial data as "success".

The tuning time gap (683 s estimate locally, 1192 s in CI) is real but smaller than the 2.7× the CI report suggests. Likely contributors:
- `torch.compile` recompile on every block (new-arch cache invalidation logic).
- Caching / `_sampling_inputs` overhead per step.
- CI docker env differences (no model warm cache, different HPU driver).

### Action for next agent
1. **Do NOT assume VRAM=1.29 GB is real**. First re-read `.azure-pipelines/scripts/performance/check_performance.py` — it likely parses log incorrectly. Fix the parser before believing the VRAM number.
2. Investigate the tuning-time gap separately from VRAM:
   - Profile `_resolve_block_forward`: does `torch.compile` actually hit the compiled path, or does `self` delegate through `__getattr__` to a context that returns `False` for `enable_torch_compile`?
   - Profile `_sampling_inputs`: per-sample tensor copies, shared-key unwrap path.
   - Compare `block_forward` call count / time per block between old and new arch under identical config.

---

## 4. HPU Test Environment (**use this to reproduce — do not reinvent**)

### 4.1 SSH chain (3 hops)

```
local  →  ssh tensorflow@clx5673.ra.intel.com
       →  ssh -i ~/.ssh/id_rsa_qun -J guest@146.152.224.86 sdp@100.81.152.55
       →  sshpass -p 1 ssh sdp@192.168.122.81           (host: kvm-01)
       →  docker exec AutoRoundDebug bash
```

- Final host `kvm-01` has **4× HL-225 (Gaudi2, 96 GB HBM each)**, driver 1.24.0, hl-1.23.0.
- Container `AutoRoundDebug`: Ubuntu 24.04, `torch 2.9.0+hpu.1.23.0.695`, `habana-torch-plugin 1.23.0.695`.
- Code lives at `/ar_work_space/auto-round-patched/` **inside** the container (NOT bind-mounted; must be copied via SSH).
- HF cache inside container: `~/.cache/huggingface/hub/` contains `models--Qwen--Qwen3-0.6B` and `Qwen3-1.7B` (already downloaded, no HF token needed).
- `auto_round` is **not** pip-installed in the container. Run via `PYTHONPATH=/ar_work_space/auto-round-patched` + `python3 -m auto_round ...`.

### 4.2 Helper script that works across all 3 hops

Saved at `/tmp/hpu_run.sh` (local). Base64-encodes the command to avoid shell escape hell:

```bash
#!/bin/bash
# Usage: /tmp/hpu_run.sh '<bash command to run inside AutoRoundDebug>'
set -e
CMD="$1"
B64=$(printf '%s' "$CMD" | base64 -w0)
ssh -o StrictHostKeyChecking=no -T tensorflow@clx5673.ra.intel.com \
  "ssh -o StrictHostKeyChecking=no -i ~/.ssh/id_rsa_qun -J guest@146.152.224.86 sdp@100.81.152.55 \
    \"sshpass -p 1 ssh -o StrictHostKeyChecking=no sdp@192.168.122.81 \\\"docker exec AutoRoundDebug bash -c 'echo $B64 | base64 -d | bash'\\\"\""
```

### 4.3 Syncing local code → container (no bind mount)

```bash
cd /home/hengguo/code/bug_fix/auto-round && \
tar -czf - auto_round/ | ssh tensorflow@clx5673.ra.intel.com \
  "ssh -i ~/.ssh/id_rsa_qun -J guest@146.152.224.86 sdp@100.81.152.55 \
    'sshpass -p 1 ssh -o StrictHostKeyChecking=no sdp@192.168.122.81 \
      \"docker exec -i AutoRoundDebug bash -c \\\"rm -rf /ar_work_space/auto-round-patched/auto_round && tar -C /ar_work_space/auto-round-patched -xzf - && echo OK\\\"\"'"
```

Repeat for `auto_round_extension/`, `setup.py`, `setup.cfg`, `pyproject.toml` if those change.

### 4.4 Canonical perf commands

**Short sanity (20 iters, ~90 s wall):**
```bash
/tmp/hpu_run.sh 'cd /ar_work_space/auto-round-patched && \
  export PYTHONPATH=/ar_work_space/auto-round-patched:$PYTHONPATH && \
  export HF_HUB_DISABLE_PROGRESS_BARS=1 TQDM_MININTERVAL=60 && \
  rm -rf /tmp/ar_test && \
  time python3 -m auto_round --model_name Qwen/Qwen3-0.6B --scheme W4A16 \
       --iters 20 --enable_torch_compile --device hpu --output_dir /tmp/ar_test 2>&1 | tail -60'
```

**Full perf run (mirrors `performance_ut.sh`, 200 iters, ~12 min):**  
Replace `--iters 20` with `--iters 200` above. Baseline for W4A16/Qwen3-0.6B is ~445 s tuning on this box.

**FP8_STATIC reproduction:** replace `--scheme W4A16` with `--scheme FP8_STATIC`.

---

## 5. Uncommitted local changes (as of this handoff)

```
auto_round/algorithms/quantization/base.py     — _sampling_inputs share_cache_keys unwrap
auto_round/utils/device.py                     — get_device_and_parallelism dict handling
auto_round/special_model_handler.py            — L207 use pre-extracted model_type (gemma4 FrozenDict)
test/test_cpu/export/test_export.py            — added autoround_old.post_init() in INT8_W8A8 test
```

Plus earlier, already-committed fixes:
- Deleted `auto_round/sign_sgd.py` (duplicate of `auto_round/algorithms/quantization/sign_round/sign_sgd.py`).
- Removed duplicate `from auto_round.sign_sgd import SignSGD` in `auto_round/compressors/base.py`.
- `calib.py`: `is_immediate_packing` flag + `tmp_dtype` definition.
- `entry.py`: Removed speculative `_maybe_disable_hpu_eager_pipeline` / `_needs_hpu_fp8_static_eager_guard` (never validated on HPU).
- `utils/device.py`: `get_device_and_parallelism` now handles `device=None` (fixes llmcompressor integration crash).

Push these before next CI run or CI logs will still show pre-fix behaviour.

---

## 6. What the next agent should do (ordered)

1. **Commit & push the uncommitted fixes in §5** so CI reflects current state.
2. **Fix `check_performance.py`** (in `.azure-pipelines/scripts/performance/`) — it is almost certainly reporting `peak_vram` from block 0 instead of the run max. Local HPU proof in §3 shows VRAM=26.6 GB is correct.
3. **Profile the real 2.7× tuning-time gap** with iters=200:
   - Add timing around `_resolve_block_forward` branches (compiled vs plain).
   - Log `self.compress_context.enable_torch_compile` once per block.
   - Compare `_sampling_inputs` CPU time between archs (new arch has extra conditional branches for share_cache_keys).
   - Check whether `torch.compile` cache is invalidated every block (`_invalidate_block_forward_cache` in `calib.py` at block boundary). Old arch reused the compiled function across blocks; new arch resets on `_dynamo.reset()` — confirm this is intentional and not the regression source.
4. Only then consider the algorithm-level code as suspect.

---

## 7. Known-good signals (sanity checks)

- `PT_HPU_LAZY_MODE=0` (eager mode) is active in this container.
- `torch.hpu.is_available() → True`, `device_count() → 4`.
- `from auto_round import __version__  →  0.13.0`.

---

## 8. Files to study first (highest-signal)

| Path | Why |
|---|---|
| `auto_round/compressors_new/entry.py` | routing, scheme pre-resolution |
| `auto_round/compressors_new/calib.py` | caching, block loop, immediate_pack, tmp_dtype |
| `auto_round/algorithms/quantization/base.py` | `_sampling_inputs`, `_get_block_outputs`, `_resolve_block_forward` |
| `auto_round/context/compress.py` / `model.py` | the three contexts `__getattr__` delegates to |
| `auto_round/compressors/base.py` (old arch) | ground-truth reference for every behaviour |
| `.azure-pipelines/scripts/performance/check_performance.py` | likely source of bogus VRAM=1.29 GB |

---

*Written 2026-04-22 during PR #1542 post-merge bug-fix session.*
