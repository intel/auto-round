# SpargeAttn ARK XPU Integration Plan

## Summary

Implement a new ARK XPU sparse-attention path that serves as the kernel-side counterpart of SpargeAttn's `spas_sage2_attn_meansim_topk_cuda`, but scope v1 to consume precomputed sparse routing metadata instead of building the routing map inside ARK.

This v1 adds:

- a new low-level sparse Sage kernel API in ARK that consumes int8 `Q`, int8 `K`, fp16/bf16 `V`, `qscale`, `kscale`, sparse `lut`, and sparse `valid_block_num`
- a new Python convenience API that still accepts `lut + valid_block_num` directly
- a new sparse Xe Sage mainloop derived from the current dense Sage path in `sycl_tla_sdpa`

This v1 does not include:

- routing or block-map construction inside ARK
- sparse int8 `PV`
- fp8 `V`
- runtime `pv_threshold` pruning

Default implementation target:

- sparse int8 `QK` + fp16/bf16 `V`
- head dim `64` and `128`
- `HND` and `NHD` layouts
- causal and non-causal support
- grouped-query attention support matching current Sage constraints

## API and Binding Changes

Add a new low-level C++ API next to the current Sage entrypoints:

- `sdpa_impl_qks8_sparse_pvhalf(...)`

Kernel contract:

- inputs: `Q_ptr`, `K_ptr`, `V_ptr`, `O_ptr`, `mask`
- sparse metadata: `lut`, `valid_block_num`
- quantization metadata: `scale_block_size`, `qscale`, `kscale`
- standard stride/layout/head/shape args
- `softmax_scale`, `is_causal`, output/value dtype

Validation rules:

- `Q` and `K` must be int8
- `V` and `O` must be fp16 or bf16
- `mask` and `is_causal` remain mutually exclusive
- `head_dim` is limited to `64` and `128`
- `num_heads_q % num_heads_kv == 0`
- `lut` and `valid_block_num` shapes must match `batch`, `num_heads_q`, and the chosen sparse tile shape

Expose two pybind symbols:

- low-level binding: `sage_sparse`
- Python convenience API: `sage_sparse(...)`

Python API behavior:

- accepts `lut + valid_block_num` directly
- mirrors current `sage(...)` behavior for layout and stride normalization
- does not build routing metadata
- does not fall back to dense attention on invalid sparse metadata; it should fail fast

## Kernel and Launcher Changes

Extend the shared SDPA/Sage launch plumbing to carry sparse metadata.

Add fields to the shared launch `Options` struct for:

- `const int* lut`
- `const int* valid_block_num`
- `int num_q_blocks`
- `int num_k_blocks`

Do not overload `page_table` for sparse routing. Keep paged-KV and sparse routing separate.

Add a new launcher selection path parallel to the existing Sage prefill selection:

- `launch_sage_sparse_prefill_kernel_128(...)`
- `launch_sage_sparse_prefill_kernel_64(...)`

Reuse the same tile families as dense Sage where practical:

- `head_dim=128` path based on the current `_256 x 64 x 32` QK family
- `head_dim=64` path based on the current `_128 x 64 x 32` QK family

This preserves current scale semantics, output tiling, and epilogue compatibility.

## Sparse Mainloop Design

Create a new sparse Sage mainloop as a separate implementation rather than modifying the dense one in place.

Recommended structure:

- keep the existing kernel wrapper pattern from dense Sage
- add a new mainloop type, for example `SparseSAGEFwdMainloop`
- reuse the existing epilogue unchanged if possible

Sparse mainloop behavior:

- one workgroup still owns one `(batch, q_head, q_block, v_tile)` output tile
- before the K-loop:
  - locate the sparse row for the current `(batch, q_head, q_block)`
  - read `valid_block_num[row]`
  - if zero, skip work and rely on an epilogue-safe zero state
- replace the dense contiguous `for K in [blk_k0, blk_k1)` iteration with:
  - initialize `cur_block = 0`
  - iterate `i in [0, valid_block_num[row])`
  - `delta = lut[row, i]`
  - `cur_block += delta`
  - use `cur_block` as the selected logical K block

Per selected K block:

- load the K tile from the selected logical block
- load the matching V tile from the selected logical block
- run int8 `QK`
- apply `qscale * kscale * softmax_scale`
- apply mask and causal handling
- update online softmax state
- accumulate `P @ V`

Scale semantics remain identical to dense Sage:

- `qscale` is indexed per Q block
- `kscale` is indexed per selected K block

Sparse traversal must operate in logical block units, not token units.

V1 exclusions:

- do not combine sparse mode with KV-cache
- do not combine sparse mode with paged-KV
- do not combine sparse mode with `block_K` / `block_V`
- if any of those are requested with sparse mode, reject explicitly

## Sparse Metadata Contract

Lock the v1 sparse metadata format to match Sparge's kernel-ready format closely:

- `lut` shape: `[B, Hq, num_q_blocks, num_k_blocks]`, `int32`
- `valid_block_num` shape: `[B, Hq, num_q_blocks]`, `int32`
- `lut` is delta-encoded per sparse row
- only the first `valid_block_num[...]` entries in each row are consumed
- remaining entries are ignored

Interpretation rules:

- rows are indexed by query heads, not KV heads
- for GQA/MQA:
  - sparse row selection is per Q head
  - K/V access still maps through `head = head_q / head_group_q`
- for causal mode:
  - kernel still applies its own causal masking
  - sparse metadata may include causal-invalid blocks
  - callers are encouraged, but not required, to pre-prune them

Python API validation:

- validate rank and dtype of `lut` and `valid_block_num`
- validate `num_q_blocks = ceil(Sq / BLKQ)`
- validate `num_k_blocks = ceil(Skv / BLKK)`
- validate `valid_block_num <= num_k_blocks`
- validate that `lut` and `valid_block_num` live on the same XPU device as the tensors

## Test Plan

Add correctness tests for:

- `head_dim=64`, non-causal
- `head_dim=128`, non-causal
- `head_dim=64`, causal
- `head_dim=128`, causal
- both `fp16` and `bf16` V/O dtypes
- a GQA case where `num_heads_q != num_heads_kv`
- both `HND` and `NHD`

Reference behavior:

- build a dense token-level mask from `lut + valid_block_num`
- compare sparse kernel output against dense ARK Sage or SDPA using that equivalent mask
- use tolerance aligned with current Sage int8-QK expectations

Sparse metadata edge-case tests:

- `valid_block_num == 0` for some rows
- a single selected K block
- dense-equivalent sparse rows selecting all K blocks
- malformed `valid_block_num > num_k_blocks`
- malformed metadata shape mismatch
- invalid dtype for `lut` or `valid_block_num`

Scale regression:

- verify sparse kernel matches dense Sage when all K blocks are selected
- use identical `qscale`, `kscale`, `scale_block_size`, and `softmax_scale`
- this confirms sparse traversal changes only block selection, not quantization math

Performance sanity:

- add a benchmark or smoke test that confirms the sparse kernel launches successfully on XPU
- confirm runtime improves versus dense Sage when a small fraction of K blocks are selected
- do not require an exact speed target in v1

## Assumptions and Defaults

- v1 scope is kernel-only for sparse logic; ARK will not build routing maps
- v1 public surface includes both a low-level pybind binding and a Python convenience API
- the Python API accepts `lut + valid_block_num` directly, not dense block maps
- v1 supports only sparse int8 `QK` + fp16/bf16 `V`; sparse int8 `PV` is deferred
- v1 does not support:
  - KV-cache
  - paged-KV
  - `block_K` / `block_V` alternate sources
  - fp8 `V`
  - runtime `pv_threshold`
- sparse tile sizes should follow the current Sage launch families already used by ARK for `head_dim=64/128`
- correctness takes priority over aggressive sparse-specific optimization in v1; the first version should reuse dense Sage softmax and epilogue behavior as much as possible

