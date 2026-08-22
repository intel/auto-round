# BF16 Sparse SDPA Tile Shape Analysis

Date: 2026-08-18

## Scope

This note summarizes the tile-shape discussion for the native BF16 sparse SDPA kernel. The question is whether reducing the workgroup Q tile from 256 to 64 while increasing the GEMM head-dimension tile from 32 to 128 can reach current performance.

The analysis is based on:

- Current ARK sparse BF16 SDPA implementation:
  - `auto_round_kernel/wrapper/include/sycl_tla_sdpa_sparse.hpp`
  - `auto_round_kernel/wrapper/include/stla/xe_sparse_sdpa_fwd_mainloop.hpp`
  - `auto_round_kernel/sdpa_sparse_sdpa.cpp`
- SYCL-TLA tuning guide:
  - `/home/yiliu4/workspace/sycl-tla/media/docs/cpp/cute/12_intel_performance_guide.md`

## Current BF16 Sparse Tile

For head_dim=128 with `q_tile_override=256`, the current native BF16 sparse SDPA path uses:

```cpp
using ShapeQK  = Shape<_256, _32, _32>;
using ShapePV  = Shape<_256, _32, _32>;
using ShapeOut = Shape<_256, _128>;
using SubgroupLayoutQK = Layout<Shape<_16, _1, _1>>;
```

Interpretation:

| Item | Value |
|---|---:|
| Q rows per workgroup | 256 |
| K-token columns per QK tile | 32 |
| Head-dim chunk per QK loop | 32 |
| Head-dim loops for head_dim=128 | 4 |
| QK score tile per workgroup | 256 x 32 |
| Output tile per workgroup | 256 x 128 |
| Subgroups per workgroup | 16 |
| Work-items per workgroup | 256 |
| Q rows per subgroup | 16 |

The sparse API variant `sdpa_impl_bf16_sparse_sdpa_qtile256_row64k` passes `sparse_q_block_size=256`, so one sparse LUT row covers the full 256-token Q workgroup tile. The logical sparse K route block is 64 tokens, but the physical QK K-token tile is 32, so each selected K64 route block maps to two K32 micro-tiles.

## Proposed Shape

The proposed direction is to reduce M and increase the head-dimension K chunk:

```cpp
using ShapeQK = Shape<_64, _32, _128>;
```

Here, `K=128` means the GEMM inner dimension, i.e. the head-dimension chunk, not the token-axis K tile.

To keep the per-subgroup Q work comparable to the current kernel, the matching subgroup layout would likely be:

```cpp
using SubgroupLayoutQK = Layout<Shape<_4, _1, _1>>;
```

This keeps:

```text
Q rows per subgroup = 64 / 4 = 16
```

which matches the current:

```text
Q rows per subgroup = 256 / 16 = 16
```

Keeping 16 subgroups for `M=64` would give only 4 Q rows per subgroup and would also change the DPAS M shape, so it is not an apples-to-apples comparison.

## Apple-To-Apple Comparison

| Config | ShapeQK | Q rows/WG | K-token tile | Head-dim chunk | Head-dim loops | Subgroups/WG | Q rows/SG | Main benefit | Main risk |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| Current | `Shape<_256, _32, _32>` | 256 | 32 | 32 | 4 | 16 | 16 | Large Q tile, stable fragments | More head-dim loop iterations |
| Proposed | `Shape<_64, _32, _128>` | 64 | 32 | 128 | 1 | 4 | 16 | Fewer head-dim loops | Larger Q/K fragments and 4x more workgroups |

Current pseudocode:

```cpp
for q_tile in Q step 256:
  for selected k_token_tile in sparse_route:  // physical token tile = 32
    S[256, 32] = 0

    for d in head_dim step 32:                // 4 loops
      q_frag = load Q[256, 32]
      k_frag = load K[32, 32]
      S += q_frag @ k_frag.T

    P = softmax(S)
    O[256, 128] += P @ V[32, 128]
```

Proposed pseudocode:

```cpp
for q_tile in Q step 64:
  for selected k_token_tile in sparse_route:  // physical token tile = 32
    S[64, 32] = 0

    for d in head_dim step 128:               // 1 loop
      q_frag = load Q[64, 128]
      k_frag = load K[32, 128]
      S += q_frag @ k_frag.T

    P = softmax(S)
    O[64, 128] += P @ V[32, 128]
```

For the same 256-query region, the proposed shape launches four workgroups instead of one. The QK math is similar, but the execution shape is different: fewer head-dim loops per workgroup, more workgroups, less per-workgroup Q reuse, and larger Q/K fragments.

## Coexisting Fragment Estimate

The table below estimates logical fragment elements per subgroup. It is not an exact GRF count, but it shows the direction of register pressure. Prefetch fragments are excluded because `XE_PREFETCH_2D` has no destination register fragment.

Assumptions:

- Current: `ShapeQK = Shape<_256, _32, _32>`, `ShapePV = Shape<_256, _32, _32>`, `ShapeOut = Shape<_256, _128>`, 16 subgroups/WG.
- Proposed: `ShapeQK = Shape<_64, _32, _128>`, `ShapePV = Shape<_64, _32, _32>`, `ShapeOut = Shape<_64, _128>`, 4 subgroups/WG.
- Both keep 16 Q rows per subgroup.
- Copy fragments and MMA fragments are counted separately because both can be live around `reorder(copy_frag, mma_frag)` and `cute::gemm(...)`.

### QK Phase

| Fragment | Role | Current per SG | Proposed per SG | Change |
|---|---:|---:|---:|---:|
| `tQrQ` | Q copy fragment | `16 x 32 = 512` | `16 x 128 = 2048` | 4.0x |
| `tSrQ` | Q MMA fragment | `16 x 32 = 512` | `16 x 128 = 2048` | 4.0x |
| `tKrK` | K copy fragment | `32 x 32 = 1024` | `32 x 128 = 4096` | 4.0x |
| `tSrK` | K MMA fragment | `32 x 32 = 1024` | `32 x 128 = 4096` | 4.0x |
| `tSrS` | QK score accumulator | `16 x 32 = 512` | `16 x 32 = 512` | 1.0x |
| `tArA` | PV/O accumulator | `16 x 128 = 2048` | `16 x 128 = 2048` | 1.0x |
| `tA_max + tA_sum` | Softmax row state | `16 + 16 = 32` | `16 + 16 = 32` | 1.0x |
| **Approximate live sum** |  | **5664** | **14880** | **2.63x** |

### PV Phase

| Fragment | Role | Current per SG | Proposed per SG | Change |
|---|---:|---:|---:|---:|
| `tSrS / tArP` | Score/probability fragment | `512 + 512` | `512 + 512` | 1.0x |
| `tVrV` | V copy fragment, one V tile | `32 x 32 = 1024` | `32 x 32 = 1024` | 1.0x |
| `tArV` | V MMA fragment, one V tile | `32 x 32 = 1024` | `32 x 32 = 1024` | 1.0x |
| `tArA` | Full output accumulator | `16 x 128 = 2048` | `16 x 128 = 2048` | 1.0x |
| `tA_max + tA_sum` | Softmax row state | `32` | `32` | 1.0x |
| **Approximate live sum** |  | **5152** | **5152** | **1.0x** |

The pressure increase is concentrated in the QK phase. Reducing M from 256 to 64 does not reduce per-subgroup Q rows if the subgroup layout is adjusted from 16 subgroups to 4 subgroups. Increasing the head-dim chunk from 32 to 128 makes the Q/K copy and MMA fragments 4x wider.

## Why Subgroups Per Workgroup Change

The subgroup count changes only if we intentionally preserve the same per-subgroup Q work.

Current:

```cpp
ShapeQK = Shape<_256, _32, _32>;
SubgroupLayoutQK = Layout<Shape<_16, _1, _1>>;
```

```text
16 subgroups/WG * 16 lanes/subgroup = 256 work-items/WG
Q rows/subgroup = 256 / 16 = 16
```

Proposed apples-to-apples layout:

```cpp
ShapeQK = Shape<_64, _32, _128>;
SubgroupLayoutQK = Layout<Shape<_4, _1, _1>>;
```

```text
4 subgroups/WG * 16 lanes/subgroup = 64 work-items/WG
Q rows/subgroup = 64 / 4 = 16
```

If we kept 16 subgroups with `M=64`, each subgroup would cover only 4 Q rows. That would make the proposed kernel a different experiment: smaller per-subgroup work, different DPAS M behavior, and more overhead relative to useful compute.

## Expected Performance Direction

The SYCL-TLA performance guide gives the key tradeoff:

- Increasing K can reduce K-loop and 2D block load issue overhead.
- Increasing K can also increase GRF pressure because larger copy/MMA fragments are live.
- Too much GRF pressure can cause compiler spill and erase the benefit.

For this sparse BF16 SDPA kernel, `M64, headK128` is not expected to be a guaranteed win because:

1. It reduces head-dim loops from 4 to 1.
2. It launches 4x more workgroups for the same Q region.
3. It makes Q/K copy and MMA fragments 4x wider per subgroup.
4. Attention already keeps score, softmax, probability, V, and output accumulator fragments live.
5. The estimated QK-phase live fragment footprint increases by about 2.6x per subgroup.

## Recommendation

Do not assume `Shape<_64, _32, _128>` will reach current performance. It is worth testing only as an A/B if profiling shows the current kernel is dominated by head-dim loop or 2D load issue overhead and unitrace/compiler reporting shows no spill.

A safer first experiment is:

```cpp
ShapeQK = Shape<_128, _32, _64>;
```

This reduces the head-dim loop count from 4 to 2 while taking a smaller GRF-risk step than `headK=128`.

Validation should include:

1. Correctness against dense SDPA.
2. Kernel timing on the same topk, layout, dtype, and sequence length.
3. `unitrace -d -v` to check GRF count and spill/private memory per thread.
4. ComputeBasic and stall metrics if timing changes materially.

