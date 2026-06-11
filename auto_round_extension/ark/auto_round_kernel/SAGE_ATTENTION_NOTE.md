# Sage Attention Notes

This note documents where the Sage attention implementation lives in this tree, how the int8 `QK` path is selected, and where `qscale`, `kscale`, and `vscale` are handled.

## Main files

- `wrapper/include/stla/xe_sage_fwd_kernel.hpp`
  - Kernel wrapper around the Sage mainloop and standard epilogue.
  - Handles per-batch/per-head tensor views, varlen offsets, cache offsets, and per-head scale pointer setup.
- `wrapper/include/stla/xe_sagev1_fwd_mainloop.hpp`
  - Sage v1 mainloop.
  - Computes `QK`, applies scale and masking, runs online softmax, and computes `PV`.
- `wrapper/include/sycl_tla_sdpa.hpp`
  - Defines `SageConfig`, including the concrete MMA types used by the Sage kernel.
- `wrapper/include/xpu_wrapper.hpp`
  - High-level wrapper that quantizes FP `Q/K/V` into int8 and builds the scale buffers for `sagev1`.
- `ark.cpp` and `__init__.py`
  - Python/C++ entrypoints.

## QK uses int8 MMA

The `stla` Sage mainloop is templated on `TiledMMAQK`, so the int8 choice is made outside the mainloop in `SageConfig`.

- In `wrapper/include/sycl_tla_sdpa.hpp:692`, `SageConfig` defines the default `QK` MMA operation as:
  - `XE_DPAS_TT<cute::gcd(SGTileQ, 8), int32_t, int8_t>`
- In `wrapper/include/sycl_tla_sdpa.hpp:717`, that operation is converted into:
  - `using TiledMMAQK = typename TiledMMAHelper<MMA_Atom<MMAOperation>, ...>::TiledMMA;`
- In `wrapper/include/stla/xe_sagev1_fwd_mainloop.hpp:303`, the mainloop instantiates:
  - `TiledMMAQK mma_qk{};`
- In `wrapper/include/stla/xe_sagev1_fwd_mainloop.hpp:422`, the actual `QK` GEMM is:
  - `cute::gemm(mma_qk, tSrQ, tSrK, tSrS);`

Conclusion:

- By default, Sage `QK` uses int8 DPAS MMA.
- This is a configuration choice from `SageConfig`, not hardcoded directly inside the mainloop.

## PV int8 is separate

The `PV` path has its own optional int8 implementation and is independent from `QK`.

- Quantized `PV` op is defined in `wrapper/include/stla/xe_sagev1_fwd_mainloop.hpp:159`.
- Quantized `PV` GEMM runs in `wrapper/include/stla/xe_sagev1_fwd_mainloop.hpp:496`.
- `vscale` is only used by this int8 `PV` path.

## Scale arguments in the Sage mainloop

The Sage mainloop arguments are defined in `wrapper/include/stla/xe_sagev1_fwd_mainloop.hpp:175`.

Relevant fields:

- `scale`: global softmax scale, converted to log2(e)-scaled form in `to_underlying_arguments`
- `scale_block_size`
- `qscale`
- `kscale`
- `vscale`

These arguments are forwarded through the launch options in `wrapper/include/sycl_tla_sdpa.hpp:651`.

## Where per-head scale pointers are computed

In `wrapper/include/stla/xe_sage_fwd_kernel.hpp`, the kernel wrapper computes the per-batch/per-head base pointers before calling the Sage mainloop.

- `scaleQ` setup: `xe_sage_fwd_kernel.hpp:266`
- `scaleK` setup: `xe_sage_fwd_kernel.hpp:269`
- `scaleV` setup: `xe_sage_fwd_kernel.hpp:272`

Layout assumptions:

- `qscale`: indexed as `[batch, q_head, q_block]`
- `kscale`: indexed as `[batch, kv_head, kv_block]`
- `vscale`: indexed as `[batch, kv_head, kv_block, d]`

The wrapper then passes those pointers into the mainloop call at `xe_sage_fwd_kernel.hpp:299`.

## Where qscale and kscale affect QK

The actual `QK` matmul runs first, producing an int32 accumulator tile. The scales are applied after the dot product and before softmax.

### Blockwise qscale

In `wrapper/include/stla/xe_sagev1_fwd_mainloop.hpp:391`, the code builds:

- `dq_scale = scaleQ[...] * params.scale`

This is the per-subgroup or per-block `Q` dequant scale combined with the global attention scale.

### kscale lookup

In `wrapper/include/stla/xe_sagev1_fwd_mainloop.hpp:412`, the code computes:

- `scalek_idx = K * get<1>(TileShapeQK{}) / params.scale_block_size`

This selects the `K` scale block for the current `K` tile.

### Applying qscale * kscale

There are two scale application modes:

1. `scale_block_size == 1`

- In `xe_sagev1_fwd_mainloop.hpp:439`, the code multiplies each score element by:
  - `scaleQ[row_idx] * scaleK[col_idx]`
- In `xe_sagev1_fwd_mainloop.hpp:445`, it then multiplies by `params.scale`.

This is effectively per-element dequantization of `QK`.

2. `scale_block_size > 1`

- In `xe_sagev1_fwd_mainloop.hpp:449`, the code computes:
  - `_scale = dq_scale * scaleK[scalek_idx]`
- In `xe_sagev1_fwd_mainloop.hpp:451`, each score is multiplied by `_scale`.

For the causal branch, the same pattern is used in `xe_sagev1_fwd_mainloop.hpp:455`.

Conclusion:

- `qscale` and `kscale` reconstruct the effective floating-point scale of the int8 `QK` scores.
- They are applied before masking and softmax.

## Where vscale is used

`vscale` is only used when `PV` is quantized to int8.

In `wrapper/include/stla/xe_sagev1_fwd_mainloop.hpp:503`, the code dequantizes the int8 `PV` accumulator with:

- `scaleV[scalev_idx]`
- multiplied by `1 / 127`

So:

- `qscale`: used for dequantizing `Q`
- `kscale`: used for dequantizing `K`
- `vscale`: used only for dequantizing `V` in the int8 `PV` path

## High-level entrypoint behavior

There are two distinct usage styles.

### 1. `sagev1`: scales are generated internally

Python entrypoint:

- `__init__.py:703`

C++ wrapper entrypoint:

- `wrapper/include/xpu_wrapper.hpp:1563`

Internal scale generation:

- `Q` quantization and `qscale` generation:
  - `xpu_wrapper.hpp:1499`
- `K` quantization and `kscale` generation:
  - `xpu_wrapper.hpp:1518`
- `V` quantization and `vscale` generation for the int8 `PV` variant:
  - `xpu_wrapper.hpp:1539`

Launches:

- `sdpa_impl_qks8_pvhalf(...)` at `xpu_wrapper.hpp:1555`
- `sdpa_impl_qks8_pvi8(...)` at `xpu_wrapper.hpp:1548`

This path starts from FP16/BF16 `Q/K/V`, quantizes them inside the wrapper, and then launches the int8 Sage kernel.

### 2. `sage` and `sage_pvi8`: scales are provided by the caller

Python low-level APIs:

- `sage`: `__init__.py:499`
- `sage_pvi8`: `__init__.py:592`

For `sage`, Python forwards:

- `qscale.data_ptr()` and `kscale.data_ptr()` in `__init__.py:571`

For `sage_pvi8`, Python forwards:

- `qscale.data_ptr()`, `kscale.data_ptr()`, `vscale.data_ptr()` in `__init__.py:681`

The expected `sage_pvi8` shapes are documented and validated in:

- `__init__.py:614`
- `__init__.py:643`

The C++ bridge then passes those buffers directly into:

- `ark::sdpa_impl_qks8_pvhalf(...)` in `ark.cpp:192`
- `ark::sdpa_impl_qks8_pvi8(...)` in `ark.cpp:210`

This path assumes the caller already prepared int8 tensors and matching scale buffers.

## Final summary

- Sage `QK` uses int8 DPAS MMA by default via `SageConfig`.
- `qscale` and `kscale` are applied after the int8 `QK` dot product to reconstruct floating-point scores before softmax.
- `vscale` is only used in the optional int8 `PV` path.
- `sagev1` computes the scales internally from FP inputs in `xpu_wrapper.hpp`.
- `sage` and `sage_pvi8` take the scale buffers directly from the caller.
