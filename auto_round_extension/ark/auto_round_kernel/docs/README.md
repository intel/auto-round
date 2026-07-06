# Docs

- `SPARGE_ARK_XPU_INTEGRATION_PLAN.md`: implementation plan for adding a sparse Sage-style XPU kernel path in ARK that consumes precomputed SpargeAttn-style `lut + valid_block_num` metadata.
- `SPARSE_SAGE_PREFILL_PARTIAL_E2E_PLAN.md`: next-slice plan focused on prefill-only partial sparsity (`Sq == Skv`) and getting the public Python `sage_sparse(...)` path working end-to-end before expanding corner-case coverage.
- `SPARSE_SAGE_CAUSAL_PREFILL_PLAN.md`: causal-first plan for the prefill-only sparse Sage path, including the kernel-side causal contract and causal sparse-prefill validation.
- `SPARSE_SAGE_DECODE_PLAN.md`: implementation plan for adding `seq_len_q == 1` cached sparse decode support on top of the existing sparse Sage kernel stack.
- `SPARSE_SAGE_KERNEL_FIX_PLAN.md`: implementation plan for fixing the remaining non-contiguous sparse-row kernel limitation by moving sparse execution to a dedicated sparse forward mainloop.
- `SPARSE_PREFETCH_NOTES.md`: notes on why dense `K + Stages` prefetch works, why LUT-driven sparse traversal needs a different scheduler, and what a sparse-aware prefetch path would require.
- `SPARSE_DEV_FAST_BUILD_20260703.md`: fast rebuild path for sparse-kernel development on this node, including the focused `D128 + BF16` sparse-only dev target, measured rebuild times, and the exact commands to use.
- `SPARSE_HALFWIDTH_FULL_KERNEL_BENCH_20260704.md`: half-width `ShapeOut=[64,64]` sparse row-linear benchmark notes showing that the reduced PV output tile does not improve the full kernel.
- `SPARSE_PV_LIVE_STATE_REDUCTION_PLAN_20260704.md`: plan for reducing PV-path live state and spill pressure in the long-sequence sparse kernel.
- `SPARSE_QUERY_TILE_256_20260704.md`: notes on adding preprocess-side `query_tile_tokens=256` support and the first benchmark results for the generic `q_tile=256` sparse path.
- `SPARSE_QTILE256_ROW64K_COMPARE_20260706.md`: benchmark comparison between the older sparse implementations and the newer decoupled `q_tile=256 / q_block=256 / k_block=64` path.
- `SAGE_MAINLOOP_BODY_DIAGRAM.html`: visual note explaining what one `mainloop_body(...)` call does, with a concrete Wan D=128 tile example.
- `SPARSE_KERNEL_BENCH_PLAN.md`: implementation note and usage guide for the dedicated C++ sparse-kernel benchmark executable with Wan- and Flux-shaped presets for later kernel tuning.
- `SPARSE_ATTN_CLEAN_BRANCH.md`: clean-branch contract for rebuilding sparse attention on top of `main`, including what is kept, what is excluded, and the build/e2e/perf commands used to verify it.
- `SPARSE_TOPK_BENCH_PLAN.md`: runbook for benchmarking torch SDPA, `sagev1`, and sparse attention across a top-k sweep, including kernel-only and end-to-end sparse timings.
- `SPARSE_SAGE_KERNEL_STATUS.md`: current development status of the ARK sparse Sage kernel, including the dedicated sparse-mainloop status plus the latest build/test commands used for validation.
- `SPARGE_PREPROCESS_PORT_PLAN.md`: implementation plan for porting the Sparge preprocess side on XPU, using Triton XPU first and pure PyTorch/XPU fallback only when necessary.
- `WAN_SPARSE_ATTENTION_PLAN.md`: first-slice plan for patching Diffusers Wan self-attention through `WanAttention.processor` and routing supported self-attention calls to ARK sparse attention.
