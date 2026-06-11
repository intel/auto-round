# Sparse Sage Decode Plan

## Summary

This slice adds **true cached decode** support to the ARK sparse Sage path.

Chosen scope for v1:

- `seq_len_q == 1`
- explicit KV cache
- causal self-attention
- non-paged KV only
- low-level sparse decode API plus Sparge-style preprocess feeder

The implementation reuses the existing sparse Sage mainloop with `CachedKV` enabled and keeps the sparse metadata contract as `lut + valid_block_num`.

## Implementation

- Add a low-level `sage_sparse_decode(...)` path that consumes:
  - int8 `Q`
  - int8 current-step `K`
  - fp16/bf16 current-step `V`
  - int8 cached `K`
  - fp16/bf16 cached `V`
  - `qscale`
  - `kscale_cache` and `kscale` over the cache and current-step blocks
  - `lut + valid_block_num`
- Reuse `block_K` / `block_V` and `seq_len_kv_cache` in the sparse kernel launcher so cached sparse decode uses the same underlying cache plumbing as dense Sage.
- Keep paged KV and varlen rejected in this slice.
- Treat sparse metadata rows as operating on the logical concatenation of cache blocks followed by current-step blocks.
- Add a decode-specific preprocess helper that routes over concatenated cache + current KV, then splits quantized cache/current tensors for the cached sparse decode call.

## Tests

- Low-level sparse decode all-selected route matches dense masked decode exactly.
- Low-level sparse decode partial route matches dense masked decode exactly.
- Preprocess-generated sparse decode matches the preprocess-generated dense masked reference.
- Existing sparse prefill, preprocess prefill, and Qwen prefill smoke tests remain green.

## Assumptions

- v1 decode uses non-paged KV cache only.
- `seq_len_q > 1` decode is deferred.
- Preprocess decode routing is top-k only.
- Decode preprocess routes over the full visible cache + current KV space and does not rely on a separate token-level causal mask.
