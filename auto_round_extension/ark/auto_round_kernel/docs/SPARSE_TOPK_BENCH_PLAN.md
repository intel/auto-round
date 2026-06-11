# Sparse Top-k Benchmark Runbook

## Summary

This benchmark compares the following implementations on XPU:

- `torch.nn.functional.scaled_dot_product_attention`
- `ark.sagev1(...)`
- `ark.sage_sparse(...)` with precomputed sparse metadata
- `ark.sparge_sage2_attn_meansim_topk_xpu(...)` end-to-end

The primary target shape is:

- batch `1`
- heads `40`
- sequence length `75600`
- head dim `128`

Defaults:

- non-causal
- `Hq == Hkv == 40`
- top-k sweep: `1.0, 0.75, 0.5, 0.25, 0.125`
- attempt dense baselines at the full shape and record failures explicitly

## What The Benchmark Records

For every row:

- implementation / mode
- batch, heads, sequence length, head dim
- dtype
- requested `topk`
- actual `selected_ratio`
- actual `selected_blocks_per_row`
- warmup count
- benchmark iteration count
- mean latency in ms
- status: `ok`, `oom`, or `error`
- note / error string

For sparse runs, the benchmark records two modes:

- `sparse_kernel_only`: preprocess once, then time only `sage_sparse(...)`
- `sparse_e2e`: time `sparge_sage2_attn_meansim_topk_xpu(...)`

## How To Run

Environment:

```bash
source /opt/intel/oneapi/setvars.sh
/home/yiliu7/workspace/venvs/ark/bin/python ../test/bench_sparse_topk.py
```

Full target run:

```bash
source /opt/intel/oneapi/setvars.sh
/home/yiliu7/workspace/venvs/ark/bin/python ../test/bench_sparse_topk.py \
  --batch 1 \
  --num-heads-q 40 \
  --num-heads-kv 40 \
  --seq-len 75600 \
  --head-dim 128 \
  --topk 1.0 0.75 0.5 0.25 0.125 \
  --warmup 5 \
  --iters 20
```

Recommended smoke run first:

```bash
source /opt/intel/oneapi/setvars.sh
/home/yiliu7/workspace/venvs/ark/bin/python ../test/bench_sparse_topk.py \
  --batch 1 \
  --num-heads-q 40 \
  --num-heads-kv 40 \
  --seq-len 1024 \
  --head-dim 128 \
  --topk 1.0 0.5 \
  --warmup 1 \
  --iters 2
```

## Interpretation

- `topk` is only the requested routing fraction.
- `selected_ratio` is the actual sparsity signal to compare across runs.
- `sparse_kernel_only` isolates sparse attention execution cost.
- `sparse_e2e` includes preprocess cost and is the real user-facing sparse path.
- Dense baselines at `S=75600` may fail due to memory pressure; that is an expected measurable outcome, not a benchmark error in itself.
