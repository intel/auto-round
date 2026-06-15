# Sparse Attention Clean Branch

This branch rebuilds the sparse-attention stack on top of `main` with two goals:

1. keep the original dense Sage code path as intact as possible
2. group sparse-specific behavior into parallel files and additive plumbing

## What This Branch Keeps

- sparse prefill and cached decode APIs:
  - `sage_sparse(...)`
  - `sage_sparse_decode(...)`
- sparse preprocess path:
  - Triton-XPU + torch fallback helpers
- sparse correctness and e2e coverage:
  - wrapper sparse prefill tests
  - preprocess replay tests
  - Qwen sparse prefill/decode smoke tests
- benchmark harnesses:
  - Python `bench_sparse_topk.py`
  - C++ `bench_ARK_XPU`
- diffusion demo integrations:
  - Wan sparse patching
  - Flux sparse patching
- sparse design/status docs

## What This Branch Intentionally Excludes

- generated benchmark outputs
- saved profile traces and logs
- generated images and videos
- build directories and compiled artifacts

The branch carries code and docs needed to reproduce those results, but not the generated outputs themselves.

## Implementation Shape

- Sparse behavior is primarily added through new files and additive wiring.
- The sparse mainloop keeps the current copy-based dense baseline in
  `xe_sparse_sagev1_fwd_mainloop.hpp` so sparse all-selected/prefix rows stay close to dense behavior.
- Original dense files are changed only where needed to:
  - register sparse entrypoints
  - route sparse launchers
  - expose Python bindings
  - include sparse-specific helpers

## Verification Commands

Build:

```bash
source /opt/intel/oneapi/setvars.sh
/home/yiliu7/workspace/venvs/ark/bin/cmake -B xbuild -DARK_BENCH=ON
/home/yiliu7/workspace/venvs/ark/bin/cmake --build xbuild --target auto_round_kernel_xpu bench_ARK_XPU -j 4
```

Sparse e2e:

```bash
source /opt/intel/oneapi/setvars.sh
/home/yiliu7/workspace/venvs/ark/bin/python wrapper/test/test_sage_sparse_prefill_e2e.py
/home/yiliu7/workspace/venvs/ark/bin/python wrapper/test/test_sparge_preprocess_topk_e2e.py
/home/yiliu7/workspace/venvs/ark/bin/python -m pytest -q ../test/test_qwen_sparse_prefill_e2e.py -s
/home/yiliu7/workspace/venvs/ark/bin/python -m pytest -q ../test/test_qwen_sparse_decode_e2e.py -s
```

Perf harness smoke:

```bash
source /opt/intel/oneapi/setvars.sh
./xbuild/bench_ARK_XPU --preset flux_single --pattern prefix --topk 1.0,0.5 --warmup 1 --iters 1
./xbuild/bench_ARK_XPU --preset wan_self --pattern prefix --topk 1.0,0.5,0.25 --warmup 1 --iters 1
/home/yiliu7/workspace/venvs/ark/bin/python ../test/bench_sparse_topk.py \
  --batch 1 --num-heads-q 4 --num-heads-kv 4 --seq-len 1024 --head-dim 128 \
  --topk 0.5 --warmup 1 --iters 1 --output-csv /tmp/bench_sparse_topk_clean_smoke.csv
```
