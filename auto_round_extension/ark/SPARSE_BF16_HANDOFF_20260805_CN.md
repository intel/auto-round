# Sparse BF16 交接文档（2026-08-06 更新）

> **说明：** 本文档是 2026-08-05 交接文档的更新版。旧版描述的是重构前的状态，并且认为
> BF16 benchmark 是失败的；这两点都已过时。BF16 sparse 路径已经从 INT8 sparse 路径中
> 干净地分离出来，且最近的 benchmark 运行可以成功完成。

## 范围

交接 `auto_round_extension/ark` 中 `sparse BF16` 的工作，以 `feat/sparse-bf16-prefill-v2`
分支上的 INT8/BF16 路径分离重构之后的状态为准。

内容包括：

- 最终架构（两条按精度分离的 sparse attention 路径）
- 公开 API
- 构建 / 验证 / 复现步骤
- 当前 benchmark 状态与已知注意事项

## 目标

两条相互独立的 sparse SDPA 路径，每种精度一条：

- **INT8** —— `sage_sparse` / `sparge_sage2_attn_meansim_topk_xpu`
- **BF16 / FP16** —— `sage_sparse_sdpa` / `sparge_sage2_attn_meansim_topk_xpu_sdpa`

此前 BF16 原生路径被混进了以 INT8 为中心的 SAGE kernel 路径（`sage_sparse_bf16` 把原生
Q/K 路由到 `SparseSageConfig` / `SPARSESAGEV1FwdMainloop`）。本次重构已移除该耦合。

## 当前环境

- 仓库：`/home/yiliu4/workspace/auto-round/auto_round_extension/ark`
- Python：`.venv/bin/python`
- PyTorch：`2.13.0+xpu`，`torch.version.xpu=20260000`
- oneAPI 编译器：`2026.1.1`（`/opt/intel/oneapi/compiler/2026.1`），MKL `2026.1`
- CMake XPU 构建目标：`intel_gpu_bmg_g21`，`ARK_SYCL_TLA=ON`，icx

## 架构（重构后）

**INT8 路径：** `sparse_attention.py:sage_sparse` → pybind `ark.cpp:sage_sparse` →
`sdpa_sparse.cpp`（`sdpa_impl_qks8_sparse_{d64,row_linear,qtile256_row64k}_pvhalf`）→
`SparseSageConfig` → `SPARSESAGEV1FwdMainloop`
（`wrapper/include/stla/xe_sparse_sagev1_fwd_mainloop.hpp`）。

- Q/K 为 `int8_t`（通过 qscale/kscale 反量化）；V/PV 为 fp16/bf16。
- Q*K MMA 固定为 int32/int8 DPAS；`SparseSageConfig` 内 `static_assert` 强制
  `ElementQ == int8_t`（仅支持 INT8）。

**BF16/FP16 原生路径：** `sparse_attention.py:sage_sparse_sdpa` → pybind
`ark.cpp:sage_sparse_sdpa` → `sdpa_sparse_sdpa.cpp`
（`sdpa_impl_{bf16,fp16}_sparse_sdpa_{d64,row_linear,qtile256_row64k}`）→
`SparseSDPAConfig` → `SPARSESDPAFwdMainloop`
（`wrapper/include/stla/xe_sparse_sdpa_fwd_mainloop.hpp`）。

- Q/K/V 为原生精度；`scale_block_size = 0`，无反量化 scale；softmax scale 直接应用。
- head_dim 64/128，q_tile_override 0/64/256（d64 / row_linear / qtile256_row64k）。

**共享部分：** `sparge_preprocess_topk`（torch 或 triton 的 preprocess 后端）、
`XeSparseSageFwdKernel`（`xe_sparse_sage_fwd_kernel.hpp`）、`SparseFMHAFwdEpilogue`。

**本次重构移除的内容：**

- `auto_round_kernel/sdpa_sparse_bf16.cpp`（由 `sdpa_sparse_sdpa.cpp` 取代）
- `sage_sparse_bf16`（pybind 绑定 + Python 别名；原先只是 `sage_sparse_sdpa` 的别名）
- `sparge_sage2_attn_meansim_topk_xpu_bf16`（端到端封装）
- `sdpa_impl_bf16_sparse_{d64,row_linear,qtile256_row64k}` 声明
- INT8 SAGE mainloop 中的原生 softmax-scale 分支

**验证阶段的修复：** `_query_tile_tokens_for_head_dim(64)` 现在返回 64（原来返回 128），使
head_dim 64 的 preprocess 默认值与 sparse e2e 封装保持一致，修复了导致
`test_sparge_preprocess_topk_e2e.py` 失败的既有 `tile_block_map` 元数据不一致问题。

## 公开 API 状态

BF16/FP16 sparse 对外暴露为：

- `sage_sparse_sdpa(query, key, value, lut, valid_block_num, ...)`
- `sparge_sage2_attn_meansim_topk_xpu_sdpa(query, key, value, ...)`（端到端）

INT8 sparse 仍为 `sage_sparse` / `sparge_sage2_attn_meansim_topk_xpu`。不再有
`sage_sparse_bf16`。

## 构建状态

XPU 扩展重建到 `auto_round_kernel/ark-xbuild`：

```bash
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
export MKLROOT=/opt/intel/oneapi/mkl/2026.1
cmake -S auto_round_kernel -B auto_round_kernel/ark-xbuild \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=icx \
  -DARK_XPU=ON -DARK_SYCL_TLA=ON -DARK_DNNL=OFF -DARK_JOINT_MATRIX=OFF \
  -DDPCPP_SYCL_TARGET=intel_gpu_bmg_g21
cmake --build auto_round_kernel/ark-xbuild -j 16
```

**重要：** `auto_round_kernel/CMakeLists.txt` 通过 `file(GLOB SRCS .../*.cpp)` 收集源文件，
且**没有** `CONFIGURE_DEPENDS`。新增/删除 `.cpp`（例如已删除的 `sdpa_sparse_bf16.cpp`）后，
**必须**先重新运行 `cmake -S ... -B ...` 再 `cmake --build`，否则旧构建仍会引用已删除的
文件，且旧的 `.so` 仍然导出旧符号。

## 复现 / 验证

```bash
cd /home/yiliu4/workspace/auto-round/auto_round_extension/ark
# 正确性（INT8 + BF16 + FP16）。设置 torch preprocess 后端，避免 triton 后端在本机
# 触发 XPU scatter/gather 越界断言。
export SAGE_ATTN_SPARSE_PREPROCESS_BACKEND=torch
.venv/bin/python auto_round_kernel/wrapper/test/test_sage_sparse_prefill_e2e.py
.venv/bin/python auto_round_kernel/wrapper/test/test_sparge_preprocess_topk_e2e.py

# benchmark（脚本已设置 SAGE_ATTN_SPARSE_PREPROCESS_BACKEND=torch、ZE_AFFINITY_MASK=6）
./tools/repro_sparse_bf16_sdpa_bench.sh
# CSV 默认输出：/tmp/bench_sparse_topk_bf16_dev6.csv
```

## Benchmark 状态（2026-08-06）

BF16 sparse benchmark **可以成功跑完**（`status: ok`）。完整结果见
`benchmarks/results/bf16_sparse_20260806_full.csv`。原生路径（`sparse_sdpa_bf16_*`，
经 `sage_sparse_sdpa`）相对 dense torch SDPA 的加速比：

| 用例 | 仅 kernel | e2e |
|---|---|---|
| seq 32768、topk 0.5 | 1.40–1.44x | 0.85–0.87x |
| seq 32768、topk 0.125 | 5.38–5.67x | 1.56–1.57x |
| seq 75600、topk 0.5 | 1.51–1.53x | 1.15x |
| seq 75600、topk 0.125 | 5.71–5.83x | 2.64–2.65x |

在中等规模（seq 32768、topk 0.5）下 e2e 主要受 preprocess 开销影响；在更大 seq_len 或
更低 topk 下则更快。这些数字与此前混合路径 `sparse_bf16_*` 一致，说明分离没有造成
性能回退。

## 已知注意事项

- **Preprocess 后端：** 使用 `SAGE_ATTN_SPARSE_PREPROCESS_BACKEND=torch`。triton
  preprocess 后端在本机会触发 PyTorch XPU scatter/gather 越界断言
  （`ScatterGatherKernels.cpp:233`）。复现脚本已设置 `torch`；`tools/run_bf16_sparse_bench.sh`
  默认是 `triton_xpu`，需要覆盖该变量。
- **原生路径 `scale_block_size == 0`：** `xe_sparse_sage_fwd_kernel.hpp` 中对
  `seq_*_pad` 除以 `scale_block_size` 做了保护，避免除零 UB；pad 只在
  `scale_block_size ? ... : nullptr` 三元表达式中使用。
- **仅 INT8 的 SAGE 配置：** `SparseSageConfig` 在编译期强制 `ElementQ == int8_t`。
  原生精度 BF16/FP16 请使用 `SparseSDPAConfig`。
- **过时文档：** 本文档（`SPARSE_BF16_HANDOFF_20260805.md`）取代重构前的交接文档；
  `SPARSE_BF16_STATUS_20260729.md` / `SPARSE_BF16_BENCH_20260805.md` 属于历史记录，
  指向旧构建和旧符号。

## 总结

BF16 原生 sparse 路径已与 INT8 SAGE 路径完全分离。构建接线、公开 API、正确性测试和
benchmark 模式均已就位；BF16 sparse 可以跑通并且快于 dense torch SDPA。主要运行时注意
事项是 preprocess 后端的选择（用 torch，不要用 triton）。后续工作可继续在 Flux/Wan
sparse 示例中通过 `sage_sparse_sdpa` 做性能调优或集成。
