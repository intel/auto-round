# 交接文档：BF16 原生稀疏 SDPA（sparge）— 2026-08-14

供后续开发接手的状态快照。以下内容均在 **Intel Battlemage XPU（24.4 GB, `intel_gpu_bmg_g21`）**、
oneAPI 2026.1、PyTorch 2.13.0+xpu 上验证。

## 1. 代码位置

- **分支 `feat/sparse-bf16-prefill-v2`** — 全部稀疏工作（INT8 SAGE **与** 原生 BF16/FP16 稀疏 SDPA）。
  **`main` 只有 INT8 SAGE 路径**（此前认为 `ark.cpp`/`sparse_attention.py` 为空是路径写错导致的误判；
  以仓库根为相对路径检查：`main` 有可用的 int8 `sparge_sage2_attn_meansim_topk_xpu`，
  位于 `auto_round_extension/ark/auto_round_kernel/sparse_attention.py`，1440 行）。
- 切换分支后 **每次都要重新编译**（`auto_round_kernel/ark-xbuild/` 里的 `.so` 只匹配编译它的分支）。

### 本交接中的关键改动
- BF16 稀疏 SDPA 已拆成独立的原生 BF16/FP16 路径（`SparseSDPAConfig`），不再复用 INT8 SAGE
  DPAS 路径。
- `benchmarks/bench_sparse_topk.py` 已加入稠密 `ark.sdpa` 基线，并输出 `speedup_vs_ark`。
- BF16 稀疏已选中 block microtile 做了 K64 vs K32 A/B。K32 是当前实验方向，本地重新编译后的
  `.so` 已恢复为 K32。
- `xe_sparse_sdpa_fwd_mainloop.hpp` 支持把一个 64-token 稀疏路由 block 展开为多个物理 K microtile；
  这让 K32 能继续沿用现有 64-token LUT 格式。
- Unitrace profile 显示 K64 回退来自 SEND/SBID 与 memory-write 压力。
- 已成功生成一张 FLUX.1-dev sparse BF16 topk 0.9 K32 图片，且无 fallback。

### 两条独立路径（勿混用）

| 路径 | 封装函数 | 配置 | DPAS |
|---|---|---|---|
| INT8 稀疏 | `sparge_sage2_attn_meansim_topk_xpu` / `sage_sparse` | `SparseSageConfig` | `XE_DPAS_TT<8, int, int8, int8, int>`（含 `static_assert(ElementQ==int8_t)`） |
| BF16/FP16 稀疏 | `sparge_sage2_attn_meansim_topk_xpu_sdpa` / `sage_sparse_sdpa` | `SparseSDPAConfig` | `XE_DPAS_TT<8, float, bf16, bf16, float>` |

**在 bf16 片段上复用 int8 DPAS 会导致设备挂起**（已通过拆分修复；见 `SPARSE_BF16_DPAS_HANG.md`）。

### 关键文件
- `auto_round_kernel/sdpa_sparse_sdpa.cpp` — BF16 启动器；`sdpa_sparse.cpp` — INT8 启动器。
- `auto_round_kernel/sparse_attention.py` — Python 封装 + `sparge_preprocess_topk`。
- `auto_round_kernel/sparge_preprocess_triton.py` — triton_xpu 预处理（含 torch 回退）。
- `wrapper/include/stla/xe_sparse_sdpa_fwd_mainloop.hpp`、`sycl_tla_sdpa_sparse.hpp` — 内核。
- `auto_round_kernel/sdpa.cpp` — **稠密** SYCL-TLA SDPA → Python `ark.sdpa(...)`
  （`xe_sdpa_fwd_mainloop.hpp`）。

## 2. 正确性 — 已验证

- **仓库 e2e 测试通过**（`test_sage_sparse_prefill_e2e.py`、`test_sparge_preprocess_topk_e2e.py`）。
- **topk=1（全 block 选中）== 稠密 SDPA**：与精确 CPU fp32 参考的最大误差 **0.0039**（bf16 舍入）。
  即"稠密门"测试。
- **INT8 == BF16 内核**：相同 Q/K/V 下选中的 block 完全一致（LUT 逐位相同），输出误差 <0.008；
  跨进程验证逐位一致。
- **预处理与内核均确定**（相同 QKV → 结果一致，进程内与跨进程均一致）。

## 3. 性能（bench `benchmarks/bench_sparse_topk.py`，seq 75000，40 头 × 128）

新增 `dense_ark_sdpa`（经 `ark.sdpa` 调用的稠密 SYCL-TLA SDPA）作为第三个稠密基线；每行新增
`speedup_vs_ark` 列。完整表格见 `BENCH_SPARSE_ARK_SDPA_20260813.md`。

| 模式 | topk | vs torch | vs sagev1 | vs ark_sdpa |
|---|---|---|---|---|
| dense_ark_sdpa | — | 1.5–1.6× | 0.82–0.87× | 1.0 |
| INT8 稀疏 qtile256（内核） | 0.5 / 0.25 / 0.125 | 3.5 / 6.9 / 13.2× | — | 2.2 / 4.3 / 8.2× |
| BF16 稀疏（内核） | 0.5 / 0.25 / 0.125 | 1.5 / 2.9 / 5.6× | — | 0.92 / 1.8 / 3.5× |

说明：`dense_ark_sdpa` 快于 torch SDPA 但慢于 `dense_sagev1`；BF16 稀疏内核仅在 topk ≤ 0.25 时
快于稠密 ark 内核。topk 0.5 处"1.5× 而非 2×"的上限源于 **SBID 分数板停顿**（内存 SEND 并发瓶颈；
INT8 因 K 尺寸减半而无此问题）——早期剖析见 `SPARSE_BF16_KERNEL_OPTIMIZATION_JOURNEY.md` /
`SPARSE_SDPA_KERNEL_OPTIMIZATION_OPTIONS.md`。

**2026-08-14 A/B 更新：** 将 BF16 稀疏 qtile256 已选中 block 的 microtile 从 K64 改为 K32 后，
kernel-only 延迟提升 **1.69–1.77×**。topk 0.5 从 1511 ms / **0.91× vs ark** 提升到
863 ms / **1.59× vs ark**。正确性 e2e 仍通过。见 `SPARSE_BF16_K32_AB_20260814.md`。

**2026-08-14 unitrace 更新：** 在 topk 0.5 下 profile K32 与 K64，确认 K64 回退主要是
memory/SEND scoreboard 压力，而不是 occupancy 下降。聚合 `SPARSESDPAFwdMainloop` GPU time 为
1684.6 ms（K32）对 3072.4 ms（K64），XVE active 64.7% 对 39.9%，XVE stall 35.1% 对 59.9%，
SBID stall 31.4% 对 55.7%。K64 的 SEND 指令增加 4.4×，GPU memory write 增加约 5.0×。
两边 SLM read/write 都是 0/0，因此 unitrace 没显示 SLM-backed spill；巨大的 LSC write 差异是当前
spill/pressure proxy。原始 profile 在 `unitrace_k32_compute_20260814/`、
`unitrace_k32_stalls_20260814/`、`unitrace_k64_compute_real_20260814/`、
`unitrace_k64_stalls_real_20260814/`。

## 4. FLUX.1-dev 端到端 — 关键坑

**必须用 `enable_sequential_cpu_offload()`，不要用 `enable_model_cpu_offload()`。**

- `enable_model_cpu_offload()` 按*组件*粒度搬运：整个约 46 GB 的 transformer 会被拉到 24.4 GB
  设备上（峰值约 24 GB，仅剩约 350 MB 余量）。稀疏启动会偶发 OOM
  （`OUT_OF_DEVICE_MEMORY` / `OUT_OF_RESOURCES`），甚至**重置 GPU**
  （`dmesg: exec queue reset` on `0000:ba:00.0` = dev5；即 `DEVICE_LOST` 错误）。
- Sequential offload 使峰值降到 **约 0.16 GB** → triton_xpu 稀疏稳定运行（0 回退）。
  代价：每步约 4 s 对 1 s（50 步生成约 4–5 分钟）。

### FLUX 图像质量结论
- 相同配置下，bf16 稀疏与 int8 稀疏**产出完全相同图像**（已验证，0.0 差异）。
- `topk=0.9` 实际丢弃 **12.5–16.7%** 的 block（路由块取整 + sim 阈值），而非 10% → 图像与稠密
  相差约 15–18。注意力质量覆盖率：topk 0.9 保留 **91.5%**、topk 0.5 保留 **64.6%** ——
  归一化后输出的偏移是 FLUX 注意力分散下的 topk 稀疏固有现象。
- **最新 K32 BF16 稀疏 topk 0.9 运行：** PNG 已保存到
  `benchmarks/results/flux_bf16_topk09_k32_20260814_112445/flux_bf16_qtile256_topk0.9_512_dev5.png`。
  配置：BF16 sparse qtile256、已选中 block K32、`topk=0.9`、512×512、50 steps、seed 0、
  `ZE_AFFINITY_MASK=5`。结果：wall **238.884 s**、sparsity **0.125**、calls **2850**、
  sparse_calls **2850**、runtime_fallbacks **0**。
- **待解决：运行间非确定性。** 相同配置（bf16 qtile256 topk0.5，seed 0）两次运行图像相差
  **48.5**（均值 97.6 对 101.3），而 topk 0.9 完全确定（0.0）。QKV、预处理、内核均已验证确定 →
  疑似扩散轨迹对其它微小扰动的放大，而非稀疏内核缺陷。做单次图像对比前建议复查。

### 开发工具（未跟踪）
- `examples/flux_gen_bf16_sweep.py` — offload-only 运行器（遍历 topk，保存 PNG + 汇总）。
- `tools/sweep_flux_bf16_topk.sh` — 单 GPU 封装（qtile256 环境变量）。**注意：`FLUX_SPARSE_KERNEL`
  现已可通过环境变量覆盖（默认 bf16）**；此前写死 bf16，导致"int8"运行实际是 bf16。
  `main` 的 `flux_sparse_patch.py` 仅 int8，且忽略 block-token 环境变量。
- `tools/bench_sparse_bf16_fp16_sweep.sh` — bench 扫描驱动。

## 5. 当前仓库状态（未提交）

- **分支**：`feat/sparse-bf16-prefill-v2`。
- **已修改（跟踪）**：`benchmarks/bench_sparse_topk.py`（新增 `dense_ark_sdpa` 模式 +
  `speedup_vs_ark` 列）、稀疏 SDPA 头文件中的 K32 route-block expansion —— 尚未提交。
- **未跟踪**：`examples/flux_gen_bf16_sweep.py`、`tools/sweep_flux_bf16_topk.sh`、
  `tools/bench_sparse_bf16_fp16_sweep.sh`。
- `.so`：`auto_round_kernel/ark-xbuild/auto_round_kernel_xpu.cpython-313-x86_64-linux-gnu.so`
  （特性分支构建）。`auto_round_kernel/xbuild/` 里有一份**过期的 main** `.so`（无碍；
  特性分支的 loader 用 `ark-xbuild`）。
- `benchmarks/results/` 已 gitignore → 下面的结果 CSV、PNG、文档均为本地。

## 6. 如何运行

```bash
# 重新编译（先 source /opt/intel/oneapi/setvars.sh --force）
cd auto_round_extension/ark/auto_round_kernel/ark-xbuild && cmake --build . -j 16

# 正确性测试
export ZE_AFFINITY_MASK=5 SAGE_ATTN_SPARSE_PREPROCESS_BACKEND=torch
.venv/bin/python auto_round_kernel/wrapper/test/test_sage_sparse_prefill_e2e.py
.venv/bin/python auto_round_kernel/wrapper/test/test_sparge_preprocess_topk_e2e.py

# FLUX 扫描（bf16 qtile256；SAGE 时设 FLUX_SPARSE_KERNEL=int8）
export ZE_AFFINITY_MASK=5 SAGE_ATTN_SPARSE_PREPROCESS_BACKEND=triton_xpu
export FLUX_SPARSE_KERNEL=bf16 FLUX_SPARSE_TOPKS="0.9 0.5" FLUX_RUN_DENSE=0
export FLUX_OUTPUT_DIR=benchmarks/results/flux_<stamp>
.venv/bin/python examples/flux_gen_bf16_sweep.py

# bench（需要特性分支 —— main 缺少 bf16 符号）
export ZE_AFFINITY_MASK=5 SAGE_ATTN_SPARSE_PREPROCESS_BACKEND=triton_xpu
.venv/bin/python benchmarks/bench_sparse_topk.py --dtype bf16 --seq-len 75000 \
  --topk 0.5 0.25 0.125 --tensor-layout HND NHD --q-tile-override 256 \
  --sparse-q-block-tokens 256 --sparse-k-block-tokens 64 --warmup 2 --iters 3 \
  --output-csv benchmarks/results/bench_<stamp>.csv
```

## 7. 待办 / 建议

1. **定位 topk-0.5 的跨运行非确定性**，再做单次图像对比（逐次抓取 QKV；稀疏算子已全部验证确定）。
2. **将 K32 BF16 稀疏 SDPA 实验整理成最终实现**（`SPARSE_BF16_K32_AB_20260814.md`）。unitrace
   A/B 说明应保持已选中 block microtile 为 K32，除非后续 compiler/assembly 优化能消除 K64 的
   write/SBID 压力。
3. 考虑让 FLUX 运行器文件名里的 `_512` 后缀跟随实际尺寸（1024×1024 运行被误标为 `_512`，纯外观问题）。
4. 决定是否提交 bench 的 `dense_ark_sdpa` 改动及开发工具（当前均未提交）。
